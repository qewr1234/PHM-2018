"""FAB 스트리밍 시뮬레이터: 웨이퍼를 처리 순서대로 흘려보내며 과거 계측만으로 예측하고 채점한다.

사건 순서 (웨이퍼 i 의 예측 시각 p_i: vm 모드 = 연마 종료, forecast 모드 = 연마 시작)
1. p_i 이전에 도착한 계측값을 검증해 이력에 넣고, 멤버(EWMA·칼만)·결합 가중치·컨포멀 구간을 갱신한다.
2. 재학습 시각이 되었으면 LightGBM 을 도착한 계측 웨이퍼로 다시 학습한다.
3. 웨이퍼 i 의 특징(정답 없는 특징 + 이력 특징)을 만들고 멤버 예측 → 결합 → 구간.
4. 계측 정책에 따라 i 를 계측할지 정한다. 계측하면 T_END + delay 에 정답이 도착한다.

기간 (시간 순 분할, 학습·테스트·검증 split 과 무관)
- 개발 기간: 0 ~ DEV_END_DAY 일. 하이퍼파라미터·정책 선택은 WARMUP_DAY 이후 개발 기간 채점으로만 한다.
- 배포 기간: DEV_END_DAY 일 이후. 설정을 고정하고 보고만 한다 (모델은 계속 도착하는 계측으로 온라인 갱신).
"""
import heapq
import time

import numpy as np
import pandas as pd

from fab_vm import HOUR, REGIMES, OnlineBlend, VMService

DEV_END_DAY = 30.0
WARMUP_DAY = 7.0
DEFAULT_DELAY_H = 1.0


POLICIES = ("all", "random", "periodic", "smart", "smart_spread", "hybrid")


def simulate(s, mode="vm", delay_h=DEFAULT_DELAY_H, policy="all", budget=1.0, cfg=None, seed=0, static=None):
    """스트림 하나를 끝까지 흘려보내고 웨이퍼별 기록(DataFrame)을 돌려준다.

    policy: all(전수 계측) | random(확률 budget) | periodic(조건별 1/budget 장마다 1장) |
            smart(불확실성 = 칼만 예측 표준편차 + 멤버 간 표준편차 가 최근 점수의 상위 budget 분위 이상이면 계측,
                  실제 계측 비율이 budget 을 따라가도록 분위를 조정 — Han 외 2025 의 분산 기반 동적 표본을 변형) |
            smart_spread(멤버 간 표준편차만) | hybrid(절반은 주기, 절반은 불확실성)
    """
    svc = VMService(s, mode, cfg, static)
    ptime = s.t_end if mode == "vm" else s.t_start
    u_rand = np.random.default_rng(seed).random(s.n)
    measured = np.zeros(s.n, bool)
    arrived = np.zeros(s.n, bool)
    info = {k: np.full(s.n, np.nan) for k in ("kal_sd", "spread", "age_h")}
    flags = {k: np.zeros(s.n, bool) for k in ("no_active", "stale", "disagree", "cold")}
    heap, lat = [], []
    cnt = {r: 0 for r in REGIMES}
    score_hist = {r: [] for r in REGIMES}
    q_level = {r: budget for r in REGIMES}
    for i in np.argsort(ptime, kind="stable"):
        p, r = ptime[i], s.reg[i]
        while heap and heap[0][0] < p:           # 1) p 이전에 도착한 계측
            _, j = heapq.heappop(heap)
            arrived[j] = svc.on_metrology(j)
        svc.maybe_retrain(p)                      # 2) 주기 재학습
        t0 = time.perf_counter()
        out = svc.on_wafer(i, p)                  # 3) 예측
        lat.append(time.perf_counter() - t0)
        for k in info:
            info[k][i] = out[k]
        for k in flags:
            flags[k][i] = out["flags"][k]
        # 4) 계측 결정
        if policy == "all":
            meas = True
        elif policy == "random":
            meas = u_rand[i] < budget
        elif policy == "periodic":
            meas = cnt[r] % max(int(round(1 / budget)), 1) == 0
            cnt[r] += 1
        elif policy in ("smart", "smart_spread", "hybrid"):
            sd = out["kal_sd"] if np.isfinite(out["kal_sd"]) else 10.0
            spread = out["spread"] if np.isfinite(out["spread"]) else 0.0
            u = spread if policy == "smart_spread" else sd + spread
            sh = score_hist[r]
            sub = budget / 2 if policy == "hybrid" else budget
            meas = u_rand[i] < sub if len(sh) < 20 else u >= np.quantile(sh[-200:], 1 - q_level[r])
            sh.append(u)
            if policy == "hybrid":
                # 절반은 주기 계측(조건별 2/budget 장마다 1장)으로 바닥을 깔고, 나머지 절반을 불확실성으로 고른다
                meas = meas or cnt[r] % max(int(round(2 / budget)), 1) == 0
                cnt[r] += 1
            q_level[r] = float(np.clip(q_level[r] + 0.02 * (sub - meas), 0.01, 0.99))
        else:
            raise ValueError(policy)
        if meas and np.isfinite(s.y[i]):
            measured[i] = True
            heapq.heappush(heap, (s.t_end[i] + delay_h * HOUR, i))

    rec = pd.DataFrame({"y": s.y, "pred": svc.pred, "width": svc.width, "measured": measured, "arrived": arrived,
                        "reg": s.reg, "day": s.day, "split": s.split, "outlier": s.outlier,
                        **info, **{f"flag_{k}": v for k, v in flags.items()}})
    for k, nme in enumerate(svc.names):
        rec[f"p_{nme}"] = svc.P[:, k]
    g = svc.gbm
    rec.attrs.update(latency_ms=float(np.mean(lat) * 1000),
                     gbm_fit_s=float(np.mean(g.fit_seconds)) if g is not None and g.fit_seconds else float("nan"),
                     gbm_fits=len(g.fit_seconds) if g is not None else 0,
                     weights={r: dict(zip(svc.names, np.round(svc.blend.w[r], 3).tolist())) for r in REGIMES},
                     measured_rate=float(measured.mean()))
    return rec


# ==============================================================================================
# 채점
# ==============================================================================================
def period_mask(rec, period):
    ok = ~rec["outlier"].to_numpy() & np.isfinite(rec["y"].to_numpy())
    d = rec["day"].to_numpy()
    if period == "dev":
        return ok & (d >= WARMUP_DAY) & (d < DEV_END_DAY)
    if period == "deploy":
        return ok & (d >= DEV_END_DAY)
    if period == "all":
        return ok
    if period == "after_warmup":
        return ok & (d >= WARMUP_DAY)
    raise ValueError(period)


def mse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean((a - b) ** 2))


def score(rec, period, col="pred", extra=None):
    m = period_mask(rec, period)
    if extra is not None:
        m = m & extra
    y, p = rec["y"].to_numpy(), rec[col].to_numpy()
    n_nan = int((m & ~np.isfinite(p)).sum())
    m = m & np.isfinite(p)  # 이력이 전혀 없는 첫 웨이퍼들(콜드 스타트)은 예측이 없어 뺀다
    out = {"mse": mse(p[m], y[m]), "n": int(m.sum()), "n_nan": n_nan}
    for r in REGIMES:
        g = m & (rec["reg"].to_numpy() == r)
        out[f"mse_{r}"] = mse(p[g], y[g])
    return out


def coverage(rec, period):
    m = period_mask(rec, period) & np.isfinite(rec["width"].to_numpy())
    y, p, w = rec["y"].to_numpy()[m], rec["pred"].to_numpy()[m], rec["width"].to_numpy()[m]
    return {"coverage": float(np.mean(np.abs(y - p) <= w)), "mean_half_width": float(np.mean(w)), "n": int(m.sum())}


def r2r_errors(rec, period, col):
    """연마 전 예측 f 로 연마 시간을 정하면(목표 제거량 / f) 실제 제거량의 상대 오차 = y/f - 1 (Preston: 제거량 ∝ 시간)."""
    m = period_mask(rec, period) & np.isfinite(rec[col].to_numpy())
    e = rec["y"].to_numpy()[m] / rec[col].to_numpy()[m] - 1
    return {"rms_pct": float(np.sqrt(np.mean(e ** 2)) * 100), "within_2pct": float(np.mean(np.abs(e) <= 0.02)),
            "within_5pct": float(np.mean(np.abs(e) <= 0.05)), "n": int(m.sum())}  # 예측이 없는 웨이퍼는 빼고 n 에 표시


def replay_blend(s, rec, delay_h, blend_cfg, mode="vm"):
    """저장한 멤버 예측(p_*)으로 결합만 다시 흉내 낸다 (멤버 재학습 없이 결합 방식 비교용, 인과 순서 유지)."""
    names = [c[2:] for c in rec.columns if c.startswith("p_")]
    P = rec[[f"p_{n}" for n in names]].to_numpy()
    blend = OnlineBlend(names, **blend_cfg)
    ptime = s.t_end if mode == "vm" else s.t_start
    arr = np.where(rec["arrived"].to_numpy(), s.t_end + delay_h * HOUR, np.inf)
    arr_order = np.argsort(arr, kind="stable")
    k = 0
    idx = {r: [] for r in REGIMES}
    since = {r: 0 for r in REGIMES}
    out = np.full(s.n, np.nan)
    for i in np.argsort(ptime, kind="stable"):
        while k < len(arr_order) and arr[arr_order[k]] < ptime[i]:
            j = arr_order[k]
            k += 1
            rj = s.reg[j]
            idx[rj].append(j)
            since[rj] += 1
            if since[rj] >= blend.refit_every:
                ids = np.asarray(sorted(idx[rj], key=lambda q: s.t_end[q]))
                blend.refit(rj, P[ids], s.y[ids])
                since[rj] = 0
        r = s.reg[i]
        yy = s.y[idx[r]] if idx[r] else np.array([])
        lo, hi = (yy.min(), yy.max()) if len(yy) else (np.nan, np.nan)
        out[i] = blend.combine(r, P[i], lo, hi)
    return out
