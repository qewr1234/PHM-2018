"""neighbors: 참조 웨이퍼 정답으로 만드는 이웃 특징의 확장 (feature_fn, 폴드마다 학습 폴드 안에서만 계산).

아이디어
- 기존 NB_*/same_*/cross_* 가 신호의 대부분을 담으므로, 같은 정보를 더 좋은 형태로 준다.
- 가족 (모두 레짐 평균 대비 편차(dev)를 쓴다)
  K: 시간 가우시안 커널 가중 평균 (같은 레짐 / 같은 그룹의 다른 스테이지 / 다른 챔버 그룹), 대역폭 여러 개
  E: 지수 가중 이동평균 (반감기: 시간·웨이퍼 수), 인과(과거)·역인과(미래) 와 그 차(국소 기울기), 같은 레짐 + 교차 스테이지
  L: 트라이큐브 커널 가중 국소 직선 (2·6·24 h 창) 의 현재 시점 추정치와 기울기
  D: 상태 변화량 - 시간상 바로 앞·뒤 참조 웨이퍼 대비 이 웨이퍼의 ACTIVE_SEC, P·V 적분, 슬러리, 에어백 압력,
     드레서·패드 사용량 차 ("마지막으로 측정된 웨이퍼 이후 무엇이 달라졌나"; 그 웨이퍼의 편차는 same_prev1/next1 에 이미 있음)
  S: 같은 드레서 수명 안에서만 찾은 이웃 (시간 커널, 드레서 사용량 커널, 수명 평균)
  O: 같은 웨이퍼 다른 스테이지의 편차와 시간 간격

누수 교훈 (중요)
- 참조 행을 leave-one-out 으로만 처리하면 넓은 커널(24 h, 드레서 수명 전체)에서 이웃 집합이 거의 같아
  LOO 평균이 자기 정답의 단조 함수가 된다 (수명 안 Spearman(S_life_mean, y) = -1.000). 트리는 그 순서를 외워
  lgb CV 7.72 → 8.58 로 크게 나빠졌다 (all_loo 변형, 앙상블 7.83).
- 해결: 참조 행은 inner 5-fold 로 나눠 자기 폴드 밖의 참조 행만으로 계산 (Spearman -0.1), 참조가 아닌 행은 참조 전체로 계산.
  cmp_data.neighbor_features 의 k 가 작은 NB_* 는 이웃 집합이 행마다 달라 이 문제가 약하다.

시도한 것 (5-fold CV 앙상블, fold_seed=0, 기준선 7.596; 특징 수는 기본 112 포함)
- all_loo  (전 가족, LOO 만, lgb 만 사용)          7.829 (+0.233, 누수)
- all      (전 가족 K E L D S O, 186)               7.470 (-0.126)
- all_lgbonly (같은 특징을 lgb 멤버만)               7.505 (-0.091; compact 멤버도 보게 하는 편이 0.035 좋음, lgb_small 8.30→7.81)
- no_E     (K L D S O, 150)                         7.487 (-0.109)
- lean     (K L S O, 138)                           7.670 (+0.074, 나빠짐 → 커널 평균 가족은 기존 NB_* 와 중복)
- DO       (D O, 126)                               7.467 (-0.129)
- EDO      (E D O, 162)                             7.427 (-0.169)  ← 최종 (가장 낮은 CV)
- EDO, fold_seed=1                                  7.458 (같은 폴드 기준선 7.510 → -0.052; 테스트 6.773, 검증 6.688)
핵심은 D(상태 변화량) 가족: 이웃의 정답만이 아니라 "이웃과 나의 공정 상태 차이" 를 같이 주는 것.

최종 (EDO): CV 앙상블 7.427, 공식 테스트 6.839, 검증 6.686 (기준선 7.596 / 6.969 / 6.850).
계산은 폴드당 약 2 초라 캐시하지 않는다.

실행: python cmp2016/exp/neighbors.py            # 최종 변형만 → cmp2016/exp/neighbors.json
      python cmp2016/exp/neighbors.py --all      # 탐색 변형 전부
      python cmp2016/exp/neighbors.py --variants EDO --seed 1 --outdir DIR
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 다른 실험과 CPU를 나눠 쓸 때 OpenMP 스핀 대기로 LightGBM이 느려지는 것을 막는다 (lightgbm import 전에 설정).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import get_table, run, summary  # noqa: E402
from cmp_data import TARGET, regime  # noqa: E402

FAMILIES = ("K", "E", "L", "D", "S", "O")
# 시간 커널 대역폭(시간): 소스별
KERNEL_BW = {"same": (0.25, 1.0, 4.0, 24.0), "cross": (1.0, 4.0, 24.0), "xgrp": (4.0, 24.0)}
EWM_HOURS = (0.5, 2.0, 8.0)      # 반감기(시간)
EWM_COUNT = (2, 5, 15)           # 반감기(웨이퍼 수)
TREND_BW = (2.0, 6.0, 24.0)      # 국소 직선 창 반폭(시간)
DELTA_COLS = ["ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum", "SLURRY_FLOW_LINE_C_s0_mean",
              "MAIN_OUTER_AIR_BAG_PRESSURE_s0_mean", "USAGE_OF_DRESSER_start", "USAGE_OF_POLISHING_TABLE_start"]
LIFE_TIME_BW = (4.0, 24.0)       # 같은 드레서 수명 안 시간 커널
LIFE_USAGE_BW = (15.0, 50.0)     # 드레서 사용량 커널
MIN_W = 1e-3                     # 커널 가중 합이 이보다 작으면 결측


def dresser_life(table):
    """챔버 그룹별로 드레서 사용량 카운터가 초기화된 횟수 = 드레서 수명 id (정답 미사용)."""
    life = np.zeros(len(table), int)
    for g, x in table.groupby("GROUP"):
        x = x.sort_values("T_START")
        u = x["USAGE_OF_DRESSER_start"].to_numpy(float)
        resets = np.concatenate([[0], (np.diff(u) < -50).astype(int)])
        life[x.index.to_numpy()] = np.cumsum(resets)
    return life


def _wmean(w, v):
    sw = w.sum(axis=1)
    m = (w @ v) / np.maximum(sw, 1e-12)
    m[sw < MIN_W] = np.nan
    return m, sw


def _wlinear(w, dt, v):
    """가중 최소제곱 직선 (dt=0 기준): 절편(현재 시점 추정)과 기울기."""
    S0, S1, S2 = w.sum(1), (w * dt).sum(1), (w * dt ** 2).sum(1)
    T0, T1 = w @ v, (w * dt) @ v
    det = S0 * S2 - S1 ** 2
    ok = (S0 >= 3 * MIN_W) & (det > 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        icpt = np.where(ok, (S2 * T0 - S1 * T1) / det, np.nan)
        slope = np.where(ok, (S0 * T1 - S1 * T0) / det, np.nan)
    # 창 안 이웃 범위로 잘라 외삽 튐을 막는다
    pos = w > 0
    vmin = np.where(pos, v[None, :], np.inf).min(1)
    vmax = np.where(pos, v[None, :], -np.inf).max(1)
    icpt = np.clip(icpt, vmin, vmax)
    return icpt, slope


INNER_FOLDS, INNER_SEED = 5, 12345


def _compute(table, idx_all, cand_mask, families, prefix, cols, ctx):
    """idx_all 행들의 특징을 cand_mask 참조 행들로 계산해 cols 에 채운다 (idx 와 cand 가 겹치면 자기 자신은 뺀다)."""
    reg, dev, hours, life, dr, state = ctx
    regs = list(np.unique(reg))

    def put(name, idx, val):
        cols.setdefault(prefix + name, np.full(len(table), np.nan))[idx] = val

    for r in regs:
        idx = idx_all[reg[idx_all] == r]
        if len(idx) == 0:
            continue
        sources = {"same": [r], "cross": [q for q in regs if q != r and q[0] == r[0]],
                   "xgrp": [q for q in regs if q[0] != r[0]]}
        for sname, sregs in sources.items():
            if not sregs:
                continue
            cand = np.flatnonzero(cand_mask & np.isin(reg, sregs))
            if len(cand) < 5:
                continue
            dt = hours[idx][:, None] - hours[cand][None, :]   # > 0 이면 참조가 과거
            notself = idx[:, None] != cand[None, :]           # 자기 자신 제외
            dv = dev[cand]
            if "K" in families:
                for h in KERNEL_BW[sname]:
                    w = np.exp(-0.5 * (dt / h) ** 2) * notself
                    m, sw = _wmean(w, dv)
                    put(f"K_{sname}_g{h:g}", idx, m)
                    if h == KERNEL_BW[sname][1]:
                        put(f"K_{sname}_n{h:g}", idx, sw)
            if "E" in families and sname in ("same", "cross"):
                past, fut = (dt > 0) & notself, (dt < 0) & notself
                adt = np.minimum(np.abs(dt), 1e4)
                for hl in EWM_HOURS:
                    p, _ = _wmean(np.where(past, 0.5 ** (adt / hl), 0.0), dv)
                    f, _ = _wmean(np.where(fut, 0.5 ** (adt / hl), 0.0), dv)
                    put(f"E_{sname}_h{hl:g}_past", idx, p)
                    put(f"E_{sname}_h{hl:g}_fut", idx, f)
                    put(f"E_{sname}_h{hl:g}_slope", idx, f - p)
                # 웨이퍼 수 기준: 과거·미래 각각 시간순 순위에 반감기를 건다
                rank_p = np.argsort(np.argsort(np.where(past, dt, np.inf), axis=1), axis=1)
                rank_f = np.argsort(np.argsort(np.where(fut, -dt, np.inf), axis=1), axis=1)
                for hl in EWM_COUNT:
                    p, _ = _wmean(np.where(past, 2.0 ** (-rank_p / hl), 0.0), dv)
                    f, _ = _wmean(np.where(fut, 2.0 ** (-rank_f / hl), 0.0), dv)
                    put(f"E_{sname}_c{hl}_past", idx, p)
                    put(f"E_{sname}_c{hl}_fut", idx, f)
                    put(f"E_{sname}_c{hl}_slope", idx, f - p)
            if "L" in families and sname == "same":
                for H in TREND_BW:
                    u = np.abs(dt) / H
                    w = np.where(u < 1, (1 - u ** 3) ** 3, 0.0) * notself
                    est, slope = _wlinear(w, dt, dv)
                    put(f"L_est_{H:g}", idx, est)
                    put(f"L_slope_{H:g}", idx, slope)
            if "D" in families and sname == "same":
                for dname, mask, sign in (("prev", (dt > 0) & notself, 1.0), ("next", (dt < 0) & notself, -1.0)):
                    dd = np.where(mask, sign * dt, np.inf)
                    j = dd.argmin(axis=1)
                    ok = np.isfinite(dd[np.arange(len(idx)), j])
                    delta = state[idx] - state[cand[j]]
                    delta[~ok] = np.nan
                    for c, cname in enumerate(DELTA_COLS):
                        put(f"D_{dname}_{DELTA_SHORT[cname]}", idx, delta[:, c])
            if "S" in families and sname == "same":
                same_life = (life[idx][:, None] == life[cand][None, :]) & notself
                for h in LIFE_TIME_BW:
                    m, _ = _wmean(np.exp(-0.5 * (dt / h) ** 2) * same_life, dv)
                    put(f"S_time_g{h:g}", idx, m)
                du = dr[idx][:, None] - dr[cand][None, :]
                for b in LIFE_USAGE_BW:
                    m, _ = _wmean(np.exp(-0.5 * (du / b) ** 2) * same_life, dv)
                    put(f"S_usage_g{b:g}", idx, m)
                m, sw = _wmean(same_life.astype(float), dv)
                put("S_life_mean", idx, m)
                put("S_life_n", idx, sw)


DELTA_SHORT = {"ACTIVE_SEC": "ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum": "PV", "SLURRY_FLOW_LINE_C_s0_mean": "SLURRY",
               "MAIN_OUTER_AIR_BAG_PRESSURE_s0_mean": "AIRBAG", "USAGE_OF_DRESSER_start": "DRESSER",
               "USAGE_OF_POLISHING_TABLE_start": "PAD"}


def make_feature_fn(families=FAMILIES, prefix="NB_", inner_folds=INNER_FOLDS):
    """families 에 든 가족만 만드는 feature_fn 을 돌려준다. prefix 가 'NB_' 이면 compact 멤버도 본다.

    누수 방지: 참조 행은 leave-one-out 만으로는 부족하다. 넓은 커널(24 h, 드레서 수명 전체)에서는 이웃 집합이
    거의 같아 LOO 평균이 자기 정답의 단조 함수가 되고 트리가 그 순서를 외운다 (lgb CV 7.72 → 8.58 로 악화).
    그래서 참조 행은 inner K-fold(inner_folds) 로 나눠 자기 폴드 밖의 참조 행만으로 계산하고,
    참조가 아닌 행(held-out·테스트·검증)은 참조 행 전체로 계산한다.
    """
    families = set(families)

    def fn(table, ref_mask):
        ref_mask = np.asarray(ref_mask, bool)
        reg = regime(table)
        y = table[TARGET].to_numpy(float)
        mu = {r: y[ref_mask & (reg == r)].mean() for r in np.unique(reg)}
        dev = y - np.array([mu[r] for r in reg])
        hours = table["T_START"].to_numpy(float) / 3600.0
        ctx = (reg, dev, hours, dresser_life(table), table["USAGE_OF_DRESSER_start"].to_numpy(float),
               table[DELTA_COLS].to_numpy(float))
        cols = {}
        _compute(table, np.flatnonzero(~ref_mask), ref_mask, families, prefix, cols, ctx)
        if inner_folds and inner_folds > 1:
            inner = np.random.RandomState(INNER_SEED).randint(0, inner_folds, len(table))
            for f in range(inner_folds):
                own = ref_mask & (inner == f)
                _compute(table, np.flatnonzero(own), ref_mask & ~own, families, prefix, cols, ctx)
        else:
            _compute(table, np.flatnonzero(ref_mask), ref_mask, families, prefix, cols, ctx)

        out = pd.DataFrame(cols, index=table.index)
        if "O" in families:
            # 같은 웨이퍼 다른 스테이지(참조 집합에 있을 때)의 편차와 시간 간격 (다른 행이므로 자기 정답과 무관)
            ref = table.loc[ref_mask, ["WAFER_ID", "STAGE", "T_START"]].assign(dev=dev[ref_mask])
            other = table[["WAFER_ID", "STAGE", "T_START"]].assign(STAGE=table["STAGE"].map({"A": "B", "B": "A"}))
            m = other.merge(ref, on=["WAFER_ID", "STAGE"], how="left", suffixes=("", "_o"))
            out[prefix + "O_dev"] = m["dev"].to_numpy()
            out[prefix + "O_gap"] = ((m["T_START_o"] - m["T_START"]) / 3600.0).to_numpy()
        return out

    return fn


VARIANTS = {
    "all": dict(families=FAMILIES, prefix="NB_"),
    "all_lgbonly": dict(families=FAMILIES, prefix="NX_"),
    "all_loo": dict(families=FAMILIES, prefix="NB_", inner_folds=0),   # LOO 만 (누수 확인용)
    "lean": dict(families=("K", "L", "S", "O"), prefix="NB_"),
    "EDO": dict(families=("E", "D", "O"), prefix="NB_"),
    "DO": dict(families=("D", "O"), prefix="NB_"),
    **{f"no_{f}": dict(families=tuple(x for x in FAMILIES if x != f), prefix="NB_") for f in FAMILIES},
}
FINAL = "EDO"


def main():
    p = argparse.ArgumentParser(description="neighbors 실험")
    p.add_argument("--all", action="store_true", help="탐색 변형 전부")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(HERE))
    args = p.parse_args()

    names = list(VARIANTS) if args.all else (args.variants or [FINAL])
    outdir = Path(args.outdir)
    for name in names:
        spec = VARIANTS[name]
        out = outdir / ("neighbors.json" if (name == FINAL and args.seed == 0) else f"neighbors_{name}_s{args.seed}.json")
        t0 = time.time()
        res = run(feature_fn=make_feature_fn(**spec), fold_seed=args.seed, label=f"neighbors_{name}", out=str(out))
        print(summary(res), f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
