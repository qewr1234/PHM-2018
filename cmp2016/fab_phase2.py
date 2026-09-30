"""Phase-II 고정 지연(fixed-lag) 가상 계측: 이중 단계 VM (Cheng 외 2007) 의 2단계 값을 FAB 규칙으로 만든다.

1단계 VM(fab_vm·fab_sim)은 연마 직후 바로 값을 낸다. 2단계 VM 은 L 시간 기다렸다가 그 사이 도착한 계측(앞·뒤 웨이퍼)까지
써서 같은 웨이퍼의 값을 다시 낸다 (품질 판정·SPC 용 확정값).

규칙 (시각은 첫 T_START 부터의 시간 h)
- 계측 웨이퍼 = 공식 학습 세트 − 큰 이상치 4개 (데이터셋 자체의 약 70% 표본). 테스트·검증 웨이퍼는 계측하지 않은
  웨이퍼로 보고 정답(y_score)은 채점에만 쓴다. 특징·모델 코드는 ym(계측 웨이퍼 정답, 나머지 NaN)만 읽는다.
- 웨이퍼 j 의 정답은 a_j = T_END_j + delay(1시간) 에 도착한다.
- 웨이퍼 i 의 2단계 값은 D_i = T_END_i + L 에 낸다. 쓸 수 있는 정답: a_j <= D_i 인 계측 웨이퍼 j != i
  (i 뒤에 연마됐어도 D_i 전에 도착했으면 쓴다 = 지연 안의 앞보기).
- 학습 멤버는 T_k = k·step 에 다시 학습한다. 학습 행 = max(D_j, a_j) <= T_k 인 계측 웨이퍼 (특징은 각자의 D_j 기준).
  T_k < D_i <= T_k + step 인 웨이퍼를 T_k 모델로 예측한다.
- 고르는 것(RTS 잡음비, 결합 멤버 조합, 동결 결합 가중치)은 개발 기간 선행 예측(prequential: WARMUP_DAY 이후,
  max(D_j, a_j) < DEV_END_DAY)으로만 고르고 고정한다. 본 결과의 결합 가중치는 매일 그때까지 나온 선행 예측으로
  다시 맞춘다 (완전 인과). 테스트·검증 정답으로는 아무것도 고르지 않는다.

구성
- PhaseData        : 웨이퍼 흐름(fab_vm.WaferStream), 정답 도착 규칙(avail)
- row_features     : D_i 시점의 정답 특징 (과거·앞보기·양쪽 커널·보간·상태 최근접·교차 스테이지·같은 웨이퍼 다른 스테이지)
- rts              : 국소 수준(local level) 모델의 두 필터(앞·뒤) 고정 지연 평활
- fit_member       : 주기 재학습 멤버 (LightGBM 원값·RTS 잔차, 레짐별·통합, 표 파운데이션 모델 잔차)
- blend            : 레짐별 NNLS 결합 (online = 매일 인과 재적합, frozen = 개발 기간 적합 후 고정, global = 참고용)
- score            : 테스트·검증(전체·7일 이후·배포 기간·레짐별), 계측 웨이퍼 선행 예측
"""
import hashlib
import inspect
import os
import warnings

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from cmp_data import CACHE_DIR, STATE_COLS, build_table  # noqa: E402
from fab_sim import DEFAULT_DELAY_H, DEV_END_DAY, WARMUP_DAY  # noqa: E402
from fab_vm import CROSS, DELTA_COLS, HOUR, REGIMES, WaferStream  # noqa: E402

KERNEL_H = (0.25, 1.0, 4.0)
GAP_H = 0.25                     # RTS 에서 이 시간보다 긴 공백은 로트 경계로 보고 점프 잡음을 더한다
QH_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
QJ_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
MIN_ROWS = 40                    # 레짐별 학습 행이 이만큼 모여야 모델을 쓴다
BLEND_MIN_ROWS = 30              # 결합 가중치를 맞출 최소 선행 예측 행
LGB = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=1)
TOPK = 120                       # LightGBM 멤버: 학습 행에서만 고른 gain 상위 열
TFM_TOPK = 100                   # 표 파운데이션 모델 멤버의 열 수
TFM_CACHE = CACHE_DIR / "tfm_cache"
TABPFN_CKPT = TFM_CACHE / "tabpfn" / "tabpfn-v2-regressor.ckpt"      # TabPFN v2 가중치만 쓴다 (2.5 이후는 비상업)
TABICL_CKPT = "tabicl-regressor-v2-20260212.ckpt"
TABDPT_CKPT = "tabdpt1_3.safetensors"

# 멤버 명세: kind(lgb|tabdpt|tabpfn|tabicl), resid(RTS 잔차 학습), pooled(세 레짐 한 모델), step(재학습 간격 h),
# seed, ctx(TFM 문맥 = 최근 학습 행 수, 0 = 전부), n_est(TFM 앙상블 수), topk(열 수, 없으면 TOPK·TFM_TOPK)
MEMBERS = {
    "gbm": dict(kind="lgb", resid=False),
    "gbm_res": dict(kind="lgb"),
    "gbm_res_s1": dict(kind="lgb", seed=1),
    "gbm_res_s2": dict(kind="lgb", seed=2),
    "res_pool": dict(kind="lgb", pooled=True),
    "res_pool_s1": dict(kind="lgb", pooled=True, seed=1),
    "res_pool_s2": dict(kind="lgb", pooled=True, seed=2),
    "res_6h": dict(kind="lgb", step=6.0),
    "res_pool6h": dict(kind="lgb", pooled=True, step=6.0),
    "tabdpt": dict(kind="tabdpt", n_est=4),
    "tabpfn_c300": dict(kind="tabpfn", n_est=4, ctx=300),
    "tabicl": dict(kind="tabicl", n_est=4),
}
LGB_MEMBERS = [k for k, v in MEMBERS.items() if v["kind"] == "lgb"]
TFM_MEMBERS = [k for k, v in MEMBERS.items() if v["kind"] != "lgb"]
BAGS = {"gbm_res_bag": ("gbm_res", "gbm_res_s1", "gbm_res_s2"), "res_pool_bag": ("res_pool", "res_pool_s1", "res_pool_s2")}
NOLEARN = ("rts", "K1", "INT")   # 학습 없는 멤버 (특징 행렬의 열)


# ==============================================================================================
# 데이터와 정답 도착 규칙
# ==============================================================================================
class PhaseData:
    """웨이퍼 흐름 (T_END 순). table: 표(없으면 캐시 → build_table), 누설 검사에서는 정답을 바꾼 표를 준다."""

    def __init__(self, table=None, delay_h=DEFAULT_DELAY_H, dev_day=DEV_END_DAY, warm_day=WARMUP_DAY, use_ts=True):
        if table is None:
            f = CACHE_DIR / "table.parquet"
            table = pd.read_parquet(f) if f.exists() else build_table()
        s = WaferStream(table.reset_index(drop=True), ts_dev_day=dev_day, use_ts=use_ts)
        self.s = s
        self.n = s.n
        self.delay_h, self.dev_day, self.warm_day = float(delay_h), float(dev_day), float(warm_day)
        self.dev_h = self.dev_day * 24.0
        self.t = (s.t_end - s.t_start.min()) / HOUR      # 연마 종료 시각 (h)
        self.arr = self.t + self.delay_h                 # 정답 도착 시각 (계측 웨이퍼만 의미)
        self.day = s.day
        self.reg = s.reg
        self.split = s.split
        self.meas = (s.split == "train") & ~s.outlier & np.isfinite(s.y)
        self.ym = np.where(self.meas, s.y, np.nan)       # 특징·모델이 읽는 유일한 정답
        self.y_score = s.y.copy()                        # 채점 전용
        self.partner = s.partner
        self.delta = s.delta
        st = s.t[STATE_COLS].to_numpy(float)
        dev = self.day < self.dev_day                    # 표준화는 개발 기간 값으로만 (정답 없음)
        self.state = np.nan_to_num((st - np.nanmean(st[dev], 0)) / (np.nanstd(st[dev], 0) + 1e-9))
        self.life = s.sched["LIFE_id"].to_numpy()
        self.midx = {r: np.flatnonzero(self.meas & (self.reg == r)) for r in REGIMES}

    def static(self):
        return self.s.static_matrix("vm")

    def ready(self, L):
        """계측 웨이퍼 j 를 학습·결합 행으로 쓸 수 있게 되는 시각: 자기 특징(D_j)과 정답(a_j)이 모두 있어야 한다."""
        return np.maximum(self.t + L, self.arr)

    def dev_rows(self, L):
        """개발 기간 선행 예측 행 (고르는 데만 쓴다)."""
        return self.meas & (self.day >= self.warm_day) & (self.ready(L) < self.dev_h)

    def avail(self, r, i, L):
        """D_i 에 쓸 수 있는 레짐 r 의 계측 정답: (과거, 앞보기) 전역 번호, 그리고 (kp, fs, kc) 위치."""
        M = self.midx[r]
        c = self.t[i] + L - self.delay_h                  # a_j <= D_i  <=>  t_j <= c
        kpos = np.searchsorted(M, i)                      # 흐름에서 i 보다 앞 (시간 순)
        kc = np.searchsorted(self.t[M], c, side="right")
        kp = min(kpos, kc)
        fs = kpos + (1 if kpos < len(M) and M[kpos] == i else 0)
        return M[:kp], (M[fs:kc] if kc > fs else M[:0]), (kp, fs, kc)


# ==============================================================================================
# D_i 시점의 정답 특징
# ==============================================================================================
def _ewm(v, w):
    s = w.sum()
    return float((w * v).sum() / s) if s > 1e-3 else np.nan


def _trend(tt, vv, t_now):
    if len(vv) < 5:
        return np.nan
    tc = tt - tt.mean()
    slope = (tc * (vv - vv.mean())).sum() / max((tc ** 2).sum(), 1e-9)
    return float(np.clip(vv.mean() + slope * (t_now - tt.mean()), vv.min(), vv.max()))


def kernel_avg(dt, v, h):
    lw = -0.5 * (dt / h) ** 2
    w = np.exp(lw - lw.max())
    return float((w * v).sum() / w.sum())


def two_sided(D, i, P, F):
    """학습 없는 양쪽 추정: 최근접(NN1·NN3), 가우스 커널(K0.25·K1·K4 시간), 앞뒤 선형 보간(INT)."""
    out = {}
    A = np.r_[P, F]
    if len(A) == 0:
        return out
    ti = D.t[i]
    dt = D.t[A] - ti
    v = D.ym[A]
    o = np.argsort(np.abs(dt), kind="stable")
    out["NN1"] = float(v[o[0]])
    out["NN3"] = float(v[o[:3]].mean())
    for h in KERNEL_H:
        out[f"K{h:g}"] = kernel_avg(dt, v, h)
    if len(P) and len(F):
        t1, t2 = D.t[P[-1]], D.t[F[0]]
        y1, y2 = D.ym[P[-1]], D.ym[F[0]]
        out["INT"] = float(y1 + (y2 - y1) * (ti - t1) / max(t2 - t1, 1e-9))
    else:
        out["INT"] = float(D.ym[P[-1]] if len(P) else D.ym[F[0]])
    return out


def row_features(D, i, L):
    """웨이퍼 i 의 정답 특징 (D_i = t_i + L 시점). P_ 과거, F_ 앞보기, S_ 상태 최근접, C_ 교차 스테이지, O_ 같은 웨이퍼."""
    r = D.reg[i]
    ti = D.t[i]
    P, F, _ = D.avail(r, i, L)
    out = {"A_n_past": float(len(P)), "A_n_fut": float(len(F))}
    out.update(two_sided(D, i, P, F))
    if len(P):
        yp, tp = D.ym[P], D.t[P]
        age = ti - tp
        out["P_last1"] = yp[-1]
        out["P_mean3"] = yp[-3:].mean()
        out["P_mean10"] = yp[-10:].mean()
        out["P_mean_all"] = yp.mean()
        out["P_age1"] = age[-1]
        out["P_n24"] = float((age < 24).sum())
        for h in (0.5, 2.0, 8.0, 24.0):
            out[f"P_ewmh{h:g}"] = _ewm(yp, 0.5 ** (age / h))
        rank = np.arange(len(yp))[::-1]
        for k in (2, 5, 15):
            out[f"P_ewmn{k}"] = _ewm(yp, 0.5 ** (rank / k))
        for k in (15, 40):
            out[f"P_trend{k}"] = _trend(tp[-k:], yp[-k:], ti)
        m = D.life[P] == D.life[i]
        out["P_life_mean"] = float(yp[m].mean()) if m.any() else np.nan
        out["P_life_n"] = float(m.sum())
        j = P[-1]
        for c, name in enumerate(DELTA_COLS):
            out[f"P_delta_{name}"] = D.delta[i, c] - D.delta[j, c]
    if len(F):
        yf, tf = D.ym[F], D.t[F]
        lead = tf - ti
        out["F_next1"] = yf[0]
        out["F_mean3"] = yf[:3].mean()
        out["F_mean10"] = yf[:10].mean()
        out["F_age1"] = lead[0]
        out["F_n24"] = float((lead < 24).sum())
        for h in (0.5, 2.0, 8.0, 24.0):
            out[f"F_ewmh{h:g}"] = _ewm(yf, 0.5 ** (lead / h))
        rank = np.arange(len(yf))
        for k in (2, 5, 15):
            out[f"F_ewmn{k}"] = _ewm(yf, 0.5 ** (rank / k))
        out["F_trend15"] = _trend(tf[:15], yf[:15], ti)
        m = D.life[F] == D.life[i]
        out["F_life_mean"] = float(yf[m].mean()) if m.any() else np.nan
        j = F[0]
        for c, name in enumerate(DELTA_COLS):
            out[f"F_delta_{name}"] = D.delta[i, c] - D.delta[j, c]
        if len(P):
            out["PF_diff"] = yf[0] - D.ym[P[-1]]
            out["PF_gap"] = tf[0] - D.t[P[-1]]
    A = np.r_[P, F]
    if len(A) >= 3:
        d = np.linalg.norm(D.state[A] - D.state[i], axis=1)
        o = np.argsort(d, kind="stable")
        out["S_nn3"] = float(D.ym[A[o[:3]]].mean())
        out["S_nn10"] = float(D.ym[A[o[:10]]].mean())
        out["S_dist1"] = float(d[o[0]])
    rc = CROSS.get(r)                                    # 4A <-> 4B (같은 패드·슬러리)
    if rc is not None:
        Pc, Fc, _ = D.avail(rc, i, L)
        if len(Pc):
            yc, tc = D.ym[Pc], D.t[Pc]
            age = ti - tc
            out["C_last1"] = yc[-1]
            out["C_mean3"] = yc[-3:].mean()
            out["C_age1"] = age[-1]
            for h in (2.0, 8.0):
                out[f"C_ewmh{h:g}"] = _ewm(yc, 0.5 ** (age / h))
        if len(Fc):
            yc, tc = D.ym[Fc], D.t[Fc]
            out["C_next1"] = yc[0]
            out["C_nmean3"] = yc[:3].mean()
            out["C_nage1"] = tc[0] - ti
            out["C_fewmh2"] = _ewm(yc, 0.5 ** ((tc - ti) / 2.0))
        Ac = np.r_[Pc, Fc]
        if len(Ac):
            out["C_K1"] = kernel_avg(D.t[Ac] - ti, D.ym[Ac], 1.0)
            out["C_K0.25"] = kernel_avg(D.t[Ac] - ti, D.ym[Ac], 0.25)
        j = D.partner[i]                                 # 같은 웨이퍼의 다른 스테이지 (D_i 전에 도착했을 때만)
        if j >= 0 and D.meas[j] and D.arr[j] <= ti + L:
            out["O_y"] = D.ym[j]
            out["O_lag"] = D.t[j] - ti
            Ac2 = Ac[Ac != j]
            if len(Ac2):                                 # 다른 스테이지 값의 자기 주변 수준 대비 편차
                out["O_dev"] = D.ym[j] - kernel_avg(D.t[Ac2] - D.t[j], D.ym[Ac2], 0.25)
    return out


def lag_matrix(D, L):
    return pd.DataFrame([row_features(D, i, L) for i in range(D.n)], index=np.arange(D.n))


# ==============================================================================================
# 국소 수준 RTS 고정 지연 평활 (두 필터 형태): x(t) 랜덤워크, Q(dt) = qh·dt + qj·[dt > GAP_H], 측정 잡음 1
# ==============================================================================================
def _Q(dt, qh, qj):
    return qh * dt + qj * (dt > GAP_H)


def _filter(T, Y, qh, qj):
    """앞 방향 필터: 관측 0..k-1 뒤의 상태 m[k], P[k] (시각 T[k-1])."""
    n = len(T)
    m = np.full(n + 1, np.nan)
    P = np.full(n + 1, np.inf)
    if n == 0:
        return m, P
    m[1], P[1] = Y[0], 1.0
    for k in range(1, n):
        pp = P[k] + _Q(T[k] - T[k - 1], qh, qj)
        K = pp / (pp + 1.0)
        m[k + 1] = m[k] + K * (Y[k] - m[k])
        P[k + 1] = pp * (1 - K)
    return m, P


def _backward_full(T, Y, qh, qj):
    """뒤 방향 필터 (끝 → k): bm[k], bP[k] (시각 T[k])."""
    n = len(T)
    bm = np.full(n + 1, np.nan)
    bP = np.full(n + 1, np.inf)
    if n == 0:
        return bm, bP
    bm[n - 1], bP[n - 1] = Y[n - 1], 1.0
    for k in range(n - 2, -1, -1):
        pp = bP[k + 1] + _Q(T[k + 1] - T[k], qh, qj)
        K = pp / (pp + 1.0)
        bm[k] = bm[k + 1] + K * (Y[k] - bm[k + 1])
        bP[k] = pp * (1 - K)
    return bm, bP


def _backward_range(T, Y, a, b, qh, qj):
    """관측 a..b-1 만으로 뒤 방향 필터를 돌린 시각 T[a] 의 상태."""
    m, P = Y[b - 1], 1.0
    for k in range(b - 2, a - 1, -1):
        pp = P + _Q(T[k + 1] - T[k], qh, qj)
        K = pp / (pp + 1.0)
        m = m + K * (Y[k] - m)
        P = pp * (1 - K)
    return m, P


def rts(D, L, q, which=None, regimes=REGIMES):
    """D_i 에 도착한 정답만으로 t_i 의 수준을 평활. q = {레짐: (qh, qj)}. 반환 (평활값, 분산, 앞 필터만)."""
    which = np.arange(D.n) if which is None else np.asarray(which)
    fused, var, fwd = (np.full(D.n, np.nan) for _ in range(3))
    for r in regimes:
        M = D.midx[r]
        T, Y = D.t[M], D.ym[M]
        qh, qj = q[r]
        fm, fP = _filter(T, Y, qh, qj)
        full = None
        for i in which[D.reg[which] == r]:
            ti = D.t[i]
            _, _, (kp, fs, kc) = D.avail(r, i, L)
            mf, Pf = (fm[kp], fP[kp] + _Q(ti - T[kp - 1], qh, qj)) if kp > 0 else (np.nan, np.inf)
            if kc > fs:
                if kc == len(T):
                    if full is None:
                        full = _backward_full(T, Y, qh, qj)
                    mb, Pb = full[0][fs], full[1][fs]
                else:
                    mb, Pb = _backward_range(T, Y, fs, kc, qh, qj)
                Pb = Pb + _Q(T[fs] - ti, qh, qj)
            else:
                mb, Pb = np.nan, np.inf
            fwd[i] = mf
            if np.isfinite(Pf) and np.isfinite(Pb):
                w = (1 / Pf) / (1 / Pf + 1 / Pb)
                fused[i] = w * mf + (1 - w) * mb
                var[i] = 1 / (1 / Pf + 1 / Pb)
            elif np.isfinite(Pf):
                fused[i], var[i] = mf, Pf
            elif np.isfinite(Pb):
                fused[i], var[i] = mb, Pb
    return fused, var, fwd


def select_rts(D, L):
    """레짐별 (qh, qj) 를 개발 기간 선행 예측 MSE 로 골라 고정한다 (계측 웨이퍼 정답만, 개발 기간에 도착한 것만)."""
    dev = D.dev_rows(L)
    q, mse = {}, {}
    for r in REGIMES:
        ev = np.flatnonzero(dev & (D.reg == r))
        res = []
        for a in QH_GRID:
            for b in QJ_GRID:
                f, _, _ = rts(D, L, {r: (a, b)}, which=ev, regimes=(r,))
                ok = np.isfinite(f[ev])
                res.append((float(np.mean((f[ev][ok] - D.ym[ev][ok]) ** 2)) if ok.any() else np.inf, a, b))
        res.sort()
        mse[r], q[r] = round(res[0][0], 4), (res[0][1], res[0][2])
    return q, mse


def build_X(D, L, q):
    """정답 없는 특징 + D_i 시점 정답 특징 + RTS (M_rts, M_rts_var, M_fwd)."""
    F = lag_matrix(D, L)
    F["M_rts"], F["M_rts_var"], F["M_fwd"] = rts(D, L, q)
    return D.static().join(F)


# ==============================================================================================
# 주기 재학습 멤버
# ==============================================================================================
def screen(X, y, k, params):
    """학습 행에서만 고르는 열: 값이 있는 열 중 짧은 LightGBM 의 gain 상위 k."""
    X = X.loc[:, X.notna().any()]
    cols = list(X.columns)
    if k and len(cols) > k:
        scr = lgb.LGBMRegressor(**{**params, "n_estimators": 120, "learning_rate": 0.1}).fit(X, y)
        g = pd.Series(scr.booster_.feature_importance("gain"), index=cols)
        cols = list(g.sort_values(ascending=False, kind="stable").index[:k])
    return cols


def _tfm_threads():
    return int(os.environ.get("FAB2_TFM_THREADS", "2"))


_TABDPT = {}


def _cached(repo_dir, name):
    """tfm_cache 의 HF 스냅숏에서 가중치 파일을 찾는다. 없으면 받는 방법을 알려 준다 (여기서는 내려받지 않는다)."""
    path = sorted((TFM_CACHE / "hf" / "hub" / repo_dir / "snapshots").glob(f"*/{name}"))
    if not path:
        raise FileNotFoundError(f"{name} 가 {TFM_CACHE}/hf/hub/{repo_dir} 에 없습니다. cmp2016/README.md 의 실행 절을 따라 "
                                f"HF_HOME={TFM_CACHE}/hf huggingface-cli download 로 먼저 받아 두세요.")
    return path[-1]


def tfm_model(kind, n_est, seed):
    """사전학습 표 파운데이션 모델 (가중치는 TFM_CACHE 에서만 읽는다, 내려받지 않음). fit(X, y) / predict(X) 를 가진 객체."""
    import torch
    torch.set_num_threads(_tfm_threads())
    if kind == "tabpfn":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, model_path=str(TABPFN_CKPT), device="cpu", n_estimators=n_est,
            ignore_pretraining_limits=True, random_state=seed, n_jobs=_tfm_threads())
    if kind == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(n_estimators=n_est, kv_cache=True, device="cpu", random_state=seed, n_jobs=_tfm_threads(),
                               checkpoint_version=TABICL_CKPT, model_path=_cached("models--jingang--TabICL", TABICL_CKPT),
                               allow_auto_download=False)
    if kind == "tabdpt":
        from tabdpt import TabDPTRegressor

        class _DPT(TabDPTRegressor):
            def _load_model(self):                       # 재학습마다 가중치를 다시 읽지 않는다 (같은 네트워크 공유)
                if self.model_weight_path not in _TABDPT:
                    super()._load_model()
                    _TABDPT[self.model_weight_path] = (self.model, self.path)
                self.model, self.path = _TABDPT[self.model_weight_path]

            def predict(self, X):
                return super().predict(X, n_ensembles=n_est, seed=seed)

        return _DPT(device="cpu", compile=False, use_flash=False, verbose=False,
                    model_weight_path=str(_cached("models--Layer6--TabDPT", TABDPT_CKPT)))
    raise ValueError(kind)


class Learner:
    """열 선별 + 모델 (LightGBM 또는 TFM). 선별도 그 재학습의 학습 행에서만 한다."""

    def __init__(self, kind, seed=0, params=None, n_est=4, ctx=0, topk=None):
        self.kind, self.seed, self.n_est, self.ctx = kind, seed, n_est, ctx
        self.params = {**(params or LGB), "random_state": seed}
        self.topk = (TOPK if kind == "lgb" else TFM_TOPK) if topk is None else topk

    def fit(self, X, y):
        self.cols = screen(X, y, self.topk, self.params)
        if self.kind == "lgb":
            self.m = lgb.LGBMRegressor(**self.params).fit(X[self.cols], y)
            return self
        sl = slice(-self.ctx, None) if self.ctx and len(y) > self.ctx else slice(None)   # 최근 ctx 행만 문맥
        self.m = tfm_model(self.kind, self.n_est, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X[self.cols].to_numpy(np.float32)[sl], np.asarray(y, np.float32)[sl])
        return self

    def predict(self, X):
        if self.kind == "lgb":
            return self.m.predict(X[self.cols])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.m.predict(X[self.cols].to_numpy(np.float32)), float).reshape(-1)


def fit_member(D, L, X, spec, params=None, min_rows=MIN_ROWS, topk=None, t_max=None, log=None, fits=None):
    """주기 재학습 멤버의 웨이퍼별 예측 (모델이 없던 웨이퍼는 NaN).

    T_k = k·step 에 학습 행 = 계측 & max(D_j, a_j) <= T_k (레짐별 또는 통합), 대상 = T_k < D_i <= T_k + step.
    resid: 목표 = y - M_rts, 예측 = 모델 + M_rts (RTS 가 없는 초기 행은 학습에서 뺀다).
    fits: 리스트를 주면 (T_k, 학습 행, 대상 행) 을 남긴다 (누설 검사용).
    """
    kind = spec["kind"]
    step, pooled, resid = spec.get("step", 24.0), spec.get("pooled", False), spec.get("resid", True)
    Dt = D.t + L
    ready = D.ready(L)
    off = X["M_rts"].to_numpy(float) if resid else np.zeros(D.n)
    y = D.ym - off
    p = np.full(D.n, np.nan)
    groups = [REGIMES] if pooled else [(r,) for r in REGIMES]
    Tk = step * np.arange(1, int(np.ceil(Dt.max() / step)) + 1)
    for k, T in enumerate(Tk):
        if t_max is not None and T > t_max:
            break
        for regs in groups:
            R = np.isin(D.reg, regs)
            tgt = np.flatnonzero(R & (Dt > T) & (Dt <= T + step) & np.isfinite(off))
            tr = np.flatnonzero(D.meas & R & (ready <= T) & np.isfinite(y))
            cnt = {r: int((D.reg[tr] == r).sum()) for r in regs}
            tgt = tgt[np.array([cnt[r] >= min_rows for r in D.reg[tgt]], bool)]
            if len(tgt) == 0:
                continue
            m = Learner(kind, spec.get("seed", 0), params, spec.get("n_est", 4), spec.get("ctx", 0), spec.get("topk", topk))
            m.fit(X.iloc[tr], y[tr])
            p[tgt] = m.predict(X.iloc[tgt]) + off[tgt]
            if fits is not None:
                fits.append((T, tr, tgt))
        if log is not None and k % 10 == 0:
            log(f"  day {T / 24:.1f}")
    return p


def add_bags(P):
    """시드 묶음 = 시드별 예측 평균 (있는 시드만)."""
    for b, parts in BAGS.items():
        have = [P[m] for m in parts if m in P]
        if have:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                P[b] = np.nanmean(np.column_stack(have), 1)
    return P


def nolearn_members(X):
    return {"rts": X["M_rts"].to_numpy(float), "K1": X["K1"].to_numpy(float), "INT": X["INT"].to_numpy(float)}


# ==============================================================================================
# 결합과 대체값
# ==============================================================================================
def _combine(M, w, lo, hi):
    """가중 평균 (없는 멤버는 빼고 다시 정규화, 가중 멤버가 모두 없으면 있는 멤버 평균) → 도착한 정답 범위로 자름."""
    w = np.broadcast_to(w, M.shape).copy()
    ok = np.isfinite(M)
    w[~ok] = 0
    alt = w.sum(1) <= 0
    w[alt] = ok[alt].astype(float)
    s = w.sum(1)
    v = (w * np.nan_to_num(M)).sum(1) / np.where(s > 0, s, np.nan)
    return np.clip(v, lo, hi)


def _nnls(M, y):
    w, _ = nnls(M, y)
    return w / w.sum() if w.sum() > 0 else None


def blend(D, L, P, names, mode, step=24.0, min_rows=BLEND_MIN_ROWS, fits=None):
    """레짐별 NNLS 결합 (가중치 합 1), 매일(step) 그때까지 도착한 정답 범위로 자른다.

    online : T_k 에 선행 예측 행(계측 & max(D_j, a_j) <= T_k)으로 다시 맞추고 T_k < D_i <= T_k + step 에 쓴다 (완전 인과)
    frozen : 개발 기간 선행 예측 행(D.dev_rows)으로 한 번 맞추고 고정 (배포 기간은 인과, 개발 기간 값은 같은 기간 적합)
    global : 7일 이후 전체 선행 예측 행으로 맞춤 — 미래 정답을 쓰므로 인과가 아니다 (낙관적 참고치)
    행이 모자라면 RTS 만 (RTS 가 없으면 균등). fits: 리스트를 주면 (T_k, 가중치 적합 행, 대상 행, 자르기 범위 행) 을 남긴다.
    """
    M = np.column_stack([P[m] for m in names])
    fin = np.isfinite(M).all(1)
    Dt = D.t + L
    ready = D.ready(L)
    base_w = (np.array(names) == "rts").astype(float) if "rts" in names else np.ones(len(names)) / len(names)
    out = np.full(D.n, np.nan)
    W = {}
    for r in REGIMES:
        R = D.reg == r
        fixed = None
        if mode in ("frozen", "global"):
            rows = D.meas & R & fin & (D.dev_rows(L) if mode == "frozen" else D.day >= D.warm_day)
            rows = fixed_rows = np.flatnonzero(rows)
            fixed = _nnls(M[rows], D.ym[rows]) if len(rows) >= min_rows else None
            fixed = base_w if fixed is None else fixed
        for T in step * np.arange(0, int(np.ceil(Dt.max() / step)) + 1):
            tg = np.flatnonzero(R & (Dt > T) & (Dt <= T + step))
            if len(tg) == 0:
                continue
            # 자르기 범위: T 까지 도착한 계측값. 대상이 계측 웨이퍼면 자기 정답이 범위에 들어갈 수 있으나(선행 예측 값에만 영향,
            # 개발 기간 선행 예측 차이 ≤ 0.001, 테스트·검증은 계측되지 않으므로 영향 없음) 결과 재현을 위해 그대로 둔다.
            lab = D.meas & R & (D.arr <= T)
            lo, hi = (D.ym[lab].min(), D.ym[lab].max()) if lab.any() else (-np.inf, np.inf)
            if fixed is not None:
                w, rows = fixed, fixed_rows
            else:
                rows = np.flatnonzero(D.meas & R & fin & (ready <= T))
                w = _nnls(M[rows], D.ym[rows]) if len(rows) >= min_rows else None
                w = base_w if w is None else w
            out[tg] = _combine(M[tg], w, lo, hi)
            W[r] = dict(zip(names, np.round(w, 3).tolist()))
            if fits is not None:
                fits.append((T, rows, tg, np.flatnonzero(lab)))
    return out, W


def finalize(D, L, p):
    """NaN(모델·멤버가 아직 없음) → D_i 에 도착한 같은 레짐 정답 평균 → 다른 레짐 포함 평균 → 그래도 없으면 NaN 유지."""
    p = p.copy()
    nfb = ~np.isfinite(p)
    for i in np.flatnonzero(nfb):
        P, F, _ = D.avail(D.reg[i], i, L)
        A = np.r_[P, F]
        if len(A) == 0:
            A = np.flatnonzero(D.meas & (D.arr <= D.t[i] + L))
            A = A[A != i]
        p[i] = D.ym[A].mean() if len(A) else np.nan
    return p, nfb


# ==============================================================================================
# 채점
# ==============================================================================================
def _mse(p, y, m):
    return round(float(np.mean((p[m] - y[m]) ** 2)), 3) if m.any() else None


def score(D, L, pred, nfb=None):
    """test·val: 전체, 7일 이후, 배포 기간(30일 이후), 개발 기간(7~30일), 레짐별(전체·배포).
    preq: 계측 웨이퍼 선행 예측 (7일 이후, 개발 = 선택에 쓴 행, 배포)."""
    y = D.y_score
    dep = D.day >= D.dev_day
    out = {}
    for split in ("test", "val"):
        m = D.split == split
        o = {"all": _mse(pred, y, m), "after7": _mse(pred, y, m & (D.day >= D.warm_day)), "deploy": _mse(pred, y, m & dep),
             "dev7_30": _mse(pred, y, m & (D.day >= D.warm_day) & ~dep)}
        for r in REGIMES:
            o[r] = _mse(pred, y, m & (D.reg == r))
            o[f"deploy_{r}"] = _mse(pred, y, m & dep & (D.reg == r))
        o["n_nan"] = int((m & ~np.isfinite(pred)).sum())
        if nfb is not None:
            o["n_fallback"] = int((m & nfb).sum())
        out[split] = o
    ok = D.meas & np.isfinite(pred)
    o = {"after7": _mse(pred, D.ym, ok & (D.day >= D.warm_day)), "dev": _mse(pred, D.ym, ok & D.dev_rows(L)),
         "deploy": _mse(pred, D.ym, ok & dep)}
    for r in REGIMES:
        o[r] = _mse(pred, D.ym, ok & (D.day >= D.warm_day) & (D.reg == r))
    o["n_after7"] = int((ok & (D.day >= D.warm_day)).sum())
    out["preq"] = o
    return out


def paired_bootstrap(e1, e0=None, ref=6.36, B=5000, seed=0):
    """웨이퍼 짝 붓스트랩. e1: 이 모델의 제곱오차, e0: 비교 모델(같은 웨이퍼). ref: 고정 기준 MSE."""
    n = len(e1)
    idx = np.random.default_rng(seed).integers(0, n, (B, n))
    m1 = e1[idx].mean(1)
    q = lambda a: [round(float(np.quantile(a, 0.025)), 3), round(float(np.quantile(a, 0.975)), 3)]  # noqa: E731
    out = {"n": int(n), "mse": round(float(e1.mean()), 3), "mse_ci95": q(m1),
           f"diff_vs_{ref}": round(float(e1.mean() - ref), 3), f"p_mse_below_{ref}": round(float(np.mean(m1 < ref)), 3)}
    if e0 is not None:
        d = m1 - e0[idx].mean(1)
        out.update({"ref_model_mse": round(float(e0.mean()), 3), "diff_vs_ref_model": round(float(e1.mean() - e0.mean()), 3),
                    "diff_vs_ref_model_ci95": q(d), "p_better_than_ref_model": round(float(np.mean(d < 0)), 3)})
    return out


def code_key():
    """멤버 예측 캐시 열쇠: 특징·RTS·멤버 학습 코드."""
    objs = (PhaseData, two_sided, row_features, rts, _filter, _backward_full, _backward_range, select_rts, build_X,
            screen, tfm_model, Learner, fit_member)
    src = "".join(inspect.getsource(o) for o in objs) + repr((LGB, TOPK, TFM_TOPK, MIN_ROWS, QH_GRID, QJ_GRID, KERNEL_H))
    return hashlib.sha1(src.encode()).hexdigest()[:12]
