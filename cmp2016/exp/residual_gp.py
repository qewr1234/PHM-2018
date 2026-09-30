"""residual_gp: 목표 가공(잔차 학습)과 매끄러운 공정 모델(GP·KNN·칼만 평활) 실험.

아이디어
(a) 잔차 학습 (target_fn): 멤버가 y 대신 y - offset 을 배운다. offset 후보 = NB_trend_15, NB_time_3,
    regime_mu + same_prev1, 세 후보의 평균(blend), 반만 빼는 부분 잔차(time3_half).
(b) 레짐별 가우시안 프로세스 멤버: 커널 = C·RBF(시간 h) + C·RBF_ARD(드레서·패드·멤브레인 사용량, 표준화) + White,
    초매개변수는 학습 폴드의 주변가능도(L-BFGS-B, 60회)로만 최적화, normalize_y. 변형: 시간 커널 Matern 1.5/2.5,
    소모품 6종, ACTIVE_SEC·P·V 항 추가, 곱 커널, 이웃 특징 항 추가(gp_rich).
(c) KNN 멤버: 작은 상태 공간(시간·소모품·이웃 특징)에서 학습 폴드 LOO-MSE 로 특징 가중치와 k 를 탐욕 좌표 탐색.
(d) 국소 수준(local level) 칼만 필터 + 양방향(two-filter) 평활 특징 (feature_fn): 레짐별 참조 편차 시계열에
    q(공정 잡음/h)·R 을 ML 로 맞추고 각 웨이퍼 시점에서 과거·미래 필터를 정밀도 가중 결합한 수준·log분산·
    과거만·미래만·기울기 (same/cross 소스). 참조 행은 inner 5-fold 로 자기 폴드 밖 참조 행만 써서 계산.

탐색 절차: 멤버 후보 (b)(c) 는 캐시된 seed 0 폴드 특징으로 OOF 를 만들어 기준 4멤버 OOF(cache_residual_gp_base_s0.npz,
harness 기준선 실행의 OOF 와 동일)와 NNLS 로 섞은 앙상블 CV 로 선별 (harness.run 과 같은 계산, --screen).
(a)(d) 와 조합은 harness.run 으로 채점 (--variants).

결과 (5-fold CV 앙상블, fold_seed=0, 기준선 7.596 / 테스트 6.969 / 검증 6.850)
- (a) a_time3 (offset=NB_time_3): 앙상블 8.223 (Δ+0.627, lgb 7.718→8.479, lgb_small 8.30→8.73). 트리가 offset 의
      잡음(offset 단독 MSE ≈10)을 되돌리려면 선형 보정을 배워야 하는데 트리에는 어렵다. 나머지 offset(NB_trend_15,
      regime_mu+same_prev1, blend)은 단독 MSE 가 더 크므로(12.6, 14.6) 실행하지 않고 중단. 잔차 학습은 기각.
- (b) gp_state (RBF 시간 + RBF_ARD 상태 + White): 멤버 8.692, 앙상블 7.403 (Δ-0.193, 가중치 0.31, extra_trees 0.00).
      변형: gp_m15(Matern1.5) 멤버 8.498/앙상블 7.403, gp_m25 8.580/7.421, gp_u6(소모품 6종) 8.695/7.392,
      gp_act(+ACTIVE_SEC·PV 항) 8.813/7.434, gp_prod(곱 커널) 9.178/7.469, gp_state+gp_m15 둘 다 7.399, gp_rich(이웃 특징 항) 폴드0 에서 더 나쁨(4A 8.29 vs 7.86).
      차이는 모두 0.05 미만이라 과제 명세대로 가장 단순한 gp_state 를 택했다. 학습된 시간 길이척도 3~7 h, 드레서 0.3σ, 멤브레인 0.6σ,
      패드 26~100σ(거의 무시) — GP 는 "최근 몇 시간의 수준 + 드레서·멤브레인 상태" 를 매끄럽게 보간하는 멤버다.
- (c) knn (탐욕 가중 KNN): 멤버 10.668, 가중치 0.00, 앙상블 7.596 (Δ0). LOO 최적 가중치가 홀드아웃에서 과적합(LOO 7.6 vs 9.4).
- (d) d_kf (칼만 평활 특징 12개): 앙상블 7.537 (Δ-0.059, lgb 7.718→7.649; 모호 구간). gp_state 와 함께(bd_gp) 7.395 로
      gp_state 단독 7.403 대비 0.008 — 잡음 수준이라 최종에서 뺐다.

최종 (b_gp_state = 기준 4멤버 + gp_state): CV 앙상블 7.403 (Δ-0.193), 공식 테스트 6.963, 검증 6.840
(가중치 lgb 0.62, lgb_small 0.04, extra_trees 0.00, ridge 0.03, gp_state 0.31).
fold_seed=1 확인: CV 7.358 (같은 폴드 기준선 7.510 → Δ-0.152), 테스트 6.917, 검증 6.761. CV 이득은 두 시드 모두 문턱(0.10)을 넘지만
공개 홀드아웃은 seed 0 에서 기준선과 같은 수준(테스트 -0.006, 검증 -0.010), seed 1 모델은 검증 -0.089.

실행: python cmp2016/exp/residual_gp.py            # 최종 변형만 (혼자 약 4분, 다른 실험과 병행 시 8.5분) → cmp2016/exp/residual_gp.json
      python cmp2016/exp/residual_gp.py --all      # 탐색 변형 전부 (harness)
      python cmp2016/exp/residual_gp.py --screen gp_state knn        # 멤버 후보 선별 (캐시 OOF)
      python cmp2016/exp/residual_gp.py --variants b_gp_state --seed 1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 다른 실험과 CPU 를 나눌 때 스핀 대기 낭비를 줄인다
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Hyperparameter, Kernel, Matern, WhiteKernel  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, base_features, fit_predict, get_table, run, summary  # noqa: E402
from cmp_data import TARGET, regime  # noqa: E402
from run_cmp import mse  # noqa: E402

BASE_OOF = HERE / "cache_residual_gp_base_s0.npz"
BLAS_THREADS = 2
STATE = ["USAGE_OF_DRESSER_start", "USAGE_OF_POLISHING_TABLE_start", "USAGE_OF_MEMBRANE_start"]
NB_SMALL = ["NB_time_3", "NB_trend_15", "same_prev1", "same_next1", "cross_prev1", "cross_next1"]


def hours_of(X):
    """X.index 로 표에서 시작 시각(시간 단위)을 찾는다 (T_START 는 특징에서 빠져 있다)."""
    return get_table().loc[X.index, "T_START"].to_numpy(float) / 3600.0


# ---------------------------------------------------------------- (a) 잔차 학습 offset
def make_target_fn(kind):
    def fn(X, table):
        mu_prev = X["regime_mu"] + X["same_prev1"].fillna(0.0)
        cands = {"trend15": X["NB_trend_15"], "time3": X["NB_time_3"], "mu_prev1": mu_prev,
                 "blend": (X["NB_trend_15"] + X["NB_time_3"] + mu_prev) / 3.0,
                 "blend2": (X["NB_trend_15"] + X["NB_time_3"]) / 2.0}
        if kind == "kf":  # (d) 칼만 평활 수준을 offset 으로 (feature_fn=make_kalman_fn() 과 함께)
            cands["kf"] = X["regime_mu"] + X["NB_KF_same_level"]
        if kind == "time3_half":  # 반만 빼는 부분 잔차 (offset 의 잡음을 트리가 되돌리기 어려운 문제를 완화)
            cands["time3_half"] = X["regime_mu"] + 0.5 * (X["NB_time_3"] - X["regime_mu"])
        return cands[kind].fillna(X["regime_mu"]).to_numpy(float)
    return fn


# ---------------------------------------------------------------- (b) GP 멤버
class Dims(Kernel):
    """지정한 열(dims)만 보는 커널 래퍼: 덧셈 구조 커널(시간 항 + 상태 항)을 만들기 위해 쓴다."""

    def __init__(self, kernel, dims):
        self.kernel, self.dims = kernel, dims

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, dims=self.dims)
        if deep:
            params.update(("kernel__" + k, v) for k, v in self.kernel.get_params().items())
        return params

    @property
    def hyperparameters(self):
        return [Hyperparameter("kernel__" + h.name, h.value_type, h.bounds, h.n_elements) for h in self.kernel.hyperparameters]

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        return type(self) is type(b) and self.kernel == b.kernel and list(self.dims) == list(b.dims)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)[:, self.dims]
        Y = None if Y is None else np.asarray(Y)[:, self.dims]
        return self.kernel(X, Y, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(np.asarray(X)[:, self.dims])

    def is_stationary(self):
        return self.kernel.is_stationary()

    @property
    def requires_vector_input(self):
        return self.kernel.requires_vector_input

    def __repr__(self):
        return f"Dims({self.kernel!r}, {list(self.dims)})"


class GPMember:
    """레짐별 GP: 입력 = 시간(h, 표준화 안 함) + 소모품 상태(표준화) [+ 이웃 특징(표준화)].

    커널 = C·RBF(시간) + C·RBF_ARD(상태) [+ C·RBF_ARD(이웃)] + White. 주변가능도 최대화는 학습 폴드 안에서만 한다.
    """

    def __init__(self, rich=False, n_restarts=0, time_bounds=(0.1, 500.0), maxiter=60, time_kernel="rbf",
                 state_cols=STATE, extra_cols=(), product=False):
        self.rich, self.n_restarts, self.time_bounds, self.maxiter = rich, n_restarts, time_bounds, maxiter
        self.time_kernel, self.state_cols, self.extra_cols, self.product = time_kernel, list(state_cols), list(extra_cols), product

    def _design(self, X):
        cols = self.state_cols + self.extra_cols + (NB_SMALL if self.rich else [])
        Z = X.reindex(columns=cols).to_numpy(float)
        return np.column_stack([hours_of(X), Z])

    def _time_kernel(self):
        if self.time_kernel == "rbf":
            return RBF(5.0, self.time_bounds)
        return Matern(5.0, self.time_bounds, nu={"matern15": 1.5, "matern25": 2.5}[self.time_kernel])

    def fit(self, X, y):
        Z = self._design(X)
        with np.errstate(all="ignore"):
            self.med = np.nan_to_num(np.nanmedian(Z, axis=0))  # 전부 결측인 열(1A 의 cross_*)은 0
        Z = np.where(np.isnan(Z), self.med, Z)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        self.mu[0], self.sd[0] = 0.0, 1.0  # 시간은 시간 단위 그대로
        Zs = (Z - self.mu) / self.sd
        ns, ne = len(self.state_cols), len(self.extra_cols)
        kt = Dims(self._time_kernel(), [0])
        ks = Dims(RBF(np.ones(ns), (1e-2, 1e2)), list(range(1, 1 + ns)))
        if self.product:  # 시간 × 상태 곱 커널 (둘 다 가까워야 비슷)
            k = ConstantKernel(1.0, (1e-2, 1e2)) * kt * ks
        else:             # 덧셈 커널 (시간 항 + 상태 항)
            k = ConstantKernel(1.0, (1e-2, 1e2)) * kt + ConstantKernel(1.0, (1e-2, 1e2)) * ks
        if ne:
            k = k + ConstantKernel(1.0, (1e-2, 1e2)) * Dims(RBF(np.ones(ne), (1e-2, 1e2)), list(range(1 + ns, 1 + ns + ne)))
        if self.rich:
            nr, o = len(NB_SMALL), 1 + ns + ne
            k = k + ConstantKernel(1.0, (1e-2, 1e2)) * Dims(RBF(np.ones(nr), (1e-2, 1e2)), list(range(o, o + nr)))
        k = k + WhiteKernel(0.5, (1e-3, 1e1))
        def opt(obj, x0, bounds):  # L-BFGS-B, 반복 상한 (학습 폴드 주변가능도만 사용)
            r = minimize(obj, x0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": self.maxiter})
            return r.x, r.fun
        self.gp = GaussianProcessRegressor(k, optimizer=opt, normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                           random_state=0)
        # n≈300~800 의 촐레스키는 작아서 BLAS 스레드를 늘려도 이득이 없고, 다른 작업과 겹치면 스레드 경합으로
        # 5배 이상 느려진다 (4스레드 경합 547초 vs 2스레드 75~133초). 2개로 고정한다.
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            self.gp.fit(Zs, y)
        return self

    def predict(self, X):
        Z = self._design(X)
        Z = np.where(np.isnan(Z), self.med, Z)
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            return self.gp.predict((Z - self.mu) / self.sd)


# ---------------------------------------------------------------- (c) KNN 멤버 (탐욕 특징 가중치)
KNN_COLS = STATE + ["USAGE_OF_BACKING_FILM_start", "USAGE_OF_DRESSER_TABLE_start", "USAGE_OF_PRESSURIZED_SHEET_start",
                    "ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum"] + NB_SMALL


class KNNGreedy:
    """작은 상태 공간의 KNN 평균. 특징 가중치(스케일)와 k 를 학습 폴드 LOO-MSE 로 좌표 탐욕 탐색한다."""

    def __init__(self, k_grid=(3, 5, 10), scales=(0.0, 0.5, 1.0, 2.0, 4.0), passes=2, cols=KNN_COLS):
        self.k_grid, self.scales, self.passes, self.cols = k_grid, scales, passes, cols

    def _design(self, X):
        Z = X.reindex(columns=self.cols).to_numpy(float)
        return np.column_stack([hours_of(X), Z])

    def fit(self, X, y):
        Z = self._design(X)
        with np.errstate(all="ignore"):
            self.med = np.nan_to_num(np.nanmedian(Z, axis=0))
        Z = np.where(np.isnan(Z), self.med, Z)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - self.mu) / self.sd
        y = np.asarray(y, float)
        n, d = Zs.shape
        dif = np.stack([(Zs[:, j, None] - Zs[None, :, j]) ** 2 for j in range(d)])  # [d, n, n]
        eye = np.eye(n, dtype=bool)
        kmax = max(self.k_grid)

        def loo(w, k):
            D = np.tensordot(w ** 2, dif, axes=1)
            D[eye] = np.inf
            nn = np.argpartition(D, kmax, axis=1)[:, :kmax]
            dn = np.take_along_axis(D, nn, axis=1)
            order = np.argsort(dn, axis=1)[:, :k]
            return mse(y[np.take_along_axis(nn, order, axis=1)].mean(1), y)

        w, k = np.ones(d), 5
        best = loo(w, k)
        for _ in range(self.passes):
            for j in range(d):
                for s in self.scales:
                    if s == w[j]:
                        continue
                    w2 = w.copy()
                    w2[j] = s
                    v = loo(w2, k)
                    if v < best - 1e-9:
                        best, w = v, w2
            for kk in self.k_grid:
                v = loo(w, kk)
                if v < best - 1e-9:
                    best, k = v, kk
        self.w, self.k, self.loo_mse, self.Z, self.y = w, k, best, Zs, y
        return self

    def predict(self, X):
        Z = self._design(X)
        Z = np.where(np.isnan(Z), self.med, Z)
        Zs = (Z - self.mu) / self.sd
        D = (((Zs[:, None, :] - self.Z[None, :, :]) * self.w) ** 2).sum(-1)
        nn = np.argpartition(D, self.k, axis=1)[:, :self.k]
        return self.y[nn].mean(1)


# ---------------------------------------------------------------- (d) 국소 수준 칼만 평활 특징
def _ll_filter(t, d, q, R):
    """국소 수준 모델의 전진 필터: 관측 j 반영 후 상태 평균·분산과 로그가능도. 시작은 확산 사전분포."""
    n = len(t)
    xf, Pf = np.empty(n), np.empty(n)
    x, P, ll = d[0], R, 0.0  # 확산 사전분포 → 첫 관측 그대로 (분산 R)
    xf[0], Pf[0] = x, P
    for j in range(1, n):
        Pp = P + q * (t[j] - t[j - 1])
        S = Pp + R
        v = d[j] - x
        K = Pp / S
        x, P = x + K * v, Pp * (1 - K)
        xf[j], Pf[j] = x, P
        ll -= 0.5 * (np.log(2 * np.pi * S) + v * v / S)
    return xf, Pf, ll


def fit_local_level(t, d):
    """q(공정 잡음/h), R(관측 잡음) 을 최대가능도로 맞춘다 (참조 계열만 사용)."""
    v0 = max(np.var(d), 1e-6)

    def nll(p):
        return -_ll_filter(t, d, np.exp(p[0]), np.exp(p[1]))[2]
    r = minimize(nll, [np.log(v0 * 0.1), np.log(v0 * 0.5)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def two_filter_query(t_obs, d_obs, q, R, t_q, self_pos=None):
    """시각 t_q 에서의 평활 추정: 과거 관측 필터와 미래 관측(역방향) 필터를 정밀도 가중 결합한다.

    self_pos 가 주어지면 질의 행이 관측 계열의 그 위치에 있으므로 그 관측을 뺀다 (leave-one-out).
    반환: level, var, fwd(과거만), bwd(미래만).
    """
    xf, Pf, _ = _ll_filter(t_obs, d_obs, q, R)
    xb, Pb, _ = _ll_filter(-t_obs[::-1], d_obs[::-1], q, R)
    xb, Pb = xb[::-1], Pb[::-1]
    if self_pos is None:
        jf = np.searchsorted(t_obs, t_q, side="left") - 1
        jb = np.searchsorted(t_obs, t_q, side="right")
    else:
        jf, jb = self_pos - 1, self_pos + 1
    okf, okb = jf >= 0, jb < len(t_obs)
    jf_, jb_ = np.clip(jf, 0, len(t_obs) - 1), np.clip(jb, 0, len(t_obs) - 1)
    mf, vf = xf[jf_], Pf[jf_] + q * np.abs(t_q - t_obs[jf_])
    mb, vb = xb[jb_], Pb[jb_] + q * np.abs(t_obs[jb_] - t_q)
    wf, wb = np.where(okf, 1.0 / vf, 0.0), np.where(okb, 1.0 / vb, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = (wf * mf + wb * mb) / (wf + wb)
        var = 1.0 / (wf + wb)
    fwd, bwd = np.where(okf, mf, np.nan), np.where(okb, mb, np.nan)
    return level, var, fwd, bwd


INNER_FOLDS, INNER_SEED = 5, 12345


def make_kalman_fn(inner_folds=INNER_FOLDS, sources=("same", "cross"), prefix="NB_KF_"):
    def fn(table, ref_mask):
        ref_mask = np.asarray(ref_mask, bool)
        reg = regime(table)
        regs = list(np.unique(reg))
        y = table[TARGET].to_numpy(float)
        mu = {r: y[ref_mask & (reg == r)].mean() for r in regs}
        dev = y - np.array([mu[r] for r in reg])
        hours = table["T_START"].to_numpy(float) / 3600.0
        inner = np.random.RandomState(INNER_SEED).randint(0, max(inner_folds, 1), len(table))
        cols = {}

        def put(name, idx, val):
            cols.setdefault(prefix + name, np.full(len(table), np.nan))[idx] = val

        for r in regs:
            src = {"same": [r], "cross": [s for s in regs if s != r and s[0] == r[0]]}
            for sname in sources:
                sregs = src[sname]
                if not sregs:
                    continue
                cand_all = np.flatnonzero(ref_mask & np.isin(reg, sregs))
                if len(cand_all) < 10:
                    continue
                o = np.argsort(hours[cand_all], kind="stable")
                t_all, d_all = hours[cand_all][o], dev[cand_all][o]
                q, R = fit_local_level(t_all, d_all)
                # 참조가 아닌 행: 참조 전체로
                idx = np.flatnonzero(~ref_mask & (reg == r))
                groups = [(idx, t_all, d_all, None)]
                # 참조 행: inner 폴드 밖 참조 행으로 (또는 LOO)
                idx_ref = np.flatnonzero(ref_mask & (reg == r))
                if inner_folds > 1:
                    for f in range(inner_folds):
                        own = idx_ref[inner[idx_ref] == f]
                        c = cand_all[inner[cand_all] != f]
                        oc = np.argsort(hours[c], kind="stable")
                        groups.append((own, hours[c][oc], dev[c][oc], None))
                else:
                    pos = {i: p for p, i in enumerate(cand_all[o])}
                    sp = np.array([pos.get(i, -1) for i in idx_ref])
                    groups.append((idx_ref, t_all, d_all, sp))
                for gidx, t_obs, d_obs, sp in groups:
                    if len(gidx) == 0:
                        continue
                    if sp is not None and (sp < 0).any():  # cross 소스에는 자기 자신이 없다
                        sp = None
                    level, var, fwd, bwd = two_filter_query(t_obs, d_obs, q, R, hours[gidx], sp)
                    put(f"{sname}_level", gidx, level)
                    put(f"{sname}_logvar", gidx, np.log(var))
                    put(f"{sname}_fwd", gidx, fwd)
                    put(f"{sname}_bwd", gidx, bwd)
                    put(f"{sname}_slope", gidx, bwd - fwd)
                put(f"{sname}_q", np.flatnonzero(reg == r), q)
        return pd.DataFrame(cols, index=table.index)

    return fn


# ---------------------------------------------------------------- 멤버 선별 (캐시된 폴드 특징 + 기준 OOF)
def base_oof():
    """기준 4멤버의 seed 0 OOF (없으면 harness 로 한 번 만든다)."""
    if not BASE_OOF.exists():
        res = run(label="base_for_screen", keep_pred=True, verbose=False)
        np.savez(BASE_OOF, **{k: np.asarray(v) for k, v in res["oof"].items()})
    z = np.load(BASE_OOF)
    return {k: z[k] for k in z.files}


def screen(new_models, target_fn=None, fold_seed=0, feature_fn=None, label=""):
    """새 멤버들의 OOF 를 만들고 기준 OOF 와 NNLS 로 섞은 앙상블 CV 를 돌려준다 (harness.run 과 같은 계산)."""
    table = get_table()
    y, reg = table[TARGET].to_numpy(), regime(table)
    tr = np.flatnonzero(((table["split"] == "train") & ~table["OUTLIER"]).to_numpy())
    folds = list(KFold(5, shuffle=True, random_state=fold_seed).split(tr))
    oof = dict(base_oof())
    t0 = time.time()
    for name, spec in new_models.items():
        o = np.full(len(table), np.nan)
        for f, (a, b) in enumerate(folds):
            ref = np.zeros(len(table), bool)
            ref[tr[a]] = True
            X = base_features(table, ref, f"s{fold_seed}_k5_f{f}")
            if feature_fn is not None:
                X = X.join(feature_fn(table, ref))
            off = np.zeros(len(table)) if target_fn is None else np.asarray(target_fn(X, table), float)
            o[tr[b]] = fit_predict(spec, X, y - off, reg, tr[a], tr[b]) + off[tr[b]]
        oof[name] = o[tr]
        print(f"[screen] {label} {name}: {mse(oof[name], y[tr]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    A = np.column_stack(list(oof.values()))
    w, _ = nnls(A, y[tr])
    w = w / w.sum()
    cv = mse(A @ w, y[tr])
    print(f"[screen] {label} ensemble {cv:.3f}  weights " + ", ".join(f"{k} {v:.2f}" for k, v in zip(oof, w)), flush=True)
    return cv, dict(zip(oof, map(float, w))), {k: mse(v, y[tr]) for k, v in oof.items()}


def member(make, cols):
    return {"make": make, "cols": cols, "fillna": None, "per_regime": True}


def pick(names):
    return lambda cols: [c for c in cols if c in names]


USAGE6 = STATE + ["USAGE_OF_BACKING_FILM_start", "USAGE_OF_DRESSER_TABLE_start", "USAGE_OF_PRESSURIZED_SHEET_start"]
EXTRA = ["ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum"]
NEW_MODELS = {
    "gp_state": member(lambda: GPMember(rich=False), pick(STATE)),
    "gp_m15": member(lambda: GPMember(time_kernel="matern15"), pick(STATE)),
    "gp_m25": member(lambda: GPMember(time_kernel="matern25"), pick(STATE)),
    "gp_u6": member(lambda: GPMember(state_cols=USAGE6), pick(USAGE6)),
    "gp_act": member(lambda: GPMember(extra_cols=EXTRA), pick(STATE + EXTRA)),
    "gp_prod": member(lambda: GPMember(product=True), pick(STATE)),
    "gp_r1": member(lambda: GPMember(n_restarts=1), pick(STATE)),
    "gp_rich": member(lambda: GPMember(rich=True), pick(STATE + NB_SMALL)),
    "knn": member(lambda: KNNGreedy(), pick(KNN_COLS)),
}

# harness.run 으로 채점하는 변형: dict(target_fn=..., models=..., feature_fn=...)
VARIANTS = {
    "a_trend15": dict(target_fn=make_target_fn("trend15")),
    "a_time3": dict(target_fn=make_target_fn("time3")),
    "a_mu_prev1": dict(target_fn=make_target_fn("mu_prev1")),
    "a_blend": dict(target_fn=make_target_fn("blend")),
    "a_blend2": dict(target_fn=make_target_fn("blend2")),
    "a_time3_half": dict(target_fn=make_target_fn("time3_half")),
    "d_kf": dict(feature_fn=make_kalman_fn()),
    "d_kf_loo": dict(feature_fn=make_kalman_fn(inner_folds=0)),
    "d_kf_same": dict(feature_fn=make_kalman_fn(sources=("same",))),
    "b_gp_state": dict(models={**BASE_MODELS, "gp_state": NEW_MODELS["gp_state"]}),
    "b_gp_rich": dict(models={**BASE_MODELS, "gp_rich": NEW_MODELS["gp_rich"]}),
    "c_knn": dict(models={**BASE_MODELS, "knn": NEW_MODELS["knn"]}),
    # 조합
    "ad_kf": dict(feature_fn=make_kalman_fn(), target_fn=make_target_fn("kf")),
    "ad_time3": dict(feature_fn=make_kalman_fn(), target_fn=make_target_fn("time3")),
    "abd_time3_gp": dict(feature_fn=make_kalman_fn(), target_fn=make_target_fn("time3"),
                         models={**BASE_MODELS, "gp_state": NEW_MODELS["gp_state"]}),
    "ab_time3_gp": dict(target_fn=make_target_fn("time3"), models={**BASE_MODELS, "gp_state": NEW_MODELS["gp_state"]}),
    "bd_gp": dict(feature_fn=make_kalman_fn(), models={**BASE_MODELS, "gp_state": NEW_MODELS["gp_state"]}),
}
FINAL = "b_gp_state"


def main():
    p = argparse.ArgumentParser(description="residual_gp 실험")
    p.add_argument("--all", action="store_true", help="탐색 변형 전부 (harness)")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--screen", nargs="*", default=None, help="멤버 후보 이름들을 캐시 OOF 로 하나씩 선별")
    p.add_argument("--screen_joint", nargs="*", default=None, help="멤버 후보들을 한꺼번에 넣어 선별")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(HERE))
    args = p.parse_args()
    outdir = Path(args.outdir)

    if args.screen is not None:
        names = args.screen or list(NEW_MODELS)
        for n in names:
            screen({n: NEW_MODELS[n]}, fold_seed=args.seed, label=n)
        return
    if args.screen_joint is not None:
        screen({n: NEW_MODELS[n] for n in args.screen_joint}, fold_seed=args.seed, label="+".join(args.screen_joint))
        return

    names = list(VARIANTS) if args.all else (args.variants or [FINAL])
    for name in names:
        spec = VARIANTS[name]
        out = outdir / ("residual_gp.json" if (name == FINAL and args.seed == 0) else f"residual_gp_{name}_s{args.seed}.json")
        t0 = time.time()
        res = run(fold_seed=args.seed, label=f"residual_gp_{name}", out=str(out), **spec)
        print(summary(res), f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
