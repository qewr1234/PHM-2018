"""최종 앙상블 멤버 3종과 레짐별 학습·예측 프로토콜.

멤버 (명세 {"make": 팩토리, "cols": 컬럼 선택, "fillna": 결측 대체값, "per_regime": 레짐별 학습 여부})
- lgb      : 레짐별 LightGBM. 전체 580 컬럼(기본 + TS_ 정적 + NB_ 이웃 확장) 중 학습 행만으로 짧은 LightGBM 을 돌려
             gain 상위 150 컬럼을 고른 뒤 본 모델을 학습한다 (TopKLGB; 폴드·레짐 안에서 고르므로 누수 없음).
- seqdl    : 원시 센서 시퀀스 1D CNN + compact 표 특징 MLP 융합 신경망 (PyTorch CPU). 전역 학습 + 레짐 임베딩,
             목표는 레짐별 학습 평균을 뺀 값. 에폭 수는 학습 행 20% 내부 분할로만 고르고 전체 학습 행으로 다시 학습, 시드 3개 평균.
- gp_state : 레짐별 가우시안 프로세스, 입력 = 시각(h) + 드레서·패드·멤브레인 사용량. 커널 = 시간 RBF + 상태 RBF_ARD + White,
             초매개변수는 학습 폴드 주변가능도로만 맞춘다. "최근 몇 시간의 수준 + 소모품 상태" 를 매끄럽게 보간하는 멤버.

스레드 (공유 CPU 4개 기준): 이 모듈을 lightgbm/torch 보다 먼저 import 해 OMP_WAIT_POLICY=PASSIVE 를 적용한다.
LightGBM n_jobs=1 (레짐당 수백 행이라 스레드가 많으면 오히려 느리다), torch 2, BLAS 2 (threadpoolctl).
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 스핀 대기로 다른 작업과 CPU 를 다툴 때 수십 배 느려지는 것을 막는다
os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Hyperparameter, Kernel, WhiteKernel  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

TORCH_THREADS, BLAS_THREADS, LGB_JOBS = 2, 2, 1
# compact 컬럼: 이웃·순서·소모품·가공 시간·레짐 (신경망 표 가지가 보는 컬럼). 정적 TS_ 특징은 lgb 만 본다.
COMPACT_PREFIX = ("NB_", "same_", "cross_", "regime_mu", "OTHER_", "USAGE_", "ACTIVE_SEC", "GROUP", "STAGE_B")
STATIC_PREFIX = ("TS_",)
GP_STATE_COLS = ["USAGE_OF_DRESSER_start", "USAGE_OF_POLISHING_TABLE_start", "USAGE_OF_MEMBRANE_start"]


def enforce_threads():
    """torch 2, BLAS 2 스레드로 고정한다 (다른 모듈이 바꿨어도 make() 마다 되돌린다)."""
    if torch.get_num_threads() != TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    threadpool_limits(limits=BLAS_THREADS, user_api="blas")


enforce_threads()


def compact_cols(columns):
    return [c for c in columns if c.startswith(COMPACT_PREFIX) and not c.startswith(STATIC_PREFIX)]


def pick(names):
    return lambda columns: [c for c in columns if c in names]


# ----------------------------------------------------------------------------- lgb: gain 상위 K 컬럼 선택 LightGBM
LGB_BASE = dict(n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=LGB_JOBS)
TOPK_SCREEN = {**LGB_BASE, "n_estimators": 300, "learning_rate": 0.05}
LGB_TOPK = 150


class TopKLGB:
    """fit 안에서 학습 행만으로 gain 상위 topk 컬럼을 고른 뒤(짧은 LightGBM) 본 모델을 그 컬럼으로 학습한다."""

    def __init__(self, topk=LGB_TOPK, **params):
        self.topk, self.params = topk, {**LGB_BASE, **params}

    def fit(self, X, y):
        self.cols = list(X.columns)
        if self.topk is not None and len(self.cols) > self.topk:
            g = pd.Series(lgb.LGBMRegressor(**TOPK_SCREEN).fit(X, y).booster_.feature_importance("gain"), index=self.cols)
            self.cols = list(g.sort_values(ascending=False).index[: self.topk])
        self.m = lgb.LGBMRegressor(**self.params).fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])


# ----------------------------------------------------------------------------- seqdl: 시퀀스 CNN + 표 MLP 융합망
class SeqEncoder(nn.Module):
    """단계 하나의 [C, L] 궤적 → 2·width 벡터 (평균 + 최대 풀링)."""

    def __init__(self, c_in, width=32, k=5, n_layers=3):
        super().__init__()
        layers, d = [], c_in
        for i in range(n_layers):
            layers += [nn.Conv1d(d, width, k if i < 2 else 3, padding=(k if i < 2 else 3) // 2, stride=1 if i == 0 else 2),
                       nn.SiLU()]
            d = width
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean(2), h.amax(2)], 1)


class FusionNet(nn.Module):
    """단계 공유 CNN(단계 존재 마스크로 가림) + 레짐 임베딩 + 표 특징 MLP 가지 → 헤드."""

    def __init__(self, c_in, d_tab, width=32, d_hidden=128, dropout=0.1, d_embed=4, n_regime=3, d_head=64):
        super().__init__()
        self.embed = nn.Embedding(n_regime, d_embed)
        self.enc = SeqEncoder(c_in, width)
        self.seq_drop = nn.Dropout(dropout)
        self.tab = nn.Sequential(nn.Linear(d_tab, d_hidden), nn.SiLU(), nn.Dropout(dropout))
        d = d_embed + 3 * 2 * width + 3 + d_hidden
        self.head = nn.Sequential(nn.Linear(d, d_head), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_head, 1))

    def forward(self, seq, mask, tab, r):
        B, S, C, L = seq.shape
        h = self.enc(seq.reshape(B * S, C, L)).reshape(B, S, -1) * mask[:, :, None]
        parts = [self.embed(r), self.seq_drop(h.reshape(B, -1)), mask, self.tab(tab)]
        return self.head(torch.cat(parts, 1)).squeeze(1)


def regime_id(X):
    """GROUP·STAGE_B 컬럼 → 레짐 번호 (1A=0, 4A=1, 4B=2)."""
    g = X["GROUP"].to_numpy() if "GROUP" in X else np.full(len(X), 4)
    b = X["STAGE_B"].to_numpy() if "STAGE_B" in X else np.zeros(len(X))
    return np.where(g == 1, 0, np.where(b == 0, 1, 2)).astype(np.int64)


SEQDL_PARAMS = dict(width=32, d_hidden=128, dropout=0.1, wd=1e-3, lr=3e-3, max_epochs=120, min_epochs=10, patience=30,
                    n_seeds=3, n_inner=2, inner_frac=0.2, batch_size=256, huber_delta=1.0, clip=5.0, seed0=0)


class SeqDL:
    """sklearn 풍 회귀기. 시퀀스는 X.index 로 seq/present 배열(표 행 순서)에서 찾고, 표 특징은 X 의 컬럼을 쓴다.

    학습: 목표 = y - 레짐별 학습 평균, 표준화, Huber, AdamW, 미니배치. 에폭 수는 학습 행의 inner_frac 내부 분할
    (시드 n_inner 개, 곡선 평균의 이동평균 최소)로 고른 뒤 전체 학습 행으로 다시 학습, 시드 n_seeds 개 평균.
    """

    def __init__(self, seq, present, **params):
        self.seq, self.present, self.p = seq, present, {**SEQDL_PARAMS, **params}

    # --- 표 특징 전처리 (중앙값 대체 + 결측 표시 + 분위수→정규) ---
    def _fit_tab(self, X):
        A = X.to_numpy(dtype=np.float64)
        self.med = np.nan_to_num(np.nanmedian(A, axis=0))
        na = np.isnan(A)
        ind, seen = [], set()
        for j in range(A.shape[1]):
            if na[:, j].any() and na[:, j].tobytes() not in seen:
                seen.add(na[:, j].tobytes())
                ind.append(j)
        self.ind_cols = ind
        self.qt = QuantileTransformer(n_quantiles=min(200, len(A)), output_distribution="normal", random_state=0)
        self.qt.fit(np.where(na, self.med[None, :], A))
        return self._tab(X)

    def _tab(self, X):
        A = X[self.cols].to_numpy(dtype=np.float64)
        na = np.isnan(A)
        Z = np.clip(self.qt.transform(np.where(na, self.med[None, :], A)), -self.p["clip"], self.p["clip"])
        if self.ind_cols:
            Z = np.column_stack([Z, na[:, self.ind_cols].astype(np.float64)])
        return Z.astype(np.float32)

    # --- 시퀀스 전처리 (학습 행 기준 채널별 표준화, 결측 0 + 마스크) ---
    def _fit_seq(self, idx):
        s = self.seq[idx]
        self.s_mu = np.nanmean(s, axis=(0, 1, 3))
        self.s_sd = np.nanstd(s, axis=(0, 1, 3))
        self.s_sd = np.where(self.s_sd < 1e-6, 1.0, self.s_sd)
        return self._seq(idx)

    def _seq(self, idx):
        z = (self.seq[idx] - self.s_mu[None, None, :, None]) / self.s_sd[None, None, :, None]
        z = np.nan_to_num(np.clip(z, -self.p["clip"], self.p["clip"]), nan=0.0).astype(np.float32)
        return z, self.present[idx].astype(np.float32)

    def _inputs(self, X, fit=False):
        idx = np.asarray(X.index)
        seq, mask = (self._fit_seq if fit else self._seq)(idx)
        tab = (self._fit_tab if fit else self._tab)(X)
        return seq, mask, tab, regime_id(X)

    @staticmethod
    def _tensors(parts):
        return [torch.from_numpy(a) for a in parts]

    def _train(self, inp, t, seed, epochs, val=None):
        p = self.p
        torch.manual_seed(seed)
        seq, mask, tab, r = inp
        net = FusionNet(seq.shape[2], tab.shape[1], width=p["width"], d_hidden=p["d_hidden"], dropout=p["dropout"])
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        loss_fn = nn.HuberLoss(delta=p["huber_delta"])
        S, M, T, R = self._tensors([seq, mask, tab, r])
        tt = torch.from_numpy(t.astype(np.float32))
        n, bs = len(t), min(p["batch_size"], len(t))
        g = torch.Generator().manual_seed(seed)
        if val is not None:
            Sv, Mv, Tv, Rv = self._tensors(val[0])
            tv = val[1]
        curve = []
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(net(S[b], M[b], T[b], R[b]), tt[b])
                loss.backward()
                opt.step()
            if val is not None:
                net.eval()
                with torch.no_grad():
                    pv = net(Sv, Mv, Tv, Rv).numpy()
                curve.append(float(np.mean((pv - tv) ** 2)))
                if len(curve) > p["min_epochs"] + p["patience"] and np.argmin(curve) < len(curve) - p["patience"]:
                    break  # patience 동안 개선 없으면 중단
        return net, curve

    def fit(self, X, y):
        p = self.p
        self.cols = list(X.columns)
        inp = self._inputs(X, fit=True)
        t = np.asarray(y, dtype=np.float64)
        r = inp[3]
        # 레짐별 학습 평균을 빼서 레짐 간 차이가 아닌 레짐 안의 변동을 학습한다
        self.r_mu = np.array([t[r == k].mean() if (r == k).any() else t.mean() for k in range(3)])
        t = t - self.r_mu[r]
        self.t_mu, self.t_sd = t.mean(), t.std() + 1e-9
        t = (t - self.t_mu) / self.t_sd
        n = len(t)
        seeds = [p["seed0"] + s for s in range(p["n_seeds"])]
        curves = []
        for s in seeds[:p["n_inner"]]:  # 내부 분할(학습 행의 20%)로 에폭 수 선택 — 학습 행만 쓴다
            perm = np.random.RandomState(1000 + s).permutation(n)
            nv = max(int(round(n * p["inner_frac"])), 10)
            va, tr = perm[:nv], perm[nv:]
            sub = tuple(a[tr] for a in inp)
            vsub = tuple(a[va] for a in inp)
            _, curve = self._train(sub, t[tr], s, p["max_epochs"], val=(vsub, t[va]))
            curves.append(curve)
        L = max(map(len, curves))
        c = np.mean([np.pad(cv, (0, L - len(cv)), mode="edge") for cv in curves], axis=0)
        k = 5
        cs = np.convolve(c, np.ones(k) / k, mode="valid")
        best = int(np.argmin(cs[p["min_epochs"]:]) + p["min_epochs"] + k // 2)
        self.best_epochs, self.inner_curve = best, c
        self.nets = [self._train(inp, t, s, best)[0] for s in seeds]
        return self

    def predict(self, X):
        inp = self._inputs(X)
        S, M, T, R = self._tensors(inp)
        outs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                outs.append(net(S, M, T, R).numpy())
        return np.mean(outs, axis=0) * self.t_sd + self.t_mu + self.r_mu[inp[3]]


# ----------------------------------------------------------------------------- gp_state: 시간 + 소모품 상태 GP
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
    """레짐별 GP: 입력 = 시각(h, 표준화 안 함; hours 에서 X.index 로 찾는다) + 소모품 상태(표준화).

    커널 = C·RBF(시간) + C·RBF_ARD(상태) + White. 주변가능도 최대화(L-BFGS-B, maxiter)는 학습 폴드 안에서만 한다.
    """

    def __init__(self, hours, state_cols=GP_STATE_COLS, time_bounds=(0.1, 500.0), maxiter=60):
        self.hours, self.state_cols, self.time_bounds, self.maxiter = hours, list(state_cols), time_bounds, maxiter

    def _design(self, X):
        Z = X.reindex(columns=self.state_cols).to_numpy(float)
        return np.column_stack([self.hours.loc[X.index].to_numpy(float), Z])

    def fit(self, X, y):
        Z = self._design(X)
        with np.errstate(all="ignore"):
            self.med = np.nan_to_num(np.nanmedian(Z, axis=0))
        Z = np.where(np.isnan(Z), self.med, Z)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        self.mu[0], self.sd[0] = 0.0, 1.0  # 시간은 시간 단위 그대로
        Zs = (Z - self.mu) / self.sd
        ns = len(self.state_cols)
        kt = Dims(RBF(5.0, self.time_bounds), [0])
        ks = Dims(RBF(np.ones(ns), (1e-2, 1e2)), list(range(1, 1 + ns)))
        k = ConstantKernel(1.0, (1e-2, 1e2)) * kt + ConstantKernel(1.0, (1e-2, 1e2)) * ks + WhiteKernel(0.5, (1e-3, 1e1))

        def opt(obj, x0, bounds):  # L-BFGS-B, 반복 상한 (학습 폴드 주변가능도만 사용)
            r = minimize(obj, x0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": self.maxiter})
            return r.x, r.fun
        self.gp = GaussianProcessRegressor(k, optimizer=opt, normalize_y=True, n_restarts_optimizer=0, random_state=0)
        # n≈300~800 의 촐레스키는 작아서 BLAS 스레드를 늘려도 이득이 없고 다른 작업과 겹치면 경합으로 크게 느려진다.
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            self.gp.fit(Zs, y)
        return self

    def predict(self, X):
        Z = self._design(X)
        Z = np.where(np.isnan(Z), self.med, Z)
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            return self.gp.predict((Z - self.mu) / self.sd)


# ----------------------------------------------------------------------------- 명세와 학습·예측 프로토콜
def make_models(table, seq, present, tiny=False):
    """최종 멤버 명세. table: 표(시각 T_START 를 GP 가 X.index 로 찾는다), seq/present: cmp_data.sequence_tensor.

    tiny=True 는 테스트용 (트리 20 그루, 에폭 4, GP 반복 3) — 수치 보고에는 쓰지 않는다.
    """
    hours = table["T_START"] / 3600.0
    lgb_over = {"n_estimators": 20} if tiny else {}
    dl_over = dict(max_epochs=4, min_epochs=1, patience=100, n_seeds=1, n_inner=1) if tiny else {}
    gp_over = {"maxiter": 3} if tiny else {}

    def spec(make, cols, per_regime):
        def mk():
            enforce_threads()
            return make()
        return {"make": mk, "cols": cols, "fillna": None, "per_regime": per_regime}

    return {
        "lgb": spec(lambda: TopKLGB(**lgb_over), "all", True),
        "seqdl": spec(lambda: SeqDL(seq, present, **dl_over), compact_cols, False),
        "gp_state": spec(lambda: GPMember(hours, **gp_over), pick(GP_STATE_COLS), True),
    }


def select_columns(spec, columns, X_train):
    """명세의 컬럼 선택("all" | callable) 뒤 학습 행에서 전부 결측인 컬럼을 뺀다."""
    cols = spec.get("cols", "all")
    use = list(columns) if cols == "all" else list(cols(list(columns)))
    return [c for c in use if X_train[c].notna().any()]


def fit_predict(spec, X, y, reg, train_idx, pred_idx):
    """train_idx 행으로 학습해 pred_idx 행을 예측한다 (per_regime 이면 레짐마다 따로). X 는 표 순서의 RangeIndex 를 유지한다."""
    pred = np.zeros(len(pred_idx))
    groups = [(train_idx, np.ones(len(pred_idx), bool))]
    if spec.get("per_regime", True):
        groups = [(train_idx[reg[train_idx] == r], reg[pred_idx] == r) for r in np.unique(reg[pred_idx])]
    for a, b in groups:
        if not b.any():
            continue
        use = select_columns(spec, X.columns, X.iloc[a])
        Xa, Xb = X.iloc[a][use], X.iloc[pred_idx[b]][use]
        if spec.get("fillna") is not None:
            Xa, Xb = Xa.fillna(spec["fillna"]), Xb.fillna(spec["fillna"])
        pred[b] = spec["make"]().fit(Xa, y[a]).predict(Xb)
    return pred
