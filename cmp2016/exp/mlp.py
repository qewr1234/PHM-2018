"""mlp: 표 특징 위의 PyTorch MLP(CPU) 를 앙상블 멤버로 추가.

아이디어
- 기존 앙상블(LightGBM 2종, ExtraTrees, 릿지)에 트리와 다른 귀납 편향을 가진 신경망을 멤버로 더한다. 특징은 그대로.
- 회귀기(TorchMLP): 중앙값 대체 + 결측 표시(같은 패턴은 하나로), 표준화(±5 클리핑) 또는 분위수→정규 변환,
  목표 표준화, 2 층 MLP(128-64, SiLU, dropout 0.1), AdamW(lr 3e-3, wd 1e-3), 전배치, Huber 손실.
  에폭 수는 학습 행의 20% 내부 분할(시드 2 개, 곡선 평균의 이동평균 최소)로 고른 뒤 전체 학습 행으로 다시 학습, 시드 5 개 평균.
  전역 모델은 레짐(1A/4A/4B) 임베딩(4 차원)을 입력에 붙인다. 스레드 1 개 (작은 행렬은 4 스레드가 부하 상태에서 40 배 느렸다).
- 탐색은 기본 4 멤버의 OOF/평가 예측 캐시(xgb_cat.py 와 같은 방식)로 새 멤버만 학습했다. 최종 실행은 진짜 멤버로 다시 학습.

시도한 것 (5-fold CV, fold_seed=0, 기준선 앙상블 7.596; "단일" 은 그 멤버의 CV, "+it" 은 기본 4 + 그 멤버 NNLS)
- 표준화 입력: 전역·all 9.518 (+it 7.550), 전역·compact 8.779 (7.564), 레짐별·compact 8.616 (7.572) → 이득 0.02~0.05, 잡음 수준.
- 분위수 변환 입력: 전역·all 9.269 (7.471, -0.125; fold_seed=1 에서도 7.510→7.396), 레짐별·all 8.768 (7.491),
  전역·compact 8.300 (7.446, -0.150), 전역·all + 잡음/드롭아웃/wd 강화 9.109 (7.476), 전역·all 3 층 256-128-64 9.122 (7.504).
  → 분위수 변환이 결정적이었다(사용량 카운터·이웃 거리 등 치우친 분포). 단일로는 LightGBM(7.718) 보다 못하지만
    잔차 상관 0.87 로 다른 오류를 내서 앙상블을 낮춘다. compact 특징이 단일·앙상블 모두 더 좋다.
- MLP 두 개(전역 compact + 레짐별 all) 를 같이 넣어도 7.412 (-0.034 추가) 로 잡음 수준이고 시간이 2 배라 하나만 쓴다.
- 단일 폴드 예비 실험: NB_trend_15 잔차 학습은 빨리 과적합해 더 나빴다(10.97 vs 10.04).

최종 (기본 4 + mlp = 전역 + 레짐 임베딩, compact 특징, 분위수 변환)
- fold_seed=0: CV 앙상블 7.446 (-0.150) (기준선 7.596), 가중치 lgb 0.62, lgb_small 0.00, extra_trees 0.02, ridge 0.02, mlp 0.34, 공식 테스트 6.791 (6.969), 검증 6.836 (6.850)
- fold_seed=1: CV 앙상블 7.372 (같은 폴드 기준선 7.510, -0.138) [캐시된 기본 멤버로 확인]
- 실행 시간: 5.4 분 (다른 실험과 CPU 를 나눈 상태, MLP 멤버 약 1.5 분; 내 탐색 5 개와 겹쳤을 때는 11.7 분)

실행: python cmp2016/exp/mlp.py                     # 최종 변형만 → cmp2016/exp/mlp.json
      python cmp2016/exp/mlp.py --all               # 탐색 변형 전부 (캐시된 기본 멤버 필요)
      python cmp2016/exp/mlp.py --screen g_quant r_quant [--seed 1]   # 지정한 변형만
      python cmp2016/exp/mlp.py --cache-base [--seed 1]                # 기본 멤버 OOF/평가 예측 캐시 (탐색용)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, get_table, run, summary  # noqa: E402
from cmp_data import TARGET  # noqa: E402

torch.set_num_threads(1)  # 작은 행렬은 스레드 동기화 비용이 더 크다 (부하 상태에서 4 스레드가 40 배 느렸다)
OUT = HERE / "mlp.json"


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ---------------------------------------------------------------------------------------------
# 기본 멤버 캐시 (탐색용): 특징이 같으므로 기본 4 멤버의 OOF/평가 예측은 고정 → 가짜 모델로 대체하면 새 멤버만 학습한다.
# xgb_cat.py 와 같은 방식. 최종 실행은 진짜 멤버로 다시 학습한다.
# ---------------------------------------------------------------------------------------------
def cache_path(seed):
    for f in [HERE / f"cache_mlp_base_s{seed}.npz", HERE / f"cache_xgb_cat_base_s{seed}.npz"]:
        if f.exists():
            return f
    return HERE / f"cache_mlp_base_s{seed}.npz"


def cache_base(seed):
    res = run(models=BASE_MODELS, fold_seed=seed, label=f"base_s{seed}", keep_pred=True)
    n = len(get_table())
    arrs = {}
    for name in BASE_MODELS:
        full = np.full(n, np.nan)
        full[res["train_idx"]] = res["oof"][name]
        full[res["eval_idx"]] = res["pred"][name]
        arrs[name] = full
    np.savez(HERE / f"cache_mlp_base_s{seed}.npz", **arrs)
    print(f"[cache] base s{seed} cv ensemble {res['cv']['ensemble']:.3f}")


class Cached:
    """캐시된 예측을 X.index 로 찾아 돌려주는 가짜 모델."""

    def __init__(self, arr):
        self.arr = arr

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.arr[np.asarray(X.index)]


def cached_models(seed):
    d = np.load(cache_path(seed))
    return {name: {"make": (lambda a=d[name]: Cached(a)), "cols": (lambda cols: ["GROUP"]), "fillna": None,
                   "per_regime": False} for name in BASE_MODELS}


# ---------------------------------------------------------------------------------------------
# MLP 회귀기 (sklearn 풍 fit/predict)
# ---------------------------------------------------------------------------------------------
def regime_id(X):
    """GROUP·STAGE_B 컬럼 → 레짐 번호 (1A=0, 4A=1, 4B=2)."""
    g = X["GROUP"].to_numpy() if "GROUP" in X else np.full(len(X), 4)
    b = X["STAGE_B"].to_numpy() if "STAGE_B" in X else np.zeros(len(X))
    return np.where(g == 1, 0, np.where(b == 0, 1, 2)).astype(np.int64)


class Net(nn.Module):
    def __init__(self, d_in, hidden, dropout, act, n_regime=0, d_embed=4):
        super().__init__()
        self.embed = nn.Embedding(n_regime, d_embed) if n_regime else None
        d = d_in + (d_embed if n_regime else 0)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), act(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x, r=None):
        if self.embed is not None:
            x = torch.cat([x, self.embed(r)], dim=1)
        return self.body(x).squeeze(1)


class TorchMLP:
    """결측: 중앙값 대체 + (중복 제거한) 결측 표시 컬럼. 스케일: 표준화(클리핑) 또는 분위수→정규.
    학습: AdamW, 전배치, 내부 분할(학습 행의 20%)로 에폭 수를 고르고 전체 학습 행으로 다시 학습, 시드 평균.
    """

    def __init__(self, hidden=(128, 64), dropout=0.1, wd=1e-3, lr=3e-3, max_epochs=200, n_seeds=5, loss="huber",
                 huber_delta=1.0, quantile=False, inner_frac=0.2, embed=False, residual=None, act="silu",
                 batch_size=None, clip=5.0, min_epochs=20, seed0=0, schedule="const", n_inner=2, noise=0.0):
        self.p = dict(hidden=hidden, dropout=dropout, wd=wd, lr=lr, max_epochs=max_epochs, n_seeds=n_seeds, loss=loss,
                      huber_delta=huber_delta, quantile=quantile, inner_frac=inner_frac, embed=embed, residual=residual,
                      act=act, batch_size=batch_size, clip=clip, min_epochs=min_epochs, seed0=seed0, schedule=schedule,
                      n_inner=n_inner, noise=noise)

    # --- 전처리 ---
    def _fit_prep(self, X):
        p = self.p
        self.cols = list(X.columns)
        A = X.to_numpy(dtype=np.float64)
        self.med = np.nanmedian(A, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        na = np.isnan(A)
        # 결측 표시: 학습에서 결측이 있는 컬럼만, 같은 패턴은 하나로
        ind_cols, seen = [], set()
        for j in range(A.shape[1]):
            if na[:, j].any():
                key = na[:, j].tobytes()
                if key not in seen:
                    seen.add(key)
                    ind_cols.append(j)
        self.ind_cols = ind_cols
        Z = self._impute(A)
        if p["quantile"]:
            self.qt = QuantileTransformer(n_quantiles=min(200, len(Z)), output_distribution="normal", random_state=0)
            Z = self.qt.fit_transform(Z)
            self.mu, self.sd = 0.0, 1.0
        else:
            self.qt = None
            self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0)
            self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        return self._transform(A)

    def _impute(self, A):
        return np.where(np.isnan(A), self.med[None, :], A)

    def _transform(self, A):
        na = np.isnan(A)
        Z = self._impute(A)
        Z = self.qt.transform(Z) if self.qt is not None else (Z - self.mu) / self.sd
        Z = np.clip(Z, -self.p["clip"], self.p["clip"])
        if self.ind_cols:
            Z = np.column_stack([Z, na[:, self.ind_cols].astype(np.float64)])
        return Z.astype(np.float32)

    def _offset(self, X):
        r = self.p["residual"]
        if r is None:
            return np.zeros(len(X))
        v = X[r].to_numpy(dtype=np.float64)
        return np.where(np.isnan(v), self.off_fill, v)

    # --- 학습 ---
    def _train(self, Z, r, t, seed, epochs, Zv=None, rv=None, tv=None):
        p = self.p
        torch.manual_seed(seed)
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[p["act"]]
        net = Net(Z.shape[1], p["hidden"], p["dropout"], act, n_regime=3 if p["embed"] else 0)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        sched = None
        if p["schedule"] == "cos":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.HuberLoss(delta=p["huber_delta"]) if p["loss"] == "huber" else nn.MSELoss()
        Zt, tt = torch.from_numpy(Z), torch.from_numpy(t.astype(np.float32))
        rt = torch.from_numpy(r) if p["embed"] else None
        n = len(Z)
        bs = p["batch_size"] or n
        g = torch.Generator().manual_seed(seed)
        curve = []
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g) if bs < n else torch.arange(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                xb = Zt[idx]
                if p["noise"] > 0:
                    xb = xb + p["noise"] * torch.randn(xb.shape, generator=g)
                out = net(xb, rt[idx] if rt is not None else None)
                loss = loss_fn(out, tt[idx])
                loss.backward()
                opt.step()
            if sched is not None:
                sched.step()
            if Zv is not None:
                net.eval()
                with torch.no_grad():
                    pv = net(torch.from_numpy(Zv), torch.from_numpy(rv) if p["embed"] else None).numpy()
                curve.append(float(np.mean((pv - tv) ** 2)))
        return net, curve

    def fit(self, X, y):
        p = self.p
        y = np.asarray(y, dtype=np.float64)
        if p["residual"] is not None:
            v = X[p["residual"]].to_numpy(dtype=np.float64)
            self.off_fill = float(np.nanmean(v)) if np.isnan(v).any() else 0.0
        Z = self._fit_prep(X)
        r = regime_id(X)
        t = y - self._offset(X)
        self.t_mu, self.t_sd = t.mean(), t.std() + 1e-9
        t = (t - self.t_mu) / self.t_sd
        n = len(Z)
        seeds = [p["seed0"] + s for s in range(p["n_seeds"])]
        # 내부 분할로 에폭 수 선택 (n_inner 개 시드, 시드마다 다른 분할, 곡선 평균의 최소)
        curves = []
        for s in seeds[:p["n_inner"]]:
            rng = np.random.RandomState(1000 + s)
            perm = rng.permutation(n)
            nv = max(int(round(n * p["inner_frac"])), 10)
            va, tr = perm[:nv], perm[nv:]
            _, curve = self._train(Z[tr], r[tr], t[tr], s, p["max_epochs"], Z[va], r[va], t[va])
            curves.append(curve)
        c = np.mean(curves, axis=0)
        k = 5
        cs = np.convolve(c, np.ones(k) / k, mode="valid")  # 이동 평균으로 잡음 완화
        best = int(np.argmin(cs[p["min_epochs"]:]) + p["min_epochs"] + k // 2)
        self.best_epochs, self.inner_curve = best, c
        self.nets = [self._train(Z, r, t, s, best)[0] for s in seeds]
        return self

    def predict(self, X):
        Z = self._transform(X[self.cols].to_numpy(dtype=np.float64))
        r = torch.from_numpy(regime_id(X)) if self.p["embed"] else None
        Zt = torch.from_numpy(Z)
        outs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                outs.append(net(Zt, r).numpy())
        return np.mean(outs, axis=0) * self.t_sd + self.t_mu + self._offset(X)


def mlp_spec(cols="all", per_regime=False, **kw):
    return {"make": lambda: TorchMLP(**kw), "cols": cols, "fillna": None, "per_regime": per_regime}


# ---------------------------------------------------------------------------------------------
# 탐색 변형
# ---------------------------------------------------------------------------------------------
VARIANTS = {
    "g_all": mlp_spec("all", False, embed=True),
    "g_compact": mlp_spec("compact", False, embed=True),
    "g_quant": mlp_spec("all", False, embed=True, quantile=True),
    "r_all": mlp_spec("all", True),
    "r_compact": mlp_spec("compact", True),
    "r_quant": mlp_spec("all", True, quantile=True),
    "g_quant_compact": mlp_spec("compact", False, embed=True, quantile=True),
    "g_quant_reg": mlp_spec("all", False, embed=True, quantile=True, noise=0.1, dropout=0.2, wd=1e-2),
    "g_quant_big": mlp_spec("all", False, embed=True, quantile=True, hidden=(256, 128, 64), dropout=0.2),
}


def blend(oof, y, names):
    A = np.column_stack([oof[n] for n in names])
    w, _ = nnls(A, y)
    w = w / w.sum()
    return mse(A @ w, y), dict(zip(names, map(float, w)))


def screen(cands, seed, label):
    """캐시된 기본 4 멤버 + 후보들: 후보 하나씩 더했을 때의 CV 앙상블."""
    t0 = time.time()
    res = run(models={**cached_models(seed), **cands}, fold_seed=seed, label=label, keep_pred=True)
    y = get_table()[TARGET].to_numpy()[res["train_idx"]]
    oof = {k: np.asarray(v) for k, v in res["oof"].items()}
    base = list(BASE_MODELS)
    b0, _ = blend(oof, y, base)
    print(f"\n[screen {label} s{seed}] base4 cv {b0:.3f}  ({time.time() - t0:.0f}s)")
    for c in cands:
        m1, w1 = blend(oof, y, base + [c])
        print(f"  {c:18s} single {res['cv'][c]:.3f}  base4+it {m1:.3f} (Δ{m1 - b0:+.3f})  weight {w1[c]:.2f}")
    if len(cands) > 1:
        m_all, w_all = blend(oof, y, base + list(cands))
        print(f"  base4+all          {m_all:.3f} (Δ{m_all - b0:+.3f})  "
              + ", ".join(f"{k} {v:.2f}" for k, v in w_all.items() if v > 0.005))
    np.savez(HERE / f"cache_mlp_screen_{label}_s{seed}.npz", **oof, y=y)
    return res


FINAL_NEW = {"mlp": VARIANTS["g_quant_compact"]}  # 전역 + 레짐 임베딩, compact 특징, 분위수 변환


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="탐색 변형 전부 (캐시된 기본 멤버 사용)")
    p.add_argument("--cache-base", action="store_true")
    p.add_argument("--screen", nargs="*", help="탐색 변형 이름들")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cached", action="store_true")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    if a.cache_base:
        cache_base(a.seed)
        return
    if a.all or a.screen is not None:
        names = list(VARIANTS) if a.all or not a.screen else a.screen
        screen({k: VARIANTS[k] for k in names}, a.seed, "_".join(names)[:40])
        return
    t0 = time.time()
    models = {**cached_models(a.seed), **FINAL_NEW} if a.cached else {**BASE_MODELS, **FINAL_NEW}
    res = run(models=models, fold_seed=a.seed, label="mlp", out=None if a.cached else a.out)
    print(summary(res))
    print(f"[time] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
