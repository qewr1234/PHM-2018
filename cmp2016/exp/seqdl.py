"""seqdl: 원시 센서 시퀀스 위의 1D CNN + 표 특징 MLP 융합 신경망(PyTorch CPU)을 앙상블 멤버로 추가.

아이디어 (Appl. Sci. 2022 "deep + shallow fusion", 테스트 MSE 6.36)
- harness.sequences(n_points) 로 챔버 단계(STEP 0,1,2)별 13 채널 궤적을 고정 길이로 만들고, 시간축이 [0,1] 로
  정규화되며 사라지는 시간 정보를 "경과 초" 채널(격자 × ACTIVE_SEC_s{step})로 되살린다 (14 채널).
  채널별 표준화(학습 행 기준), 없는 단계는 0 + 단계 존재 마스크.
- 모델: 단계 공유 1D CNN(32폭 3층, 평균+최대 풀링) → 단계별 벡터를 마스크로 가리고 이어붙임 + 레짐 임베딩(4)
  + compact 표 특징(분위수→정규 변환 + 결측 표시) MLP 가지 → 헤드(64) → 1.
  mode: fusion(시퀀스 + 표), seq(시퀀스만), tab(표만 = mlp.py 와 같은 조건의 대조군).
- 학습: 목표 표준화, Huber, AdamW(lr 3e-3, wd 1e-3), 미니배치 256. 에폭 수는 학습 행의 20% 내부 분할
  (시드 2 개, 곡선 평균의 이동평균 최소)로 고른 뒤 전체 학습 행으로 다시 학습, 시드 3 개 평균.
- 탐색은 mlp.py 와 같이 기본 4 멤버의 OOF/평가 예측 캐시(cache_xgb_cat_base_s*.npz)로 새 멤버만 학습했다.

시도한 것 (5-fold CV 앙상블, fold_seed=0, 기준선 7.596; "단일" 은 그 멤버 CV, "+it" 은 기본 4 + 그 멤버 NNLS)
- fusion (활성 구간 32 점, 표 + 시퀀스)        단일 8.089, +it 7.296 (-0.300)
- tab (표만, 같은 학습 조건의 대조군)          단일 8.449, +it 7.476 (-0.120; mlp.py 의 7.446 과 같은 수준)
  → 시퀀스 가지가 앙상블을 0.18 더 낮춘다.
- seq (시퀀스만 + 레짐 임베딩)                 단일 25.432 (시퀀스만으로는 약하다: 레짐 평균 ~51 보다는 낫지만 이웃 특징에 못 미친다)
- fusion_c (목표에서 레짐별 학습 평균을 뺌)     단일 7.757, +it 7.243 (-0.353); fold_seed=1: 7.510 → 7.209 (-0.301)
- tab_c                                        단일 8.212, +it 7.478 (-0.118)
- fusion_all64 (기록 전체 64 점, 중심화 없음)   단일 8.118
- fusion_all64_c (기록 전체 64 점 + 레짐 중심화) 단일 7.506, +it 7.108 (-0.488), 가중치 0.55 ← 최종
  fold_seed=1 확인: 기준선 7.510 → 6.963 (-0.546), 단일 7.362, 가중치 0.56.
  → 챔버 그룹 1 웨이퍼의 56% 는 활성 행(가압+슬러리 C)이 없어 활성 구간 시퀀스가 비어 있었다.
    기록 전체(비활성 포함)를 쓰면 모든 웨이퍼·단계가 채워지고(99.9%), 램프·대기 구간도 보인다.
  → 레짐 중심화: 풀링된 목표 표준편차 30 중 대부분이 레짐 간 차이라 임베딩이 이를 배우느라 낭비되던 용량을 레짐 안 변동에 쓴다.
- fusion(32) + fusion_c 를 같이 넣은 NNLS 7.211: 멤버 하나로 충분. fusion_r(레짐별 학습)은 시간 부족으로 중단.
- 스레드: 2 개가 가장 빨랐다 (에폭당 0.20 s; 1 개 0.28 s, 4 개 0.34 s, 다른 실험과 CPU 공유 상태).

최종 (python cmp2016/exp/seqdl.py, 기본 4 멤버 + seqdl, fold_seed=0, seqdl.json)
- CV 앙상블 7.108 (기준선 7.596), 가중치 lgb 0.45 / seqdl 0.55 / 나머지 0
- 테스트 6.425 (기준선 6.969; 1A 9.706, 4A 5.681, 4B 5.798), 검증 6.409 (기준선 6.850). seqdl 단일: 테스트 6.599, 검증 6.986
- 실행 시간 11.1 분 (다른 실험과 CPU 공유; 스레드 2, SEQDL_THREADS 로 조정)

실행: python cmp2016/exp/seqdl.py                   # 최종 변형만 → cmp2016/exp/seqdl.json
      python cmp2016/exp/seqdl.py --all             # 탐색 변형 전부 (캐시된 기본 멤버 필요)
      python cmp2016/exp/seqdl.py --screen fusion seq [--seed 1]
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, get_raw, get_table, run, sequences, summary  # noqa: E402
from cmp_data import KEY, TARGET  # noqa: E402

N_THREADS = int(os.environ.get("SEQDL_THREADS", "2"))
torch.set_num_threads(N_THREADS)
OUT = HERE / "seqdl.json"


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ---------------------------------------------------------------------------------------------
# 시퀀스 텐서: 표 행 순서(RangeIndex)로 정렬해 X.index 로 바로 찾는다. 정답은 쓰지 않는다.
# ---------------------------------------------------------------------------------------------
_SEQ = {}


def load_seq(n_points, active_only=True):
    """(seq[n_table, 3, 14, L] float32, 없는 단계는 NaN; present[n_table, 3] bool).

    active_only=False: 가압·슬러리 구간만이 아니라 기록 전체 (챔버 그룹 1 웨이퍼의 56% 는 활성 행이 없다).
    """
    key = (n_points, active_only)
    if key in _SEQ:
        return _SEQ[key]
    f = HERE / f"cache_seqdl_{n_points}{'' if active_only else '_all'}.npz"
    if not f.exists():
        keys, X = sequences(get_raw(), n_points=n_points, active_only=active_only)
        np.savez_compressed(f, wafer=keys["WAFER_ID"].to_numpy(), stage=keys["STAGE"].to_numpy().astype(str), X=X)
    d = np.load(f)
    keys = pd.DataFrame({"WAFER_ID": d["wafer"], "STAGE": d["stage"]})
    table = get_table()
    assert (table.index.to_numpy() == np.arange(len(table))).all()
    pos = table[KEY].merge(keys.reset_index().rename(columns={"index": "pos"}), on=KEY, how="left")["pos"].to_numpy()
    assert not np.isnan(pos.astype(float)).any()
    X = d["X"][pos.astype(int)]  # [n, 3, L, 13]
    L = X.shape[2]
    # 경과 초 채널: 정규화된 격자 × 단계 길이(활성 초, 기록 전체면 행 수) (시간 정보 복원)
    if active_only:
        step_len = np.nan_to_num(table[[f"ACTIVE_SEC_s{s}" for s in range(3)]].to_numpy(dtype=np.float32))
    else:
        raw = get_raw()
        cnt = raw.groupby(KEY + ["STEP"]).size().unstack("STEP").reindex(columns=range(3))
        step_len = np.nan_to_num(table[KEY].merge(cnt.reset_index(), on=KEY, how="left")[list(range(3))]
                                 .to_numpy(dtype=np.float32))
    elapsed = np.linspace(0, 1, L, dtype=np.float32)[None, None, :, None] * step_len[:, :, None, None]
    X = np.concatenate([X, elapsed], axis=3)
    present = ~np.isnan(X[:, :, 0, 0])
    X[~present] = np.nan
    seq = np.ascontiguousarray(X.transpose(0, 1, 3, 2))  # [n, 3, C, L]
    _SEQ[key] = (seq, present)
    return _SEQ[key]


def regime_id(X):
    """GROUP·STAGE_B 컬럼 → 레짐 번호 (1A=0, 4A=1, 4B=2)."""
    g = X["GROUP"].to_numpy() if "GROUP" in X else np.full(len(X), 4)
    b = X["STAGE_B"].to_numpy() if "STAGE_B" in X else np.zeros(len(X))
    return np.where(g == 1, 0, np.where(b == 0, 1, 2)).astype(np.int64)


# ---------------------------------------------------------------------------------------------
# 신경망
# ---------------------------------------------------------------------------------------------
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
    def __init__(self, c_in, d_tab, mode, width=32, d_hidden=128, dropout=0.1, d_embed=4, n_regime=3, d_head=64):
        super().__init__()
        self.mode = mode
        self.embed = nn.Embedding(n_regime, d_embed)
        d = d_embed
        self.enc = self.tab = None
        if mode in ("fusion", "seq"):
            self.enc = SeqEncoder(c_in, width)
            self.seq_drop = nn.Dropout(dropout)
            d += 3 * 2 * width + 3
        if mode in ("fusion", "tab"):
            self.tab = nn.Sequential(nn.Linear(d_tab, d_hidden), nn.SiLU(), nn.Dropout(dropout))
            d += d_hidden
        self.head = nn.Sequential(nn.Linear(d, d_head), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_head, 1))

    def forward(self, seq, mask, tab, r):
        parts = [self.embed(r)]
        if self.enc is not None:
            B, S, C, L = seq.shape
            h = self.enc(seq.reshape(B * S, C, L)).reshape(B, S, -1) * mask[:, :, None]
            parts += [self.seq_drop(h.reshape(B, -1)), mask]
        if self.tab is not None:
            parts.append(self.tab(tab))
        return self.head(torch.cat(parts, 1)).squeeze(1)


# ---------------------------------------------------------------------------------------------
# sklearn 풍 회귀기: X.index 로 시퀀스를 찾는다
# ---------------------------------------------------------------------------------------------
class SeqDL:
    def __init__(self, mode="fusion", n_points=32, active_only=True, width=32, d_hidden=128, dropout=0.1, wd=1e-3, lr=3e-3,
                 max_epochs=120, min_epochs=10, patience=30, n_seeds=3, n_inner=2, inner_frac=0.2, batch_size=256,
                 huber_delta=1.0, clip=5.0, seed0=0, noise=0.0, center=None):
        self.p = dict(mode=mode, n_points=n_points, active_only=active_only, width=width, d_hidden=d_hidden, dropout=dropout, wd=wd, lr=lr,
                      max_epochs=max_epochs, min_epochs=min_epochs, patience=patience, n_seeds=n_seeds, n_inner=n_inner,
                      inner_frac=inner_frac, batch_size=batch_size, huber_delta=huber_delta, clip=clip, seed0=seed0,
                      noise=noise, center=center)

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

    # --- 시퀀스 전처리 (채널별 표준화, 결측 0 + 마스크) ---
    def _fit_seq(self, idx):
        seq, present = load_seq(self.p["n_points"], self.p["active_only"])
        s = seq[idx]
        self.s_mu = np.nanmean(s, axis=(0, 1, 3))
        self.s_sd = np.nanstd(s, axis=(0, 1, 3))
        self.s_sd = np.where(self.s_sd < 1e-6, 1.0, self.s_sd)
        return self._seq(idx)

    def _seq(self, idx):
        seq, present = load_seq(self.p["n_points"], self.p["active_only"])
        z = (seq[idx] - self.s_mu[None, None, :, None]) / self.s_sd[None, None, :, None]
        z = np.nan_to_num(np.clip(z, -self.p["clip"], self.p["clip"]), nan=0.0).astype(np.float32)
        return z, present[idx].astype(np.float32)

    def _inputs(self, X, fit=False):
        idx = np.asarray(X.index)
        p = self.p
        seq = mask = tab = None
        if p["mode"] in ("fusion", "seq"):
            seq, mask = (self._fit_seq if fit else self._seq)(idx)
        if p["mode"] in ("fusion", "tab"):
            tab = (self._fit_tab if fit else self._tab)(X)
        return seq, mask, tab, regime_id(X)

    # --- 학습 ---
    @staticmethod
    def _tensors(parts, sel=None):
        out = []
        for a in parts:
            if a is None:
                out.append(None)
            else:
                out.append(torch.from_numpy(a if sel is None else a[sel]))
        return out

    def _train(self, inp, t, seed, epochs, val=None):
        p = self.p
        torch.manual_seed(seed)
        seq, mask, tab, r = inp
        net = FusionNet(0 if seq is None else seq.shape[2], 0 if tab is None else tab.shape[1], p["mode"],
                        width=p["width"], d_hidden=p["d_hidden"], dropout=p["dropout"])
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
                sb = None if S is None else S[b]
                tb = None if T is None else T[b]
                if p["noise"] > 0:
                    if sb is not None:
                        sb = sb + p["noise"] * torch.randn(sb.shape, generator=g)
                    if tb is not None:
                        tb = tb + p["noise"] * torch.randn(tb.shape, generator=g)
                loss = loss_fn(net(sb, None if M is None else M[b], tb, R[b]), tt[b])
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
        # center="regime": 레짐별 학습 평균을 빼서 레짐 간 차이가 아닌 레짐 안의 변동을 학습한다
        self.r_mu = np.array([t[r == k].mean() if (r == k).any() else t.mean() for k in range(3)]) \
            if p["center"] == "regime" else np.zeros(3)
        t = t - self.r_mu[r]
        self.t_mu, self.t_sd = t.mean(), t.std() + 1e-9
        t = (t - self.t_mu) / self.t_sd
        n = len(t)
        seeds = [p["seed0"] + s for s in range(p["n_seeds"])]
        curves = []
        for s in seeds[:p["n_inner"]]:  # 내부 분할(학습 행의 20%)로 에폭 수 선택
            perm = np.random.RandomState(1000 + s).permutation(n)
            nv = max(int(round(n * p["inner_frac"])), 10)
            va, tr = perm[:nv], perm[nv:]
            sub = tuple(None if a is None else a[tr] for a in inp)
            vsub = tuple(None if a is None else a[va] for a in inp)
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


def spec(mode="fusion", per_regime=False, **kw):
    cols = (lambda c: ["GROUP", "STAGE_B"]) if mode == "seq" else "compact"
    return {"make": lambda: SeqDL(mode=mode, **kw), "cols": cols, "fillna": None, "per_regime": per_regime}


# ---------------------------------------------------------------------------------------------
# 기본 멤버 캐시로 탐색 (mlp.py / xgb_cat.py 와 같은 방식)
# ---------------------------------------------------------------------------------------------
class Cached:
    def __init__(self, arr):
        self.arr = arr

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.arr[np.asarray(X.index)]


def cached_models(seed):
    f = HERE / f"cache_xgb_cat_base_s{seed}.npz"
    d = np.load(f)
    return {name: {"make": (lambda a=d[name]: Cached(a)), "cols": (lambda cols: ["GROUP"]), "fillna": None,
                   "per_regime": False} for name in BASE_MODELS}


def blend(oof, y, names):
    A = np.column_stack([oof[n] for n in names])
    w, _ = nnls(A, y)
    w = w / w.sum()
    return mse(A @ w, y), dict(zip(names, map(float, w)))


VARIANTS = {
    "fusion": spec("fusion"),
    "seq": spec("seq", max_epochs=250, patience=40),
    "tab": spec("tab"),
    "fusion_c": spec("fusion", center="regime"),
    "seq_c": spec("seq", max_epochs=250, patience=40, center="regime"),
    "tab_c": spec("tab", center="regime"),
    "fusion_r": spec("fusion", per_regime=True),
    "fusion_all64": spec("fusion", n_points=64, active_only=False),
    "fusion_all64_c": spec("fusion", n_points=64, active_only=False, center="regime"),
    "fusion64_c": spec("fusion", n_points=64, center="regime"),
    "seq_all64": spec("seq", n_points=64, active_only=False, max_epochs=250, patience=40),
    "fusion64": spec("fusion", n_points=64),
    "seq64": spec("seq", n_points=64),
    "fusion_w64": spec("fusion", width=64, dropout=0.2),
    "fusion_noise": spec("fusion", noise=0.1, dropout=0.2, wd=1e-2),
    "fusion_s5": spec("fusion", n_seeds=5),
}


def screen(cands, seed, label):
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
    np.savez(HERE / f"cache_seqdl_screen_{label}_s{seed}.npz", **oof, y=y)
    return res


FINAL_NEW = {"seqdl": VARIANTS["fusion_all64_c"]}  # 기록 전체 64 점 + 레짐 중심화 융합망


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="탐색 변형 전부 (캐시된 기본 멤버 사용)")
    p.add_argument("--screen", nargs="*", help="탐색 변형 이름들")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cached", action="store_true")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    if a.all or a.screen is not None:
        names = list(VARIANTS) if a.all or not a.screen else a.screen
        screen({k: VARIANTS[k] for k in names}, a.seed, "_".join(names)[:40])
        return
    t0 = time.time()
    models = {**cached_models(a.seed), **FINAL_NEW} if a.cached else {**BASE_MODELS, **FINAL_NEW}
    res = run(models=models, fold_seed=a.seed, label="seqdl", out=None if a.cached else a.out)
    print(summary(res))
    print(f"[time] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
