"""스태킹 실험 (stacking): 멤버 결합 방식과 후처리 개선 (OOF 행 위 nested 5-fold CV 로 채점).

아이디어: 기준 앙상블은 4 멤버(lgb, lgb_small, extra_trees, ridge)의 OOF 를 전역 NNLS 로 섞는다. 이 결합 단계를
(a) 레짐별 NNLS, (b) 절편·메타 특징을 넣은 NNLS/릿지 스태커, (c) LightGBM 메타 학습기, (d) 시간 평활,
(e) 국소 참조 평균으로의 수축으로 바꿔 본다. harness.run(keep_pred=True) 의 OOF·평가 예측을 캐시하고,
스태커는 OOF 행 위에서 멤버와 같은 5-fold 로 nested 채점한다 (학습 폴드에서만 맞추고 held-out 폴드에 적용).

핵심 발견 (seed 0, base4)
- 기준의 "CV 앙상블 7.596" 은 전체 OOF 로 맞춘 NNLS 를 같은 행에 채점한 in-sample 값. 정직한 nested 값은 8.109.
  원인: ridge 멤버가 4B 한 행(y 73.1)에 -84.7 을 예측(ridge CV 21.27 의 대부분). 그 행이 든 폴드를 빼고 NNLS 를 맞추면
  ridge 가중치가 0.04 → 0.22 로 뒤집혀 held-out 폴드 MSE 9.5. 전역 in-sample 가중치는 그 행 덕에 "운 좋게" ridge 를 억눌렀다.
- 해결: 멤버 예측을 레짐별 학습 정답 범위 [min, max] 로 자른다 (클리핑, 정답은 학습 행만 사용). 전역 NNLS nested 7.592.
- 레짐별 NNLS + 클리핑 nested 7.543 (in-sample 7.490). 1A 는 lgb 대신 extra_trees·ridge 가 주력(가중치 0.5+)이라 레짐별 가중치가 유효.

시도한 것 (nested CV, seed 0; 클리핑 전 / 후)
- 전역 NNLS 8.109 / 7.592, +절편 8.158 / -, 정규화 없음 8.129
- 레짐별 NNLS 7.652 / 7.543, +절편 7.755 / 7.62, 레짐별→전역 수축(β 0.25~0.75) 7.65~7.75 (β=1 이 최선)
- 릿지 스태커 [예측, 레짐, 메타(NB_time_dist1, NB_time_3/10, NB_trend_15, ACTIVE_SEC, 드레서·패드 사용량, 간격 …)]
  P+레짐 8.166, +메타 7.889, +교호작용 8.134, +둘 다 7.861 (클리핑 후 7.5~7.9, 레짐별 NNLS 보다 못함)
- LightGBM 메타 (직접 8.335, 레짐별 NNLS 잔차 + 메타 shrink 0.5/1: 7.86~8.28; 클리핑 후도 이득 없음)
- (e) NB_time_3 / NB_trend_15 를 멤버로 추가하거나 λ 로 수축: 이득 없음 (λ≈0 선택)
- (d) 시간 평활 (같은 폴드 안 이웃의 커널 평균, h·λ nested 선택): 전역 8.109 → 8.101, 레짐별 7.623 → 7.623 (이득 없음).
  같은 폴드 이웃의 잔차 상관은 h=0.5h 에서 0.5 로 크지만 예측 평활로는 못 쓴다 (정답이 필요; 이미 NB_* 특징이 담당).
- 추가 멤버: et_all(전체 컬럼 ExtraTrees, 멤버 8.27) 은 전역 NNLS 7.570(in-sample) 로 -0.026, 레짐별+클리핑 7.517 (base4 7.543, 차이 0.026 < 0.05);
  knn(compact, 멤버 11.1) 은 가중치 0. 단순함을 위해 base4 유지.

최종: base4 + 멤버 예측 클리핑(레짐별 학습 정답 범위) + 레짐별 NNLS (멤버 학습은 그대로, 결합 단계만 바뀜).
- seed 0: nested CV 7.543 (Δ-0.053 vs in-sample 기준 7.596; nested 전역 8.109, 클리핑 전역 7.592 대비 -0.566 / -0.049),
  테스트 6.856 (기준 6.969), 검증 6.789 (기준 6.850). 레짐별 가중치 1A: et 0.53 ridge 0.37 lgb 0.10 /
  4A: lgb 0.71 ridge 0.18 et 0.11 / 4B: lgb 0.60 lgb_small 0.26 ridge 0.14.
- seed 1 확인: nested CV 7.399 (in-sample 기준 7.510 대비 -0.111; 클리핑 전역 7.483 대비 -0.084), 테스트 6.886, 검증 6.798
  (같은 폴드 harness 전역 NNLS 테스트 6.960, 검증 6.785). 레짐별 가중치는 시드에 따라 1A 에서 et↔ridge 가 바뀌지만 CV 이득은 두 시드 모두 확인.
- 결과 JSON 의 cv.ensemble / test.ensemble / val.ensemble 은 스태킹 결과, *nnls_insample*/nnls_ensemble/weights_nnls 는
  harness 원래 전역 NNLS 값. 나머지 스태커·후처리는 모두 레짐별 NNLS 보다 못했다 (seed 0/1 모두).

실행: python cmp2016/exp/stacking.py             # 최종 변형만 (harness 1회 + 스태킹, 캐시 없으면 3~7분), stacking.json 저장
      python cmp2016/exp/stacking.py --seed 1    # fold_seed 1 확인
      python cmp2016/exp/stacking.py --all       # 탐색 변형 (base4 + et_all + knn 캐시 위에서 스태커 비교)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, BASELINE, N_FOLDS, base_features, get_table, run, summary  # noqa: E402
from cmp_data import TARGET, regime  # noqa: E402
from run_cmp import mse  # noqa: E402

NJ = 4
BASE4 = ["lgb", "lgb_small", "extra_trees", "ridge"]
# 값싼 추가 멤버 (탐색용): 전체 컬럼 ExtraTrees, compact 컬럼 KNN
EXTRA_MODELS = {
    "et_all": {"make": lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.3,
                                                   n_jobs=NJ, random_state=1),
               "cols": "all", "fillna": -999, "per_regime": True},
    "knn": {"make": lambda: make_pipeline(SimpleImputer(), StandardScaler(),
                                          KNeighborsRegressor(n_neighbors=15, weights="distance")),
            "cols": "compact", "fillna": None, "per_regime": True},
}
# 스태커 메타 특징 (폴드별 특징 캐시에서 읽는다; 이웃 특징은 참조 폴드 기준이라 폴드마다 다름)
META_COLS = ["NB_time_dist1", "NB_time_3", "NB_time_10", "NB_trend_15", "regime_mu", "ACTIVE_SEC",
             "USAGE_OF_DRESSER_start", "USAGE_OF_POLISHING_TABLE_start", "same_prev_gap", "same_next_gap",
             "OTHER_STAGE_RATE"]


# ----------------------------------------------------------------------------- 1단계: 멤버 OOF·평가 예측
def level1(seed, members, tag):
    """멤버들의 OOF·평가 예측과 메타 특징을 만들고 cache_stacking_{tag}_s{seed}.npz 에 캐시한다."""
    f = HERE / f"cache_stacking_{tag}_s{seed}.npz"
    table = get_table()
    if f.exists():
        z = np.load(f, allow_pickle=True)
        L = {k: ((z[k].item() if z[k].ndim == 0 else z[k].tolist()) if z[k].dtype == object else z[k]) for k in z.files}
    else:
        t0 = time.time()
        models = {**BASE_MODELS, **{k: EXTRA_MODELS[k] for k in members if k in EXTRA_MODELS}}
        res = run(models=models, fold_seed=seed, label=f"l1_{tag}_s{seed}", keep_pred=True)
        tr, ev = np.asarray(res["train_idx"]), np.asarray(res["eval_idx"])
        names = list(models)
        oof = np.column_stack([res["oof"][k] for k in names])
        pred = np.column_stack([res["pred"][k] for k in names])
        # 메타 특징: 학습 행은 자기가 빠진 폴드의 특징표, 평가 행은 전체 학습 기준 특징표
        train_mask = ((table["split"] == "train") & ~table["OUTLIER"]).to_numpy()
        folds = list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr))
        meta_tr = np.full((len(tr), len(META_COLS)), np.nan)
        fold_id = np.zeros(len(tr), int)
        for k, (a, b) in enumerate(folds):
            ref = np.zeros(len(table), bool)
            ref[tr[a]] = True
            X = base_features(table, ref, f"s{seed}_k{N_FOLDS}_f{k}")
            meta_tr[b] = X.iloc[tr[b]][META_COLS].to_numpy(dtype=float)
            fold_id[b] = k
        X_full = base_features(table, train_mask, "full")
        meta_ev = X_full.iloc[ev][META_COLS].to_numpy(dtype=float)
        L = {"names": names, "oof": oof, "pred": pred, "train_idx": tr, "eval_idx": ev, "fold_id": fold_id,
             "meta_tr": meta_tr, "meta_ev": meta_ev, "meta_cols": META_COLS,
             "res": {k: v for k, v in res.items() if k not in ("oof", "pred", "train_idx", "eval_idx")}}
        np.savez(f, **{k: (np.array(v, dtype=object) if isinstance(v, (dict, list)) else v) for k, v in L.items()})
        print(f"[level1] cached {f.name} ({time.time() - t0:.0f}s)", flush=True)
    tr, ev = L["train_idx"], L["eval_idx"]
    y, reg = table[TARGET].to_numpy(), regime(table)
    t = table["T_START"].to_numpy(dtype=float) / 3600.0
    L.update(y_tr=y[tr], y_ev=y[ev], R_tr=reg[tr], R_ev=reg[ev], t_tr=t[tr], t_ev=t[ev],
             split_ev=table["split"].to_numpy()[ev])
    return L


# ----------------------------------------------------------------------------- 스태커
class NNLSStack:
    """비음수 최소제곱 결합. per_regime: 레짐별 가중치, intercept: 부호 자유 절편(+1, -1 열), normalize: 합 1."""

    def __init__(self, per_regime=False, intercept=False, normalize=True):
        self.per_regime, self.intercept, self.normalize = per_regime, intercept, normalize

    def _groups(self, R):
        return {r: R == r for r in np.unique(R)} if self.per_regime else {"all": np.ones(len(R), bool)}

    def _design(self, P, m):
        A = P[m]
        return np.column_stack([A, np.ones(m.sum()), -np.ones(m.sum())]) if self.intercept else A

    def fit(self, P, M, R, y):
        self.w = {}
        for r, m in self._groups(R).items():
            w, _ = nnls(self._design(P, m), y[m])
            if self.normalize and not self.intercept and w.sum() > 0:
                w = w / w.sum()
            self.w[r] = w
        return self

    def predict(self, P, M, R):
        out = np.zeros(len(R))
        for r, m in self._groups(R).items():
            if r in self.w:
                out[m] = self._design(P, m) @ self.w[r]
        return out


class RidgeStack:
    """릿지 스태커: [멤버 예측, 레짐 지시자, (메타 특징), (예측×레짐 교호작용)] 표준화 후 RidgeCV (학습 행 LOO 로 alpha)."""

    def __init__(self, meta=False, interact=False, alphas=np.logspace(-2, 4, 25)):
        self.meta, self.interact, self.alphas = meta, interact, alphas

    def _design(self, P, M, R):
        onehot = (R[:, None] == self.regs[None, :]).astype(float)
        cols = [P, onehot]
        if self.interact:
            cols += [P * onehot[:, [j]] for j in range(len(self.regs))]
        if self.meta:
            cols.append(M)
        return np.column_stack(cols)

    def fit(self, P, M, R, y):
        self.regs = np.unique(R)
        self.m = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=self.alphas)).fit(self._design(P, M, R), y)
        return self

    def predict(self, P, M, R):
        return self.m.predict(self._design(P, M, R))


LGB_META = dict(n_estimators=300, learning_rate=0.02, num_leaves=4, min_child_samples=30, subsample=0.7,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=0, verbose=-1, n_jobs=1)


class LGBStack:
    """LightGBM 메타 학습기. residual: 기본 스태커(base) 잔차를 학습해 shrink 배로 더한다. meta: 메타 특징 사용."""

    def __init__(self, residual=True, meta=True, shrink=1.0, base=None, **params):
        import lightgbm as lgb
        self.lgb, self.residual, self.meta, self.shrink = lgb, residual, meta, shrink
        self.base = base or (lambda: NNLSStack(per_regime=True))
        self.params = {**LGB_META, **params}

    def _design(self, P, M, R):
        code = pd.Series(R).map({r: i for i, r in enumerate(self.regs)}).to_numpy(dtype=float)
        return np.column_stack([P, code[:, None]] + ([M] if self.meta else []))

    def fit(self, P, M, R, y):
        self.regs = np.unique(R)
        off = np.zeros(len(y))
        if self.residual:
            self.b = self.base().fit(P, M, R, y)
            off = self.b.predict(P, M, R)
        self.m = self.lgb.LGBMRegressor(**self.params).fit(self._design(P, M, R), y - off)
        return self

    def predict(self, P, M, R):
        off = self.b.predict(P, M, R) if self.residual else 0.0
        return off + self.shrink * self.m.predict(self._design(P, M, R))


def clip_members(P, R, y_ref, R_ref):
    """멤버 예측을 레짐별 참조(학습) 정답 범위 [min, max] 로 자른다 (ridge 의 파국적 외삽 방지)."""
    Q = P.copy()
    for r in np.unique(R):
        yr = y_ref[R_ref == r]
        Q[R == r] = np.clip(P[R == r], yr.min(), yr.max())
    return Q


def nested(make, P, M, R, y, fold_id, clip=True):
    """OOF 행 위 5-fold nested 채점: 스태커를 학습 폴드에서 맞추고 held-out 폴드를 예측 (폴드는 멤버 OOF 와 동일)."""
    out = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        a, b = fold_id != k, fold_id == k
        Pa, Pb = (clip_members(P[a], R[a], y[a], R[a]), clip_members(P[b], R[b], y[a], R[a])) if clip else (P[a], P[b])
        out[b] = make().fit(Pa, M[a], R[a], y[a]).predict(Pb, M[b], R[b])
    return out


# ----------------------------------------------------------------------------- 후처리 (d), (e)
def kernel_avg(p, t, R, pool, h):
    """같은 레짐·같은 풀(pool) 안에서 자기 자신을 뺀 시간 가우시안 커널(대역폭 h 시간) 가중 예측 평균."""
    out = p.copy()
    for key in set(zip(R, pool)):
        idx = np.flatnonzero((R == key[0]) & (pool == key[1]))
        if len(idx) < 2:
            continue
        K = np.exp(-0.5 * (np.abs(t[idx][:, None] - t[idx][None, :]) / h) ** 2)
        np.fill_diagonal(K, 0.0)
        den = K.sum(1)
        out[idx] = np.where(den > 1e-6, (K @ p[idx]) / np.maximum(den, 1e-12), p[idx])
    return out


H_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0)
LAM_GRID = np.round(np.arange(0.0, 0.81, 0.05), 2)


def nested_post(p, q_by_h, y, fold_id):
    """p 와 보조 예측 q 의 혼합 (1-λ)p + λq: (h, λ) 를 outer 학습 행에서만 고르고 held-out 폴드에 적용."""
    out = np.full(len(y), np.nan)
    picks = []
    for k in range(N_FOLDS):
        a, b = fold_id != k, fold_id == k
        best = min((mse((1 - lam) * p[a] + lam * q[a], y[a]), h, lam) for h, q in q_by_h.items() for lam in LAM_GRID)
        _, h, lam = best
        out[b] = (1 - lam) * p[b] + lam * q_by_h[h][b]
        picks.append((h, float(lam)))
    return out, picks


# ----------------------------------------------------------------------------- 최종 변형
def final(L, members=BASE4):
    """클리핑 + 레짐별 NNLS: nested CV 와 평가(테스트·검증) 채점. harness 결과 형식에 스태킹 값을 얹어 돌려준다."""
    cols = [L["names"].index(m) for m in members]
    P, Pe, M, Me = L["oof"][:, cols], L["pred"][:, cols], L["meta_tr"], L["meta_ev"]
    R, Re, y, ye, fid = L["R_tr"], L["R_ev"], L["y_tr"], L["y_ev"], L["fold_id"]
    make = lambda: NNLSStack(per_regime=True)  # noqa: E731
    cv_stack = mse(nested(make, P, M, R, y, fid), y)
    cv_glob_clip = mse(nested(lambda: NNLSStack(), P, M, R, y, fid), y)
    cv_glob_raw = mse(nested(lambda: NNLSStack(), P, M, R, y, fid, clip=False), y)
    st = make().fit(clip_members(P, R, y, R), M, R, y)
    p_ev = st.predict(clip_members(Pe, Re, y, R), Me, Re)
    res = json.loads(json.dumps(L["res"]))
    res["label"] = "stacking_regime_nnls_clip"
    res["cv"]["nnls_insample"] = res["cv"]["ensemble"]
    res["cv"].update(nnls_nested=cv_glob_raw, nnls_nested_clip=cv_glob_clip, ensemble=cv_stack)
    res["weights_nnls"] = res["weights"]
    res["weights"] = {r: dict(zip(members, map(float, w))) for r, w in st.w.items()}
    for split in ["test", "val"]:
        m = L["split_ev"] == split
        res[split]["nnls_ensemble"] = res[split]["ensemble"]
        res[split]["ensemble"] = {"mse": mse(p_ev[m], ye[m]),
                                  **{f"mse_{r}": mse(p_ev[m & (Re == r)], ye[m & (Re == r)]) for r in np.unique(Re)}}
    res["stack"] = {"members": list(members), "clip": "regime train-target range", "stacker": "per-regime NNLS (sum 1)",
                    "cv_nested": cv_stack, "cv_insample": mse(st.predict(clip_members(P, R, y, R), M, R), y)}
    return res


# ----------------------------------------------------------------------------- 탐색
def explore(L):
    names, P, M, R, y, fid, t = L["names"], L["oof"], L["meta_tr"], L["R_tr"], L["y_tr"], L["fold_id"], L["t_tr"]
    mc = L["meta_cols"]
    out = {}

    def score(label, p):
        out[label] = mse(p, y)
        print(f"  {label:44s} {mse(p, y):.3f}  " + " ".join(f"{r} {mse(p[R == r], y[R == r]):.2f}" for r in np.unique(R)), flush=True)

    print("members", names, {k: round(v, 3) for k, v in L["res"]["cv"].items()})
    for sub, cols in [("base4", [names.index(n) for n in BASE4]), ("all", list(range(len(names))))]:
        Ps = P[:, cols]
        w, _ = nnls(Ps, y)
        score(f"[{sub}] nnls in-sample (harness)", Ps @ (w / w.sum()))
        for clip in (False, True):
            c = "clip" if clip else "raw"
            score(f"[{sub}/{c}] nnls global", nested(lambda: NNLSStack(), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] nnls global +intercept", nested(lambda: NNLSStack(intercept=True), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] nnls per-regime", nested(lambda: NNLSStack(per_regime=True), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] nnls per-regime +intercept", nested(lambda: NNLSStack(per_regime=True, intercept=True), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] ridge P+regime", nested(lambda: RidgeStack(), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] ridge P+regime+meta", nested(lambda: RidgeStack(meta=True), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] ridge P+regime+interact+meta", nested(lambda: RidgeStack(interact=True, meta=True), Ps, M, R, y, fid, clip))
            score(f"[{sub}/{c}] lgb direct meta", nested(lambda: LGBStack(residual=False, meta=True, n_estimators=600), Ps, M, R, y, fid, clip))
            for shrink in (0.5, 1.0):
                score(f"[{sub}/{c}] lgb resid(per-regime nnls) meta x{shrink}",
                      nested(lambda: LGBStack(residual=True, meta=True, shrink=shrink), Ps, M, R, y, fid, clip))
    Ps = P[:, [names.index(n) for n in BASE4]]
    p_reg = nested(lambda: NNLSStack(per_regime=True), Ps, M, R, y, fid, True)
    for extra in ["NB_time_3", "NB_trend_15"]:  # (e) 국소 참조 평균: 멤버로 추가 / λ 수축
        Pe = np.column_stack([Ps, M[:, mc.index(extra)]])
        score(f"[base4/clip] per-regime + member {extra}", nested(lambda: NNLSStack(per_regime=True), Pe, M, R, y, fid, True))
        ps, picks = nested_post(p_reg, {0: M[:, mc.index(extra)]}, y, fid)
        score(f"[base4/clip] per-regime, shrink to {extra} λ={[l for _, l in picks]}", ps)
    q = {h: kernel_avg(p_reg, t, R, fid, h) for h in H_GRID}  # (d) 같은 폴드 이웃만 (정직)
    ps, picks = nested_post(p_reg, q, y, fid)
    score(f"[base4/clip] per-regime, time smoothing {picks}", ps)
    e = y - p_reg
    print("  same-fold residual autocorrelation:", {h: round(float(np.corrcoef(e, kernel_avg(e, t, R, fid, h))[0, 1]), 3) for h in (0.5, 2.0, 8.0)})
    return out


def main():
    p = argparse.ArgumentParser(description="stacking 실험")
    p.add_argument("--all", action="store_true", help="탐색 변형 (base4 + et_all + knn OOF 위 스태커 비교)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    if a.all:
        L = level1(a.seed, ["et_all", "knn"], "base6")
        out = explore(L)
        Path(HERE / f"stacking_explore_s{a.seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"[explore] {time.time() - t0:.0f}s")
        return
    L = level1(a.seed, [], "base4")
    res = final(L)
    res["seconds"] = round(time.time() - t0, 1)
    out = HERE / ("stacking.json" if a.seed == 0 else f"stacking_s{a.seed}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[stack] nested cv: global nnls raw {res['cv']['nnls_nested']:.3f}, global nnls clip {res['cv']['nnls_nested_clip']:.3f}, "
          f"per-regime nnls clip {res['cv']['ensemble']:.3f} (harness in-sample nnls {res['cv']['nnls_insample']:.3f})")
    print("[stack] per-regime weights:", {r: {k: round(v, 2) for k, v in w.items()} for r, w in res["weights"].items()})
    for split in ["test", "val"]:
        print(f"[{split}] stacked {res[split]['ensemble']['mse']:.3f} (harness nnls {res[split]['nnls_ensemble']['mse']:.3f})  "
              + " ".join(f"{k} {v:.2f}" for k, v in res[split]["ensemble"].items() if k != "mse"))
    print(summary(res))


if __name__ == "__main__":
    main()
