"""blend: 멤버 OOF 결합 (stacking.py 최종 방식) — 레짐별 학습 정답 범위 클리핑 + 레짐별 NNLS(비음수, 합 1),
harness 폴드 위 nested 채점.

- harness.run 의 "cv.ensemble" 은 전체 OOF 로 맞춘 전역 NNLS 를 같은 행에 채점한 in-sample 값이라 낙관적이다
  (base4 seed 0: in-sample 7.596, nested 8.109; ridge 가 4B 한 행에 -84.7 을 예측해 폴드마다 가중치가 뒤집힌다).
- 고침: 멤버 예측을 레짐별 학습 정답 [min, max] 로 자르고(clip_bounds/clip_to) 레짐별 NNLS 를 맞춘다.
  nested 채점은 멤버 OOF 와 같은 폴드(KFold(5, shuffle, random_state=fold_seed), 학습 행 순서)에서
  학습 폴드로 클리핑 범위·가중치를 정하고 held-out 폴드를 예측한다.
- 이 모듈은 numpy/scipy/sklearn 만 쓴다 (harness 는 단위 테스트에서만 import).

API
    harness_folds(n, fold_seed)                          harness.run 과 같은 (a, b) 폴드 목록
    clip_bounds(y, reg) -> {레짐: (lo, hi)}              학습 정답 범위
    clip_and_weights(oof, y, reg, folds) -> (weights, nested_cv, in_sample_cv)
        oof: {멤버: OOF 배열(학습 행)}, folds: (a, b) 목록 | fold_id 배열 | int(fold_seed).
        weights: {레짐: {멤버: w}} (per_regime=False 면 {"all": {...}}), clip=False 면 클리핑 없음.
    apply(pred, reg_eval, weights, bounds) -> 결합 예측   pred: {멤버: 평가 배열}, bounds: clip_bounds(y_train, reg_train)
    nested_blend(oof, y, reg, folds) -> nested OOF 결합 예측 (레짐별 채점용)
    greedy_forward(oof, y, reg, folds, start=["lgb"], min_gain=0.05) -> [(추가 멤버, nested cv), ...]

검증: python cmp2016/exp/blend.py   (cache_stacking_base4_s{0,1}.npz 로 stacking.py 의 수치 재현:
      seed 0 nested 전역 raw 8.109 → 클리핑 전역 7.592 → 클리핑 레짐별 7.543 (in-sample 7.490), 테스트 6.856, 검증 6.789;
      seed 1 nested 7.399, 테스트 6.886, 검증 6.798)
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold

N_FOLDS = 5


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


# ----------------------------------------------------------------------------- 폴드
def harness_folds(n, fold_seed=0, n_folds=N_FOLDS):
    """harness.run 과 같은 폴드: KFold(n_folds, shuffle, random_state=fold_seed).split(학습 행 0..n-1)."""
    return list(KFold(n_folds, shuffle=True, random_state=fold_seed).split(np.arange(n)))


def fold_ids(folds, n):
    """folds ((a, b) 목록 | fold_id 배열 | int fold_seed) → 길이 n 의 fold_id 배열."""
    if isinstance(folds, (int, np.integer)):
        folds = harness_folds(n, int(folds))
    if len(folds) and not isinstance(folds[0], (tuple, list)):
        fid = np.asarray(folds, dtype=int)
        assert len(fid) == n, f"fold_id 길이 {len(fid)} != {n}"
        return fid
    fid = np.full(n, -1, int)
    for k, (_, b) in enumerate(folds):
        fid[np.asarray(b)] = k
    assert (fid >= 0).all(), "폴드가 학습 행 전부를 덮지 않습니다"
    return fid


# ----------------------------------------------------------------------------- 클리핑 · NNLS
def clip_bounds(y, reg):
    """레짐별 (학습) 정답 [min, max]."""
    y, reg = np.asarray(y, float), np.asarray(reg)
    return {r: (float(y[reg == r].min()), float(y[reg == r].max())) for r in np.unique(reg)}


def clip_to(P, reg, bounds):
    """멤버 예측 행렬 P[n, m] 을 행의 레짐 범위로 자른다 (bounds 에 없는 레짐은 그대로)."""
    Q = np.array(P, dtype=float, copy=True)
    reg = np.asarray(reg)
    for r, (lo, hi) in bounds.items():
        m = reg == r
        Q[m] = np.clip(Q[m], lo, hi)
    return Q


def _matrix(pred, names):
    P = np.column_stack([np.asarray(pred[n], float) for n in names])
    assert np.isfinite(P).all(), "멤버 예측에 NaN/inf 가 있습니다"
    return P


def fit_weights(P, y, reg, per_regime=True):
    """NNLS (비음수, 합 1). 반환 {레짐: w 배열} (per_regime=False 면 {"all": w}). 전부 0 이면 균등."""
    y, reg = np.asarray(y, float), np.asarray(reg)
    groups = {r: reg == r for r in np.unique(reg)} if per_regime else {"all": np.ones(len(y), bool)}
    out = {}
    for r, m in groups.items():
        w, _ = nnls(P[m], y[m])
        out[r] = w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1.0 / P.shape[1])
    return out


def predict_weights(P, reg, W):
    if "all" in W:
        return P @ W["all"]
    reg = np.asarray(reg)
    missing = set(np.unique(reg)) - set(W)
    assert not missing, f"가중치가 없는 레짐: {missing}"
    out = np.zeros(len(P))
    for r, w in W.items():
        m = reg == r
        out[m] = P[m] @ w
    return out


# ----------------------------------------------------------------------------- API
def nested_blend(oof, y, reg, folds, per_regime=True, clip=True, names=None):
    """OOF 행 위 nested 결합: 폴드마다 학습 폴드로 클리핑 범위·NNLS 가중치를 정하고 held-out 폴드를 예측한다."""
    names = list(oof) if names is None else list(names)
    y, reg = np.asarray(y, float), np.asarray(reg)
    P, fid = _matrix(oof, names), fold_ids(folds, len(y))
    out = np.full(len(y), np.nan)
    for k in np.unique(fid):
        a, b = fid != k, fid == k
        Pa, Pb = P[a], P[b]
        if clip:
            bounds = clip_bounds(y[a], reg[a])
            Pa, Pb = clip_to(Pa, reg[a], bounds), clip_to(Pb, reg[b], bounds)
        out[b] = predict_weights(Pb, reg[b], fit_weights(Pa, y[a], reg[a], per_regime))
    return out


def clip_and_weights(oof, y, reg, folds, per_regime=True, clip=True):
    """(weights {레짐: {멤버: w}}, nested_cv, in_sample_cv). 최종 가중치는 학습 행 전체(클리핑 후)로 맞춘다."""
    names = list(oof)
    y, reg = np.asarray(y, float), np.asarray(reg)
    nested_cv = mse(nested_blend(oof, y, reg, folds, per_regime, clip, names), y)
    P = _matrix(oof, names)
    if clip:
        P = clip_to(P, reg, clip_bounds(y, reg))
    W = fit_weights(P, y, reg, per_regime)
    in_sample_cv = mse(predict_weights(P, reg, W), y)
    weights = {r: dict(zip(names, map(float, w))) for r, w in W.items()}
    return weights, nested_cv, in_sample_cv


def apply(pred, reg_eval, weights, bounds=None):
    """평가 행 결합: 멤버 예측을 bounds(clip_bounds 결과) 로 자른 뒤 weights[레짐] 으로 섞는다."""
    names = list(next(iter(weights.values())))
    reg_eval = np.asarray(reg_eval)
    P = _matrix(pred, names)
    if bounds is not None:
        P = clip_to(P, reg_eval, bounds)
    W = {r: np.array([w[n] for n in names]) for r, w in weights.items()}
    return predict_weights(P, reg_eval, W)


def per_regime_mse(p, y, reg):
    """{"all": mse, 레짐: mse, ...}."""
    y, reg = np.asarray(y, float), np.asarray(reg)
    return {"all": mse(p, y), **{r: mse(p[reg == r], y[reg == r]) for r in np.unique(reg)}}


def greedy_forward(oof_by_member, y, reg, folds, start=("lgb",), min_gain=0.05, per_regime=True, clip=True, verbose=True):
    """탐욕 전진 선택 (nested CV 만 사용): start 에서 시작해 nested CV 를 가장 줄이는 멤버를 하나씩 더하고,
    개선이 min_gain 미만이면 멈춘다. 반환 [(start 를 '+' 로 이은 이름, cv), (추가 멤버, cv), ...]."""
    chosen = list(start)
    y, reg = np.asarray(y, float), np.asarray(reg)
    fid = fold_ids(folds, len(y))

    def score(names):
        return mse(nested_blend(oof_by_member, y, reg, fid, per_regime, clip, names), y)

    cur = score(chosen)
    path = [("+".join(chosen), cur)]
    if verbose:
        print(f"[greedy] start {path[0][0]}: nested {cur:.3f}", flush=True)
    pool = [n for n in oof_by_member if n not in chosen]
    while pool:
        scores = {c: score(chosen + [c]) for c in pool}
        best = min(scores, key=scores.get)
        if verbose:
            print("  " + "  ".join(f"{c} {v:.3f}" for c, v in sorted(scores.items(), key=lambda kv: kv[1])), flush=True)
        if cur - scores[best] < min_gain:
            if verbose:
                print(f"[greedy] stop: best +{best} {scores[best]:.3f} (gain {cur - scores[best]:.3f} < {min_gain})", flush=True)
            break
        chosen.append(best)
        cur = scores[best]
        pool.remove(best)
        path.append((best, cur))
        if verbose:
            print(f"[greedy] + {best}: nested {cur:.3f}", flush=True)
    return path


# ----------------------------------------------------------------------------- 단위 테스트 (stacking.py 수치 재현)
def _load_stacking_cache(seed):
    import json
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from harness import get_table
    from cmp_data import TARGET, regime
    z = np.load(here / f"cache_stacking_base4_s{seed}.npz", allow_pickle=True)
    names = list(z["names"])
    table = get_table()
    y, reg, split = table[TARGET].to_numpy(), regime(table), table["split"].to_numpy()
    tr, ev = z["train_idx"], z["eval_idx"]
    L = dict(names=names, oof={n: z["oof"][:, i] for i, n in enumerate(names)},
             pred={n: z["pred"][:, i] for i, n in enumerate(names)}, fold_id=z["fold_id"],
             y_tr=y[tr], y_ev=y[ev], R_tr=reg[tr], R_ev=reg[ev], split_ev=split[ev])
    f = here / ("stacking.json" if seed == 0 else f"stacking_s{seed}.json")
    L["ref"] = json.loads(f.read_text()) if f.exists() else None
    return L


def _test(seed):
    L = _load_stacking_cache(seed)
    oof, y, R, fid = L["oof"], L["y_tr"], L["R_tr"], L["fold_id"]
    ref = L["ref"]
    assert (fold_ids(harness_folds(len(y), seed), len(y)) == fid).all(), "harness_folds 가 캐시 fold_id 와 다릅니다"
    assert (fold_ids(seed, len(y)) == fid).all()
    _, raw_glob, ins_glob = clip_and_weights(oof, y, R, fid, per_regime=False, clip=False)
    _, clip_glob, _ = clip_and_weights(oof, y, R, fid, per_regime=False, clip=True)
    W, nested, ins = clip_and_weights(oof, y, R, fid)
    bounds = clip_bounds(y, R)
    p_ev = apply(L["pred"], L["R_ev"], W, bounds)
    test = mse(p_ev[L["split_ev"] == "test"], L["y_ev"][L["split_ev"] == "test"])
    val = mse(p_ev[L["split_ev"] == "val"], L["y_ev"][L["split_ev"] == "val"])
    print(f"[blend s{seed}] nested global raw {raw_glob:.3f} (in-sample {ins_glob:.3f}) -> global clip {clip_glob:.3f} "
          f"-> per-regime clip {nested:.3f} (in-sample {ins:.3f});  test {test:.3f}, val {val:.3f}")
    print(f"[blend s{seed}] weights " + " / ".join(f"{r}: " + ", ".join(f"{n} {w:.2f}" for n, w in d.items()) for r, d in W.items()))
    print(f"[blend s{seed}] nested per regime " + ", ".join(f"{k} {v:.3f}" for k, v in per_regime_mse(nested_blend(oof, y, R, fid), y, R).items()))
    if ref is not None:
        exp = {"nested global raw": (raw_glob, ref["cv"]["nnls_nested"]), "in-sample global": (ins_glob, ref["cv"]["nnls_insample"]),
               "nested global clip": (clip_glob, ref["cv"]["nnls_nested_clip"]), "nested per-regime clip": (nested, ref["cv"]["ensemble"]),
               "in-sample per-regime": (ins, ref["stack"]["cv_insample"]), "test": (test, ref["test"]["ensemble"]["mse"]),
               "val": (val, ref["val"]["ensemble"]["mse"])}
        for k, (got, want) in exp.items():
            assert abs(got - want) < 1e-6, f"{k}: {got:.6f} != stacking.json {want:.6f}"
        for r in W:
            for n in W[r]:
                assert abs(W[r][n] - ref["weights"][r][n]) < 1e-6, f"weight {r}/{n}"
        print(f"[blend s{seed}] matches stacking{'' if seed == 0 else f'_s{seed}'}.json to 1e-6 "
              f"(nested {ref['cv']['ensemble']:.3f}, test {ref['test']['ensemble']['mse']:.3f}, val {ref['val']['ensemble']['mse']:.3f})")
    path = greedy_forward(oof, y, R, fid, start=["lgb"], min_gain=0.05, verbose=False)
    print(f"[blend s{seed}] greedy from lgb: " + " -> ".join(f"{n} {v:.3f}" for n, v in path))
    # apply 와 nested 의 일관성: 폴드 하나를 손으로 계산
    a, b = fid != 0, fid == 0
    Wk, _, _ = clip_and_weights({n: v[a] for n, v in oof.items()}, y[a], R[a], fold_ids(seed, len(y))[a] % 4)
    p_b = apply({n: v[b] for n, v in oof.items()}, R[b], Wk, clip_bounds(y[a], R[a]))
    assert np.allclose(p_b, nested_blend(oof, y, R, fid)[b]), "apply 와 nested_blend 가 다릅니다"


if __name__ == "__main__":
    for s in (0, 1):
        _test(s)
    print("[blend] all tests passed")
