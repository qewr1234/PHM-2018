"""멤버 OOF 결합: 레짐별 학습 정답 범위 클리핑 + 레짐별 NNLS(비음수, 합 1), nested 교차검증 채점.

전체 OOF 로 맞춘 전역 NNLS 를 같은 행에 채점한 "in-sample" 값은 낙관적이다 (기준 4 멤버: in-sample 7.596 인데 nested
8.109 — 릿지가 4B 한 행에 -84.7 을 예측해 폴드마다 가중치가 뒤집혔다). 고침: 멤버 예측을 레짐별 학습 정답 [min, max] 로
자르고 레짐별로 NNLS 를 맞춘다. 헤드라인은 nested 값: 멤버 OOF 와 같은 폴드에서 학습 폴드로 클리핑 범위·가중치를 정하고
held-out 폴드를 예측해 채점한다. 최종 가중치는 학습 행 전체(클리핑 후)로 맞춘다. numpy/scipy/sklearn 만 쓴다.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def fold_ids(folds, n):
    """folds ((a, b) 목록 | fold_id 배열 | int fold_seed) → 길이 n 의 fold_id 배열 (KFold(5, shuffle) 기준)."""
    if isinstance(folds, (int, np.integer)):
        folds = list(KFold(5, shuffle=True, random_state=int(folds)).split(np.arange(n)))
    if len(folds) and not isinstance(folds[0], (tuple, list)):
        fid = np.asarray(folds, dtype=int)
        assert len(fid) == n, f"fold_id 길이 {len(fid)} != {n}"
        return fid
    fid = np.full(n, -1, int)
    for k, (_, b) in enumerate(folds):
        fid[np.asarray(b)] = k
    assert (fid >= 0).all(), "폴드가 학습 행 전부를 덮지 않습니다"
    return fid


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


def nested_blend(oof, y, reg, folds, per_regime=True, clip=True, names=None):
    """OOF 행 위 nested 결합 예측: 폴드마다 학습 폴드로 클리핑 범위·NNLS 가중치를 정하고 held-out 폴드를 예측한다."""
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
    """(weights {레짐: {멤버: w}}, nested_cv, in_sample_cv). 최종 가중치는 학습 행 전체(클리핑 후)로 맞춘다.

    oof: {멤버: OOF 배열(학습 행)}, folds: (a, b) 목록 | fold_id 배열 | int(fold_seed).
    per_regime=False, clip=False 면 기준선(run_cmp 이전 버전)과 같은 전역 NNLS 정의다.
    """
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
