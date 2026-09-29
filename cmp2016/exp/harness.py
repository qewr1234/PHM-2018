"""실험 공통 프로토콜 (run_cmp.py와 같은 채점 방식).

- 학습 데이터(측정 오류 제외) 5-fold 교차검증. 이웃·순서 특징 등 정답에서 만든 특징은 학습 폴드 안에서만 찾는다.
- 모델별 OOF 예측을 비음수 최소제곱(NNLS)으로 섞어 앙상블 가중치를 정한다 (CV로만 결정).
- 학습 전체로 다시 학습해 공식 테스트·검증 정답으로 채점한다 (보고용, 선택에는 쓰지 않는다).

사용 예:
    from harness import run, get_table, get_raw, BASE_MODELS
    res = run(static_fn=my_features)                      # 특징 추가
    res = run(models={**BASE_MODELS, "xgb": {...}})       # 모델 추가
    print(res["cv"]["ensemble"], res["test"]["ensemble"]["mse"], res["val"]["ensemble"]["mse"])

기준선(run_cmp.py 결과): CV 앙상블 7.596, 테스트 6.969, 검증 6.850.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lightgbm as lgb  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from cmp_data import KEY, PROCESS_COLS, RAW, TARGET, build_table, load_labels, load_split, regime  # noqa: E402
from run_cmp import COMPACT_PREFIX, EXCLUDE, mse  # noqa: E402
from run_cmp import features as _features  # noqa: E402

# 기준선(탐색 시작 시점의 run_cmp.py) 4 멤버. run_cmp.py 는 최종 멤버(cmp_models)로 바뀌었으므로 여기서 그대로 보존한다.
MODELS = {
    "lgb": (lambda: lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                      random_state=0, verbose=-1), "all"),
    "lgb_small": (lambda: lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=7, min_child_samples=20,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                            random_state=1, verbose=-1), "compact"),
    "extra_trees": (lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                n_jobs=-1, random_state=0), "compact"),
    "ridge": (lambda: make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
              "compact"),
}


def features(table, ref_mask):
    """기준 특징 112 컬럼 (기본 + 이웃 + 순서). run_cmp.features 는 이제 NB_ 확장까지 붙이므로 여기서 잘라낸다."""
    X = _features(table, ref_mask)
    return X[[c for c in X.columns if not c.startswith(("NB_E_", "NB_D_", "NB_O_"))]]

CACHE = RAW.parent
N_FOLDS, FOLD_SEED = 5, 0
BASELINE = {"cv": 7.596, "test": 6.969, "val": 6.850}

# 모델 명세: {"make": 팩토리, "cols": "all" | "compact" | 접두사 튜플 | callable(컬럼 목록) -> 목록,
#            "fillna": 결측 대체값(None이면 그대로), "per_regime": 레짐별 학습 여부}
BASE_MODELS = {
    name: {"make": make, "cols": cols, "fillna": -999 if name == "extra_trees" else None, "per_regime": True}
    for name, (make, cols) in MODELS.items()
}


_DEFAULT_TABLE = None


def get_table():
    """build_table() 결과 (data/cmp2016/table.parquet 에 캐시, 약 1분 → 0.1초). 항상 같은 객체를 돌려준다."""
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        f = CACHE / "table.parquet"
        if not f.exists():
            build_table().to_parquet(f)
        _DEFAULT_TABLE = pd.read_parquet(f)
    return _DEFAULT_TABLE


def get_raw():
    """세 split의 센서 행 전체 (split, GROUP, STEP, ACTIVE 컬럼 추가, KEY+TIMESTAMP 정렬, parquet 캐시)."""
    f = CACHE / "raw_all.parquet"
    if f.exists():
        return pd.read_parquet(f)
    frames = []
    for split in ["train", "test", "val"]:
        raw = load_split(split).sort_values(KEY + ["TIMESTAMP"])
        group = raw.groupby(KEY)["CHAMBER"].transform("min")
        raw = raw.assign(split=split, GROUP=np.where(group <= 3, 1, 4))
        raw = raw.assign(STEP=(raw["CHAMBER"] - raw["GROUP"]).clip(0, 2).astype(int))
        raw = raw.assign(ACTIVE=(raw["PRESSURIZED_CHAMBER_PRESSURE"] > 0) & (raw["SLURRY_FLOW_LINE_C"] > 0))
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(f)
    return raw


def sequences(raw, n_points=64, cols=PROCESS_COLS, active_only=True):
    """웨이퍼·스테이지마다 챔버 단계(STEP 0,1,2)별 센서 구간을 n_points 길이로 선형 보간해 고정 길이 배열로 만든다.

    반환: keys(DataFrame WAFER_ID, STAGE), X(float32, [n, 3, n_points, len(cols)]), 없는 단계는 NaN.
    """
    r = raw[raw["ACTIVE"]] if active_only else raw
    keys = raw[KEY].drop_duplicates().reset_index(drop=True)
    pos = {tuple(k): i for i, k in enumerate(keys.itertuples(index=False))}
    X = np.full((len(keys), 3, n_points, len(cols)), np.nan, dtype=np.float32)
    grid = np.linspace(0, 1, n_points)
    for (w, s, step), g in r.groupby(KEY + ["STEP"]):
        v = g[cols].to_numpy(dtype=float)
        if len(v) < 2:
            continue
        src = np.linspace(0, 1, len(v))
        X[pos[(w, s)], step] = np.column_stack([np.interp(grid, src, v[:, j]) for j in range(v.shape[1])])
    return keys, X


def _select(spec, columns, X_train):
    cols = spec.get("cols", "all")
    if cols == "all":
        use = list(columns)
    elif cols == "compact":
        use = [c for c in columns if c.startswith(COMPACT_PREFIX)]
    elif callable(cols):
        use = list(cols(list(columns)))
    else:
        use = [c for c in columns if c.startswith(tuple(cols))]
    return [c for c in use if X_train[c].notna().any()]


def fit_predict(spec, X, y, reg, train_idx, pred_idx):
    pred = np.zeros(len(pred_idx))
    groups = [(train_idx, np.ones(len(pred_idx), bool))]
    if spec.get("per_regime", True):
        groups = [(train_idx[reg[train_idx] == r], reg[pred_idx] == r) for r in np.unique(reg[pred_idx])]
    for a, b in groups:
        if not b.any():
            continue
        use = _select(spec, X.columns, X.iloc[a])
        Xa, Xb = X.iloc[a][use], X.iloc[pred_idx[b]][use]
        if spec.get("fillna") is not None:
            Xa, Xb = Xa.fillna(spec["fillna"]), Xb.fillna(spec["fillna"])
        pred[b] = spec["make"]().fit(Xa, y[a]).predict(Xb)
    return pred


def base_features(table, ref_mask, cache_key=None):
    """run_cmp.features(): 기본 + 이웃 + 순서 특징. 폴드별 결과를 parquet에 캐시한다 (표는 고정)."""
    f = None if cache_key is None else CACHE / "feat_cache" / f"{cache_key}.parquet"
    if f is not None and f.exists():
        return pd.read_parquet(f)
    X = features(table, ref_mask)
    if f is not None:
        f.parent.mkdir(exist_ok=True)
        X.to_parquet(f)
    return X


def build_features(table, ref_mask, static=None, feature_fn=None, cache_key=None):
    X = base_features(table, ref_mask, cache_key)
    if static is not None:
        X = X.join(static)
    if feature_fn is not None:
        X = X.join(feature_fn(table, ref_mask))
    return X


def run(static_fn=None, feature_fn=None, models=None, table=None, fold_seed=FOLD_SEED, n_folds=N_FOLDS,
        target_fn=None, label="", out=None, verbose=True, keep_pred=False):
    """실험 하나를 기준 프로토콜로 채점한다.

    static_fn(table) -> DataFrame: 정답을 쓰지 않는 웨이퍼별 특징 (한 번만 계산).
    feature_fn(table, ref_mask) -> DataFrame: 참조(학습 폴드) 정답에서 만드는 특징 (폴드마다 계산).
    models: {이름: 명세} (기본 BASE_MODELS). 명세는 BASE_MODELS 항목 참고. make()는 fit(X, y)/predict(X)를 가진 객체.
    target_fn(X, table) -> (offset 배열): 모델은 y - offset을 학습하고 예측에 offset을 더한다 (잔차 학습). None이면 0.
    반환: {"cv": {모델: mse, "ensemble": mse}, "weights": {...}, "test": {...}, "val": {...}, "n_features": int, ...}
    """
    t0 = time.time()
    models = BASE_MODELS if models is None else models
    table = get_table() if table is None else table
    y = table[TARGET].to_numpy()
    reg = regime(table)
    train_mask = ((table["split"] == "train") & ~table["OUTLIER"]).to_numpy()
    tr = np.flatnonzero(train_mask)
    ev = np.flatnonzero(table["split"].isin(["test", "val"]).to_numpy())
    static = None if static_fn is None else static_fn(table)
    if static is not None:
        bad = [c for c in static.columns if c in table.columns]
        assert not bad, f"static_fn 컬럼이 기존 컬럼과 겹칩니다: {bad}"

    folds = list(KFold(n_folds, shuffle=True, random_state=fold_seed).split(tr))
    fold_X = []
    for a, _ in folds:
        ref = np.zeros(len(table), bool)
        ref[tr[a]] = True
        key = None if table is not _DEFAULT_TABLE else f"s{fold_seed}_k{n_folds}_f{len(fold_X)}"
        fold_X.append(build_features(table, ref, static, feature_fn, key))
    X_full = build_features(table, train_mask, static, feature_fn, None if table is not _DEFAULT_TABLE else "full")

    oof, ev_pred = {}, {}
    for name, spec in models.items():
        o = np.full(len(table), np.nan)
        for (a, b), X in zip(folds, fold_X):
            off = np.zeros(len(table)) if target_fn is None else np.asarray(target_fn(X, table), dtype=float)
            o[tr[b]] = fit_predict(spec, X, y - off, reg, tr[a], tr[b]) + off[tr[b]]
        oof[name] = o[tr]
        off = np.zeros(len(table)) if target_fn is None else np.asarray(target_fn(X_full, table), dtype=float)
        ev_pred[name] = fit_predict(spec, X_full, y - off, reg, tr, ev) + off[ev]
        if verbose:
            print(f"[cv] {label} {name}: {mse(oof[name], y[tr]):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    A = np.column_stack(list(oof.values()))
    w, _ = nnls(A, y[tr])
    w = w / w.sum()
    cv_blend = mse(A @ w, y[tr])
    blend = np.full(len(table), np.nan)
    blend[ev] = np.column_stack(list(ev_pred.values())) @ w
    preds = {name: np.full(len(table), np.nan) for name in models}
    for name in models:
        preds[name][ev] = ev_pred[name]
    preds["ensemble"] = blend

    res = {"label": label, "cv": {**{k: mse(v, y[tr]) for k, v in oof.items()}, "ensemble": cv_blend},
           "weights": dict(zip(oof, map(float, w))), "n_features": int(X_full.shape[1]),
           "n_folds": n_folds, "fold_seed": fold_seed, "seconds": round(time.time() - t0, 1)}
    for split in ["test", "val"]:
        m = (table["split"] == split).to_numpy()
        res[split] = {}
        for name, p in preds.items():
            res[split][name] = {"mse": mse(p[m], y[m]), **{f"mse_{r}": mse(p[m & (reg == r)], y[m & (reg == r)])
                                                         for r in np.unique(reg)}}
    if keep_pred:
        res["oof"] = {k: v.tolist() for k, v in oof.items()}
        res["pred"] = {k: v[ev].tolist() for k, v in preds.items()}
        res["train_idx"], res["eval_idx"] = tr.tolist(), ev.tolist()
    if verbose:
        print(f"[cv] {label} ensemble {cv_blend:.3f}  weights " + ", ".join(f"{k} {v:.2f}" for k, v in res["weights"].items()))
        print(f"[test] {label} ensemble {res['test']['ensemble']['mse']:.3f}   [val] ensemble {res['val']['ensemble']['mse']:.3f}"
              f"   (baseline cv {BASELINE['cv']} / test {BASELINE['test']} / val {BASELINE['val']})")
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def summary(res):
    """결과 한 줄 요약."""
    return (f"{res.get('label', '')}: cv {res['cv']['ensemble']:.3f} (Δ{res['cv']['ensemble'] - BASELINE['cv']:+.3f}), "
            f"test {res['test']['ensemble']['mse']:.3f}, val {res['val']['ensemble']['mse']:.3f}, "
            f"features {res['n_features']}, {res['seconds']}s")


if __name__ == "__main__":
    t = get_table()
    print("[cache] table", t.shape)
    r = get_raw()
    print("[cache] raw", r.shape)
    res = run(label="baseline", out=str(HERE / "baseline.json"))
    print(summary(res))
