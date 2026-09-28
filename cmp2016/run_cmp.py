"""PHM 2016 CMP 연마량 예측: 학습, 공식 테스트·검증 정답으로 평가, 물리 해석 결과 저장.

모델
- 챔버·스테이지 평균: 가장 단순한 기준
- 물리 모델: 소모품(드레서·패드·멤브레인) 상태에 따라 Preston 계수가 변한다고 보고,
  챔버 그룹·스테이지마다 릿지 회귀로 맞춘 해석용 모델
- 이웃 평균: 시간상 가까운 학습 웨이퍼 3장의 연마량 평균
- 앙상블(최종): 레짐(챔버 그룹 + 스테이지)마다 LightGBM 2종, ExtraTrees, 릿지를 학습하고,
  학습 데이터 5-fold 교차검증 예측(OOF)으로 정한 비음수 가중치로 섞는다

사용 예:
    python cmp2016/run_cmp.py --out cmp2016/results.json
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_data import TARGET, build_table, neighbor_features, regime, sequence_features  # noqa: E402

EXCLUDE = {"split", "WAFER_ID", "STAGE", TARGET, "OUTLIER", "T_START"}
COMPACT_PREFIX = ("NB_", "same_", "cross_", "regime_mu", "OTHER_", "USAGE_", "ACTIVE_SEC", "GROUP", "STAGE_B")
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


def physics_design(x):
    """드레서·패드·멤브레인 사용량(백 단위)과 평균 PV, 가공 시간으로 만든 설계 행렬."""
    dr = x["USAGE_OF_DRESSER_start"] / 100
    pad = x["USAGE_OF_POLISHING_TABLE_start"] / 100
    mem = x["USAGE_OF_MEMBRANE_start"] / 100
    pv = x["PV_PRESSURIZED__HEAD__sum"] / x["ACTIVE_SEC"].clip(lower=1) / 1e4
    return pd.DataFrame({
        "dresser": dr, "dresser^2": dr ** 2, "pad": pad, "pad^2": pad ** 2,
        "membrane": mem, "dresser*pad": dr * pad, "PV": pv, "active_sec": x["ACTIVE_SEC"] / 100,
    }).fillna(0)


def physics_model(table, train_mask):
    pred = pd.Series(np.nan, index=table.index)
    coefs = {}
    for (grp, stage), x in table.groupby(["GROUP", "STAGE"]):
        tr = x[train_mask[x.index]]
        m = RidgeCV(alphas=np.logspace(-3, 2, 12)).fit(physics_design(tr), tr[TARGET])
        pred[x.index] = m.predict(physics_design(x))
        coefs[f"{grp}{stage}"] = {"intercept": float(m.intercept_),
                                  **dict(zip(physics_design(tr).columns, map(float, m.coef_)))}
    return pred, coefs


def features(table, ref_mask):
    """참조(학습) 집합 기준으로 이웃·순서 특징을 붙인 전체 특징 표."""
    base = table[[c for c in table.columns if c not in EXCLUDE]]
    return base.join(neighbor_features(table, ref_mask)).join(sequence_features(table, ref_mask))


def fit_predict(name, X, y, reg, train_idx, pred_idx):
    """레짐마다 따로 학습해 예측한다. 레짐 안에서 전부 비어 있는 컬럼은 뺀다."""
    make, cols = MODELS[name]
    cols = list(X.columns) if cols == "all" else [c for c in X.columns if c.startswith(COMPACT_PREFIX)]
    pred = np.zeros(len(pred_idx))
    for r in np.unique(reg):
        a = train_idx[reg[train_idx] == r]
        b = reg[pred_idx] == r
        use = [c for c in cols if X.iloc[a][c].notna().any()]
        Xa, Xb = X.iloc[a][use], X.iloc[pred_idx[b]][use]
        if name == "extra_trees":
            Xa, Xb = Xa.fillna(-999), Xb.fillna(-999)
        pred[b] = make().fit(Xa, y[a]).predict(Xb)
    return pred


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def wear_curves(table, train_mask, bins=8):
    """드레서·패드 사용량 구간별 평균 연마량 (챔버 그룹·스테이지별)."""
    d = table[train_mask]
    out = {}
    for (grp, stage), x in d.groupby(["GROUP", "STAGE"]):
        key = f"{grp}{stage}"
        out[key] = {}
        for col, name in [("USAGE_OF_DRESSER_start", "dresser"), ("USAGE_OF_POLISHING_TABLE_start", "pad")]:
            b = pd.qcut(x[col], bins, duplicates="drop")
            g = x.groupby(b, observed=True).agg(u=(col, "median"), y=(TARGET, "mean"),
                                                sd=(TARGET, "std"), n=(TARGET, "size"))
            out[key][name] = g.round(3).to_dict("records")
    return out


def main():
    p = argparse.ArgumentParser(description="PHM 2016 CMP 연마량 예측")
    p.add_argument("--out", default="cmp2016/results.json")
    args = p.parse_args()

    table = build_table()
    y = table[TARGET].to_numpy()
    train_mask = ((table["split"] == "train") & ~table["OUTLIER"]).to_numpy()
    tr = np.flatnonzero(train_mask)
    print(f"[data] train {len(tr)} (outliers removed {int(table['OUTLIER'].sum())}), "
          f"test {int((table.split == 'test').sum())}, val {int((table.split == 'val').sum())}")

    phys_pred, phys_coef = physics_model(table, train_mask)
    regime_mean = table.loc[train_mask].groupby(["GROUP", "STAGE"])[TARGET].mean()
    base_pred = np.asarray(table.set_index(["GROUP", "STAGE"]).index.map(regime_mean), dtype=float)
    nb_all = neighbor_features(table, train_mask)

    # 학습 데이터 5-fold 교차검증: 이웃·순서 특징도 학습 폴드 안에서만 찾는다.
    reg = regime(table)
    ev = np.flatnonzero(table["split"].isin(["test", "val"]).to_numpy())
    folds = list(KFold(5, shuffle=True, random_state=0).split(tr))
    fold_X = []
    for a, _ in folds:
        ref = np.zeros(len(table), bool)
        ref[tr[a]] = True
        fold_X.append(features(table, ref))
    X_full = features(table, train_mask)
    oof, ev_pred = {}, {}
    for name in MODELS:
        o = np.full(len(table), np.nan)
        for (a, b), X in zip(folds, fold_X):
            o[tr[b]] = fit_predict(name, X, y, reg, tr[a], tr[b])
        oof[name] = o[tr]
        ev_pred[name] = fit_predict(name, X_full, y, reg, tr, ev)
        print(f"[cv] {name}: {mse(oof[name], y[tr]):.3f}", flush=True)
    A = np.column_stack(list(oof.values()))
    weights, _ = nnls(A, y[tr])
    weights = weights / weights.sum()
    blend_pred = np.full(len(table), np.nan)
    blend_pred[ev] = np.column_stack(list(ev_pred.values())) @ weights
    lgb_pred = np.full(len(table), np.nan)
    lgb_pred[ev] = ev_pred["lgb"]
    cv_blend = mse(A @ weights, y[tr])
    print(f"[cv] ensemble: {cv_blend:.3f}  weights " + ", ".join(f"{k} {w:.2f}" for k, w in zip(oof, weights)))

    results = {"cv_mse": {**{k: mse(v, y[tr]) for k, v in oof.items()}, "ensemble": cv_blend},
               "weights": dict(zip(oof, map(float, weights))), "splits": {}}
    models = {
        "regime_mean": base_pred,
        "physics": phys_pred.to_numpy(),
        "neighbors_time3": nb_all["NB_time_3"].to_numpy(),
        "lightgbm": lgb_pred,
        "ensemble": blend_pred,
    }
    for split in ["test", "val"]:
        m = (table["split"] == split).to_numpy()
        row = {}
        for name, pred in models.items():
            row[name] = {"mse": mse(pred[m], y[m])}
            for r in np.unique(reg):
                g = m & (reg == r)
                row[name][f"mse_{r}"] = mse(pred[g], y[g])
        results["splits"][split] = row
        print(f"[{split}] " + ", ".join(f"{k} {v['mse']:.3f}" for k, v in row.items()))

    # 중요 특징: 전체 데이터로 학습한 LightGBM(레짐 공통)의 gain
    gm = MODELS["lgb"][0]().fit(X_full.iloc[tr], y[tr])
    gain = pd.Series(gm.booster_.feature_importance("gain"), index=X_full.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    results["importance"] = [[k, float(v)] for k, v in gain.head(15).items()]
    results["physics_coef"] = phys_coef
    results["wear_curves"] = wear_curves(table, train_mask)
    m = table["split"].isin(["test", "val"]).to_numpy()
    results["predictions"] = pd.DataFrame({
        "split": table.loc[m, "split"], "group": table.loc[m, "GROUP"], "stage": table.loc[m, "STAGE"],
        "t": ((table.loc[m, "T_START"] - table["T_START"].min()) / 86400).round(3),
        "y": y[m].round(3), "pred": blend_pred[m].round(3), "physics": phys_pred[m].round(3),
    }).to_dict("records")
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
