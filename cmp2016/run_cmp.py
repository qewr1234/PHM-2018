"""PHM 2016 CMP 연마량 예측: 학습, 공식 테스트·검증 정답으로 평가, 물리 해석 결과 저장.

모델
- 챔버·스테이지 평균: 가장 단순한 기준
- 물리 모델: 소모품(드레서·패드·멤브레인) 상태에 따라 Preston 계수가 변한다고 보고,
  챔버 그룹·스테이지마다 릿지 회귀로 맞춘 해석용 모델
- 이웃 평균: 시간상 가까운 학습 웨이퍼 3장의 연마량 평균
- 앙상블(최종): 멤버 3종 — 레짐(챔버 그룹 + 스테이지)별 LightGBM(gain 상위 150 컬럼), 시퀀스 CNN + 표 특징 융합 신경망,
  레짐별 가우시안 프로세스(시간 + 소모품 상태) — 의 학습 데이터 5-fold 교차검증 예측(OOF)을 레짐별 학습 정답 범위로
  자른 뒤 레짐별 비음수 가중치(NNLS)로 섞는다 (cmp_models.py, cmp_blend.py).

특징 (cmp_data.py): 기본 112 (단계별 공정 요약·소모품·Preston 항·이웃·순서) + TS_ 시퀀스 모양 418 (정답 미사용, 정적)
+ NB_E_/NB_D_/NB_O_ 이웃 확장 50 (참조 정답 사용 → 폴드마다 학습 폴드 안에서만 계산) = 580 컬럼.

채점: 헤드라인은 nested 교차검증 (폴드마다 학습 폴드 OOF 로 클리핑 범위·가중치를 정해 held-out 폴드를 채점).
전체 OOF 로 맞춘 가중치를 같은 행에 채점한 in-sample 값은 낙관적이라 비교용으로만 함께 적는다.
모델·가중치 선택은 교차검증으로만 하고, 테스트·검증 정답은 마지막 보고에만 쓴다.

사용 예:
    python cmp2016/run_cmp.py --out cmp2016/results.json     # 약 6분 (4 CPU; 정적 특징·시퀀스 캐시가 없으면 +2분)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmp_models as cm  # noqa: E402  (lightgbm/torch 보다 먼저: 스레드 환경변수)
import cmp_blend as cb  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from cmp_data import (EXCLUDE, TARGET, build_table, neighbor_ext_features, neighbor_features, regime,  # noqa: E402
                      sequence_features, sequence_tensor, tsshape_features)

COMPACT_PREFIX = cm.COMPACT_PREFIX
N_FOLDS, FOLD_SEED = 5, 0


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


def features(table, ref_mask, static=None):
    """참조(학습) 집합 기준 특징 표: 기본 + 이웃 + 순서 [+ 정적 static] + 이웃 확장(NB_E_/D_/O_). 표의 index 를 유지한다."""
    base = table[[c for c in table.columns if c not in EXCLUDE]]
    X = base.join(neighbor_features(table, ref_mask)).join(sequence_features(table, ref_mask))
    if static is not None:
        X = X.join(static)
    return X.join(neighbor_ext_features(table, ref_mask))


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
    p.add_argument("--seed", type=int, default=FOLD_SEED, help="교차검증 폴드 시드 (보고 수치는 0)")
    args = p.parse_args()
    t0 = time.time()

    table = build_table()
    y = table[TARGET].to_numpy()
    train_mask = ((table["split"] == "train") & ~table["OUTLIER"]).to_numpy()
    tr = np.flatnonzero(train_mask)
    print(f"[data] train {len(tr)} (outliers removed {int(table['OUTLIER'].sum())}), "
          f"test {int((table.split == 'test').sum())}, val {int((table.split == 'val').sum())}  ({time.time() - t0:.0f}s)")

    phys_pred, phys_coef = physics_model(table, train_mask)
    regime_mean = table.loc[train_mask].groupby(["GROUP", "STAGE"])[TARGET].mean()
    base_pred = np.asarray(table.set_index(["GROUP", "STAGE"]).index.map(regime_mean), dtype=float)
    nb_all = neighbor_features(table, train_mask)

    # 정답을 쓰지 않는 입력: 시퀀스 모양 정적 특징(TS_)과 신경망용 시퀀스 텐서 (data/cmp2016/ 에 캐시)
    static = tsshape_features(table)
    seq, present = sequence_tensor(table)
    print(f"[features] static TS_ {static.shape[1]}, sequences {tuple(seq.shape)}  ({time.time() - t0:.0f}s)", flush=True)

    # 학습 데이터 5-fold 교차검증: 참조 정답에서 만드는 특징(이웃·순서·이웃 확장)은 학습 폴드 안에서만 찾는다.
    reg = regime(table)
    ev = np.flatnonzero(table["split"].isin(["test", "val"]).to_numpy())
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=args.seed).split(tr))
    fold_X = []
    for a, _ in folds:
        ref = np.zeros(len(table), bool)
        ref[tr[a]] = True
        fold_X.append(features(table, ref, static))
    X_full = features(table, train_mask, static)
    print(f"[features] {X_full.shape[1]} columns  ({time.time() - t0:.0f}s)", flush=True)

    models = cm.make_models(table, seq, present)
    oof, ev_pred = {}, {}
    for name, spec in models.items():
        o = np.full(len(table), np.nan)
        for (a, b), X in zip(folds, fold_X):
            o[tr[b]] = cm.fit_predict(spec, X, y, reg, tr[a], tr[b])
        oof[name] = o[tr]
        ev_pred[name] = cm.fit_predict(spec, X_full, y, reg, tr, ev)
        print(f"[cv] {name}: {mse(oof[name], y[tr]):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # 결합: 레짐별 학습 정답 범위로 자른 뒤 레짐별 NNLS. nested 값이 헤드라인, in-sample 은 비교용.
    weights, cv_nested, cv_insample = cb.clip_and_weights(oof, y[tr], reg[tr], folds)
    _, _, cv_global = cb.clip_and_weights(oof, y[tr], reg[tr], folds, per_regime=False, clip=False)  # 이전 버전(전역 NNLS) 정의
    nested_by_regime = cb.per_regime_mse(cb.nested_blend(oof, y[tr], reg[tr], folds), y[tr], reg[tr])
    bounds = cb.clip_bounds(y[tr], reg[tr])
    blend_pred = np.full(len(table), np.nan)
    blend_pred[ev] = cb.apply(ev_pred, reg[ev], weights, bounds)
    print(f"[cv] nested ensemble: {cv_nested:.3f}  " + ", ".join(f"{k} {v:.3f}" for k, v in nested_by_regime.items() if k != "all"))
    print(f"[cv] in-sample ensemble: {cv_insample:.3f} (per-regime clipped NNLS), {cv_global:.3f} (global NNLS, 이전 정의; 낙관적)")
    for r, w in weights.items():
        print(f"[cv] weights {r}: " + ", ".join(f"{k} {v:.2f}" for k, v in w.items()))

    results = {"cv_mse": {**{k: mse(v, y[tr]) for k, v in oof.items()}, "ensemble": cv_insample,
                          "ensemble_global_nnls": cv_global},
               "cv_nested": cv_nested, "cv_nested_by_regime": nested_by_regime,
               "weights": weights, "clip_bounds": bounds, "members": list(models), "n_features": int(X_full.shape[1]),
               "fold_seed": args.seed, "splits": {}}
    member_pred = {}
    for name in models:
        member_pred[name] = np.full(len(table), np.nan)
        member_pred[name][ev] = ev_pred[name]
    rows = {
        "regime_mean": base_pred,
        "physics": phys_pred.to_numpy(),
        "neighbors_time3": nb_all["NB_time_3"].to_numpy(),
        "lightgbm": member_pred["lgb"],
        "seqdl": member_pred["seqdl"],
        "gp_state": member_pred["gp_state"],
        "ensemble": blend_pred,
    }
    for split in ["test", "val"]:
        m = (table["split"] == split).to_numpy()
        row = {}
        for name, pred in rows.items():
            row[name] = {"mse": mse(pred[m], y[m])}
            for r in np.unique(reg):
                g = m & (reg == r)
                row[name][f"mse_{r}"] = mse(pred[g], y[g])
        results["splits"][split] = row
        print(f"[{split}] " + ", ".join(f"{k} {v['mse']:.3f}" for k, v in row.items()))

    # 중요 특징: 전체 학습 행으로 학습한 LightGBM(레짐 공통, 전체 컬럼)의 gain
    gm = lgb.LGBMRegressor(**cm.LGB_BASE).fit(X_full.iloc[tr], y[tr])
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
    results["runtime_seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False))
    print(f"[save] {args.out}  ({results['runtime_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
