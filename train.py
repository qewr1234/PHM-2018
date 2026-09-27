"""PHM 2018 베이스라인 학습 스크립트.

고장 모드별로 HistGradientBoostingRegressor를 하나씩 학습해 TTF(초)를 예측한다.
장비(tool) 단위 GroupKFold 교차검증으로 처음 보는 장비에 대한 성능을 확인한 뒤,
전체 데이터로 다시 학습한 모델을 저장한다.

사용 예:
    python train.py --train-dir data/synthetic/train --model-out models/baseline.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from phm_data import (DEFAULT_WINDOW_SEC, FEATURE_COLS, TTF_COLS, build_training_table,
                      load_cached_table, phm_subscores)


def make_model(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)


def cross_validate(table, ttf_col, n_splits, seed, max_ttf_sec=None):
    """장비 단위 GroupKFold 교차검증.

    라벨이 있는 윈도우로만 학습하고, 검증 장비의 모든 윈도우(라벨이 NaN인 구간 포함)를 예측한다.
    (모델 예측, 학습 fold 중앙값 예측)을 table 행 순서대로 반환한다.
    """
    X = table[FEATURE_COLS]
    y = table[ttf_col].clip(upper=max_ttf_sec)
    labeled = y.notna().to_numpy()
    oof = np.full(len(table), np.nan)
    naive = np.full(len(table), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, groups=table["tool"]):
        tr = tr[labeled[tr]]
        model = make_model(seed).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = np.clip(model.predict(X.iloc[va]), 0, max_ttf_sec)
        naive[va] = y.iloc[tr].median()
    return oof, naive


def phm_score(table, preds):
    """공식 채점 방식을 윈도우 단위로 근사한 점수 (낮을수록 좋음).

    장비별로 (윈도우 x 고장 모드 3개) 부분 점수의 평균을 내고 장비 점수를 모두 더한다.
    preds에 없는 고장 모드는 NaN 예측으로 채점한다.
    """
    sub = pd.DataFrame({c: phm_subscores(table[c], preds.get(c, np.nan)) for c in TTF_COLS})
    return sub.groupby(table["tool"].to_numpy()).mean().mean(axis=1).sum()


def train(train_dir, model_out, window_sec=DEFAULT_WINDOW_SEC, ttf_dir=None,
          max_ttf_hours=None, n_splits=5, seed=42, features_dir=None):
    if features_dir:
        # 캐시된 특징은 prepare_features.py 실행 시의 윈도우 길이를 따른다.
        table = load_cached_table(features_dir)
    else:
        table = build_training_table(train_dir, window_sec, ttf_dir)
    max_ttf_sec = max_ttf_hours * 3600 if max_ttf_hours else None
    k = min(n_splits, table["tool"].nunique())

    models, metrics, oof_preds = {}, {}, {}
    for ttf_col in TTF_COLS:
        y = table[ttf_col].clip(upper=max_ttf_sec)
        labeled = y.notna()
        if not labeled.any():
            print(f"[skip] {ttf_col}: 라벨이 없습니다.")
            continue

        if k >= 2:
            oof, naive = cross_validate(table, ttf_col, k, seed, max_ttf_sec)
            oof_preds[ttf_col] = oof
            m = labeled.to_numpy()
            err = oof[m] - y[m].to_numpy()
            mae = np.abs(err).mean() / 3600
            rmse = np.sqrt(np.mean(err**2)) / 3600
            naive_mae = np.abs(naive[m] - y[m].to_numpy()).mean() / 3600
            metrics[ttf_col] = {"mae_h": mae, "rmse_h": rmse, "naive_mae_h": naive_mae}
            print(f"[cv] {ttf_col}: MAE={mae:.1f}h RMSE={rmse:.1f}h "
                  f"(중앙값 예측 MAE={naive_mae:.1f}h, {k}-fold by tool)")

        models[ttf_col] = make_model(seed).fit(table.loc[labeled, FEATURE_COLS], y[labeled])

    if oof_preds:
        score = phm_score(table, oof_preds)
        nan_score = phm_score(table, {})
        metrics["phm_score"] = score
        print(f"[cv] PHM 점수(윈도우 근사, 낮을수록 좋음): {score:.2f} "
              f"(모두 NaN으로 예측할 때 {nan_score:.2f})")

    bundle = {
        "window_sec": window_sec,
        "feature_cols": FEATURE_COLS,
        "max_ttf_sec": max_ttf_sec,
        "models": models,
    }
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    print(f"[save] {model_out}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="PHM 2018 TTF 예측 베이스라인 학습")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-dir", help="학습용 센서 CSV가 있는 폴더")
    src.add_argument("--features-dir", help="prepare_features.py로 만든 특징 폴더")
    p.add_argument("--ttf-dir", default=None, help="TTF CSV 폴더 (기본: <train-dir>/train_ttf)")
    p.add_argument("--model-out", default="models/baseline.joblib")
    p.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC,
                   help="윈도우 길이(초). --features-dir를 쓰면 무시되고, 특징을 만들 때 쓴 값을 따른다.")
    p.add_argument("--max-ttf-hours", type=float, default=None,
                   help="TTF 라벨 상한(시간). 지정하면 그 이상은 상한값으로 자른다.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(args.train_dir, args.model_out, args.window_sec, args.ttf_dir,
          args.max_ttf_hours, args.n_splits, args.seed, args.features_dir)


if __name__ == "__main__":
    main()
