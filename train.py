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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from phm_data import DEFAULT_WINDOW_SEC, FEATURE_COLS, TTF_COLS, build_training_table


def make_model(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)


def cross_validate(X, y, groups, n_splits, seed, max_ttf_sec=None):
    """장비 단위 교차검증. (모델 MAE, 모델 RMSE, 중앙값 예측 MAE)를 시간(hour) 단위로 반환한다."""
    oof = np.zeros(len(y))
    naive = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model(seed).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = np.clip(model.predict(X.iloc[va]), 0, max_ttf_sec)
        naive[va] = y.iloc[tr].median()
    err = oof - y.to_numpy()
    return (
        np.abs(err).mean() / 3600,
        np.sqrt(np.mean(err**2)) / 3600,
        np.abs(naive - y.to_numpy()).mean() / 3600,
    )


def train(train_dir, model_out, window_sec=DEFAULT_WINDOW_SEC, ttf_dir=None,
          max_ttf_hours=None, n_splits=5, seed=42):
    table = build_training_table(train_dir, window_sec, ttf_dir)
    max_ttf_sec = max_ttf_hours * 3600 if max_ttf_hours else None

    models, metrics = {}, {}
    for ttf_col in TTF_COLS:
        data = table.dropna(subset=[ttf_col])
        if data.empty:
            print(f"[skip] {ttf_col}: 라벨이 없습니다.")
            continue
        X = data[FEATURE_COLS]
        y = data[ttf_col].clip(upper=max_ttf_sec)
        groups = data["tool"]

        k = min(n_splits, groups.nunique())
        if k >= 2:
            mae, rmse, naive_mae = cross_validate(X, y, groups, k, seed, max_ttf_sec)
            metrics[ttf_col] = {"mae_h": mae, "rmse_h": rmse, "naive_mae_h": naive_mae}
            print(f"[cv] {ttf_col}: MAE={mae:.1f}h RMSE={rmse:.1f}h "
                  f"(중앙값 예측 MAE={naive_mae:.1f}h, {k}-fold by tool)")

        models[ttf_col] = make_model(seed).fit(X, y)

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
    p.add_argument("--train-dir", required=True, help="학습용 센서 CSV가 있는 폴더")
    p.add_argument("--ttf-dir", default=None, help="TTF CSV 폴더 (기본: <train-dir>/train_ttf)")
    p.add_argument("--model-out", default="models/baseline.joblib")
    p.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC, help="윈도우 길이(초)")
    p.add_argument("--max-ttf-hours", type=float, default=None,
                   help="TTF 라벨 상한(시간). 지정하면 그 이상은 상한값으로 자른다.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(args.train_dir, args.model_out, args.window_sec, args.ttf_dir,
          args.max_ttf_hours, args.n_splits, args.seed)


if __name__ == "__main__":
    main()
