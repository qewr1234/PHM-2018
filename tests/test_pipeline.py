import numpy as np
import pandas as pd

from make_synthetic import generate
from phm_data import FEATURE_COLS, SENSOR_COLS, TTF_COLS, make_features, make_labels
from predict import predict
from train import train


def test_make_features_uses_window_statistics():
    sensor = pd.DataFrame({"time": np.arange(0, 40, 5, dtype=float)})  # 0,5,...,35
    for c in SENSOR_COLS:
        sensor[c] = np.arange(8, dtype=np.float32)

    feats = make_features(sensor, window_sec=10)

    assert list(feats.index) == [0, 1, 2, 3]
    assert list(feats["time"]) == [5, 15, 25, 35]
    assert list(feats["n_samples"]) == [2, 2, 2, 2]
    assert list(feats["FLOWCOOLPRESSURE_mean"]) == [0.5, 2.5, 4.5, 6.5]
    assert list(feats["FLOWCOOLPRESSURE_last"]) == [1, 3, 5, 7]
    # 첫 윈도우는 비교할 과거가 없고, 두 번째 윈도우는 (2.5 - 0.5) / 0.5
    assert np.isnan(feats["FLOWCOOLPRESSURE_trend"].iloc[0])
    assert np.isclose(feats["FLOWCOOLPRESSURE_trend"].iloc[1], 4.0)
    assert list(feats.columns) == ["time", *FEATURE_COLS]


def test_make_labels_takes_ttf_at_window_end():
    ttf = pd.DataFrame({"time": [0.0, 5.0, 10.0, 15.0]})
    for c in TTF_COLS:
        ttf[c] = [20.0, 15.0, 10.0, np.nan]

    labels = make_labels(ttf, window_sec=10)

    assert list(labels[TTF_COLS[0]]) == [15.0, 10.0]


def test_end_to_end_on_synthetic_data(tmp_path):
    train_dir, test_dir = generate(tmp_path / "data", n_tools=3, days=12, test_days=3, seed=1)
    model_path = tmp_path / "model.joblib"

    metrics = train(train_dir, model_path, n_splits=3, max_ttf_hours=72)
    results = predict(test_dir, model_path, tmp_path / "pred")

    assert set(metrics) == set(TTF_COLS)
    for m in metrics.values():
        assert np.isfinite(m["mae_h"]) and m["mae_h"] >= 0
    assert len(results) == 3
    for out in results.values():
        assert list(out.columns) == ["tool", "time", *TTF_COLS]
        assert (out[TTF_COLS] >= 0).all().all()
        assert (out[TTF_COLS] <= 72 * 3600 + 1e-6).all().all()
