import numpy as np
import pandas as pd

from phm_data import SENSOR_COLS
from row_features import fit_step_stats, make_row_features
from train_rows import FAR_CLASS, N_CLASSES, cost_matrix, decide, ttf_to_class


def test_ttf_to_class_bins_and_far():
    cls = ttf_to_class([0, 29, 30, 3599, 10799, 10800, np.nan])
    assert list(cls[:5]) == [0, 0, 1, 8, 11]
    assert cls[5] == FAR_CLASS and cls[6] == FAR_CLASS


def test_decide_answers_nan_when_far_and_number_when_certain():
    rng = np.random.default_rng(0)
    ttf = np.r_[rng.uniform(900, 1500, 1000), np.full(1000, np.nan)]
    C, actions = cost_matrix(ttf, np.ones(len(ttf)))

    far = np.zeros((1, N_CLASSES))
    far[0, FAR_CLASS] = 1
    near = np.zeros((1, N_CLASSES))
    near[0, ttf_to_class([1200])[0]] = 1

    assert np.isnan(decide(far, C, actions)[0])
    assert 900 <= decide(near, C, actions)[0] <= 1500


def test_row_features_are_causal():
    n = 200
    df = pd.DataFrame({"time": np.arange(n) * 4.0, "stage": 1, "Lot": 1, "runnum": 1,
                       "recipe": 1, "recipe_step": 1})
    for c in SENSOR_COLS:
        df[c] = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    stats = fit_step_stats(df.iloc[:100])
    base = make_row_features(df, step_stats=stats)
    changed = df.copy()
    changed.loc[150:, SENSOR_COLS] += 100  # 미래 값만 바꿈
    after = make_row_features(changed, step_stats=stats)
    assert "since_p_low" in base.columns and "FLOWCOOLPRESSURE_z" in base.columns
    pd.testing.assert_frame_equal(base.iloc[:150], after.iloc[:150])
