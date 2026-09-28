import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmp2016"))
from cmp_data import STATE_COLS, TARGET, neighbor_features, sequence_features  # noqa: E402


def make_table(n=12):
    rng = np.random.default_rng(0)
    t = pd.DataFrame({
        "WAFER_ID": np.arange(n), "STAGE": "A", "GROUP": 4,
        "T_START": np.arange(n) * 3600.0, TARGET: np.arange(n, dtype=float) * 10,
    })
    for c in STATE_COLS:
        t[c] = rng.random(n)
    return t


def test_neighbor_features_exclude_self():
    t = make_table()
    nb = neighbor_features(t, np.ones(len(t), bool))
    # 가장 가까운 이웃은 자기 자신이 아니라 한 시간 앞이나 뒤의 웨이퍼여야 한다.
    assert (nb["NB_time_1"] != t[TARGET]).all()
    assert nb.loc[5, "NB_time_1"] in (40.0, 60.0)


def test_neighbor_features_ignore_non_reference_labels():
    t = make_table()
    ref = np.arange(len(t)) < 8
    base = neighbor_features(t, ref)
    changed = t.copy()
    changed.loc[~ref, TARGET] = 9999.0  # 참조 밖(평가 대상) 정답을 바꿔도 특징은 그대로여야 한다
    after = neighbor_features(changed, ref)
    pd.testing.assert_frame_equal(base, after)


def test_trend_feature_follows_linear_drift():
    t = make_table(40)
    nb = neighbor_features(t, np.ones(len(t), bool))
    # 연마량이 시간에 정확히 비례하면 국소 추세 추정은 자기 값에 가깝다 (이웃 범위 안으로 잘림).
    mid = slice(10, 30)
    assert np.allclose(nb["NB_trend_15"].iloc[mid], t[TARGET].iloc[mid], atol=1e-6)


def make_two_stage_table(n=10):
    a = make_table(n)
    b = make_table(n).assign(STAGE="B", WAFER_ID=np.arange(n) + 100, T_START=np.arange(n) * 3600.0 + 1800)
    return pd.concat([a, b], ignore_index=True)


def test_sequence_features_exclude_self_and_non_reference_labels():
    t = make_two_stage_table()
    ref = np.ones(len(t), bool)
    ref[[3, 13]] = False  # 평가 대상 두 행
    base = sequence_features(t, ref)
    changed = t.copy()
    changed.loc[~ref, TARGET] = 9999.0
    pd.testing.assert_frame_equal(base, sequence_features(changed, ref))
    # 참조 행의 '바로 앞 웨이퍼'는 자기 자신이 아니라 한 시간 앞의 웨이퍼다.
    mu = t.loc[ref & (t.STAGE == "A").to_numpy(), TARGET].mean()
    assert np.isclose(base.loc[5, "same_prev1"], t.loc[4, TARGET] - mu)
    # 다른 스테이지(B) 웨이퍼는 30분 뒤에 처리되므로 A의 5번 행 바로 앞 B는 B의 4번(=14번 행)이다.
    mu_b = t.loc[ref & (t.STAGE == "B").to_numpy(), TARGET].mean()
    assert np.isclose(base.loc[5, "cross_prev1"], t.loc[14, TARGET] - mu_b)
