"""FAB 스트리밍 시뮬레이터 검사: 미래 정답을 보지 않는지, 계측 지연·검증·표본 비율이 지켜지는지."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmp2016"))
from cmp_data import TARGET, USAGE_COLS  # noqa: E402
from fab_sim import simulate  # noqa: E402
from fab_vm import DELTA_COLS, History, WaferStream  # noqa: E402

SMALL_GBM = {"gbm": {"min_rows": 15, "n_estimators": 20, "topk": None, "retrain_hours": 6.0}}


def make_table(n=240, seed=0):
    """세 레짐이 섞여 30분 간격으로 처리되는 가짜 웨이퍼 표 (드레서 사용량은 증가하다 한 번 교체)."""
    rng = np.random.default_rng(seed)
    reg = np.array(["1A", "4A", "4B"])[np.arange(n) % 3]
    group = np.where(reg == "1A", 1, 4)
    t = pd.DataFrame({
        "split": "train", "WAFER_ID": np.arange(n) // 2 * 10 + (reg == "1A"), "STAGE": [r[1] for r in reg],
        "GROUP": group, "T_START": np.arange(n) * 1800.0, "DURATION": 360.0, "N_ROWS": 340.0,
        "OUTLIER": False, "STAGE_B": (reg == "4B").astype(int),
    })
    for c in USAGE_COLS:
        t[f"{c}_start"] = np.cumsum(rng.random(n))
        t[f"{c}_delta"] = rng.random(n)
    t["USAGE_OF_DRESSER_start"] = (np.arange(n) * 5.0) % 600
    for c in DELTA_COLS:
        if c not in t:
            t[c] = rng.normal(100, 5, n)
    base = np.where(reg == "1A", 150.0, np.where(reg == "4A", 73.0, 80.0))
    t[TARGET] = base - 0.01 * t["USAGE_OF_DRESSER_start"] + rng.normal(0, 2, n)
    return t


def test_predictions_do_not_see_future_labels():
    t = make_table()
    s1 = WaferStream(t, use_ts=False)
    r1 = simulate(s1, mode="vm", delay_h=0.5, cfg=SMALL_GBM)
    cutoff = np.quantile(s1.t_end, 0.5)
    t2 = t.copy()
    later = (t2["T_START"] + t2["DURATION"]) >= cutoff
    t2.loc[later, TARGET] += 25.0  # 컷오프 이후 연마된 웨이퍼의 정답만 바꾼다
    s2 = WaferStream(t2, use_ts=False)
    r2 = simulate(s2, mode="vm", delay_h=0.5, cfg=SMALL_GBM)
    before = s1.t_end < cutoff  # 이 웨이퍼들의 예측 시각에는 바뀐 정답이 아직 도착하지 않았다
    a, b = r1["pred"].to_numpy()[before], r2["pred"].to_numpy()[before]
    assert np.allclose(a, b, equal_nan=True)
    assert not np.allclose(r1["pred"].to_numpy()[~before], r2["pred"].to_numpy()[~before], equal_nan=True)


def test_metrology_delay_is_respected():
    s = WaferStream(make_table(), use_ts=False)
    rec = simulate(s, mode="vm", delay_h=3.0, cfg={"gbm": False})
    age = rec["age_h"].to_numpy()
    assert np.nanmin(age) >= 3.0 - 1e-9  # 가장 최근에 쓴 계측도 연마 후 3시간 이상 지난 것


def test_forecast_mode_uses_no_own_sensor_columns():
    s = WaferStream(make_table(), use_ts=False)
    cols = s.static_matrix("forecast").columns
    assert "ACTIVE_SEC" not in cols and "DURATION" not in cols and "N_ROWS" not in cols
    assert "USAGE_OF_DRESSER_start" in cols


def test_measurement_validation_rejects_gross_errors():
    t = make_table()
    t.loc[100, TARGET] = 4000.0
    s = WaferStream(t, use_ts=False)
    H = History(s)
    for i in range(60):
        H.add(i)
    i_bad = int(np.flatnonzero(s.y == 4000.0)[0])
    assert not H.accept(i_bad)
    assert H.accept(int(np.flatnonzero(s.y < 200)[70]))


def test_sampling_policies_follow_budget():
    s = WaferStream(make_table(n=600), use_ts=False)
    for policy in ("random", "periodic", "smart"):
        rec = simulate(s, mode="vm", delay_h=1.0, policy=policy, budget=0.2, cfg={"gbm": False})
        assert abs(rec["measured"].mean() - 0.2) < 0.07, policy
