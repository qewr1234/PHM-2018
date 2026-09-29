import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmp2016"))
from cmp_data import (DELTA_COLS, INNER_SEED, PROCESS_COLS, STATE_COLS, TARGET, TS_FAMILIES, USAGE_COLS,  # noqa: E402
                      annotate_raw, neighbor_ext_features, neighbor_features, sequence_features, sequence_tensor,
                      shape_family, tsshape_features, wafer_features)


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


# ----------------------------------------------------------------------------- 이웃 확장 (NB_E_/NB_D_/NB_O_)
def make_ext_table(n=30):
    """같은 웨이퍼 id 로 A·B 두 스테이지 (B 는 30분 뒤), 이웃 확장 특징이 쓰는 상태 컬럼 포함."""
    rng = np.random.default_rng(1)
    a = make_table(n)
    b = make_table(n).assign(STAGE="B", T_START=np.arange(n) * 3600.0 + 1800, **{TARGET: np.arange(n) * 5.0})
    t = pd.concat([a, b], ignore_index=True)
    for c in DELTA_COLS:
        if c not in t:
            t[c] = rng.random(len(t)) * 100
    t["USAGE_OF_DRESSER_start"] = np.arange(len(t), dtype=float) * 3  # 변화량을 손으로 셀 수 있게 단조 증가
    return t


def test_neighbor_ext_features_shape_and_non_reference_labels():
    t = make_ext_table()
    ref = np.ones(len(t), bool)
    ref[[4, 17, 40]] = False  # 평가 대상 세 행
    out = neighbor_ext_features(t, ref)
    assert out.shape == (len(t), 50) and all(c.startswith(("NB_E_", "NB_D_", "NB_O_")) for c in out.columns)
    changed = t.copy()
    changed.loc[~ref, TARGET] = 9999.0  # 참조 밖 정답은 특징에 영향을 주면 안 된다
    pd.testing.assert_frame_equal(out, neighbor_ext_features(changed, ref))


def test_neighbor_ext_delta_excludes_self_and_own_inner_fold():
    t = make_ext_table()
    n = len(t)
    ref = np.ones(n, bool)
    ref[[4, 17, 40]] = False
    out = neighbor_ext_features(t, ref)
    dr = t["USAGE_OF_DRESSER_start"].to_numpy()
    inner = np.random.RandomState(INNER_SEED).randint(0, 5, n)
    stage_a = (t["STAGE"] == "A").to_numpy()
    for i in np.flatnonzero(stage_a):
        # 직전 참조 웨이퍼(같은 레짐): 참조가 아닌 행은 참조 전체에서, 참조 행은 자기 inner 폴드 밖 참조 행에서 찾는다.
        # 어느 쪽이든 자기 자신은 아니다.
        cand = [j for j in range(i) if stage_a[j] and ref[j] and (not ref[i] or inner[j] != inner[i])]
        if not cand:
            assert np.isnan(out.loc[i, "NB_D_prev_DRESSER"])
            continue
        j = max(cand)
        assert j != i and np.isclose(out.loc[i, "NB_D_prev_DRESSER"], dr[i] - dr[j])
        assert out.loc[i, "NB_D_prev_DRESSER"] != 0.0
    # 다른 스테이지(B)의 편차: 같은 웨이퍼 B 행 정답 - B 참조 평균, B 가 참조가 아니면 결측
    mu_b = t.loc[ref & ~stage_a, TARGET].mean()
    assert np.isclose(out.loc[5, "NB_O_dev"], t.loc[35, TARGET] - mu_b) and np.isclose(out.loc[5, "NB_O_gap"], 0.5)
    assert np.isnan(out.loc[10, "NB_O_dev"])  # 40번 행(B) 은 참조가 아니다
    # 과거 지수 가중 평균: 정답이 시간에 비례하므로 자기 편차보다 작고 미래 평균은 크다 (자기 자신이 섞이지 않았다는 뜻)
    dev = t[TARGET].to_numpy() - t.loc[ref & stage_a, TARGET].mean()
    mid = np.flatnonzero(stage_a & ref)[5:-5]
    assert (out.loc[mid, "NB_E_same_c5_past"].to_numpy() < dev[mid]).all()
    assert (out.loc[mid, "NB_E_same_c5_fut"].to_numpy() > dev[mid]).all()


# ----------------------------------------------------------------------------- 원본 센서 행 기반 (TS_ 정적 특징, 시퀀스 텐서)
def make_raw(n_wafers=6, seed=0):
    """웨이퍼마다 챔버 4→5→6 세 단계 (마지막 웨이퍼는 챔버 6 없음), 1 Hz 센서 행. 정답 컬럼은 없다."""
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0
    for w in range(n_wafers):
        for step, ch in enumerate([4, 5, 6]):
            if w == n_wafers - 1 and ch == 6:
                continue
            n = int(rng.integers(20, 40))
            level = rng.random(len(PROCESS_COLS)) * 100 + 10
            for i in range(n):
                on = 3 <= i < n - 3
                vals = {c: (level[j] * (1 if on else 0.2)) + rng.normal(0, 1) for j, c in enumerate(PROCESS_COLS)}
                vals["PRESSURIZED_CHAMBER_PRESSURE"] = level[0] if on else 0.0
                vals["SLURRY_FLOW_LINE_C"] = level[8] if on else 0.0
                rows.append({"WAFER_ID": w, "STAGE": "A", "CHAMBER": ch, "TIMESTAMP": t, **vals,
                             **{u: 100.0 * w + i for u in USAGE_COLS}})
                t += 1.0
            t += 5.0
    return pd.DataFrame(rows)


def make_raw_table(raw):
    table = wafer_features(raw)
    table[TARGET] = np.arange(len(table), dtype=float) * 10
    table["STAGE_B"] = 0
    return table


def test_tsshape_features_are_label_free_and_lite_families_only():
    raw = make_raw()
    table = make_raw_table(raw)
    ts = tsshape_features(table, raw=raw)
    assert len(ts) == len(table) and ts.shape[1] > 0 and all(c.startswith("TS_") for c in ts.columns)
    assert (ts.index == table.index).all()
    # 정답 컬럼을 바꾸거나 아예 없애도 같은 결과여야 한다 (정적 특징은 정답을 읽지 않는다)
    changed = table.copy()
    changed[TARGET] = 9999.0
    pd.testing.assert_frame_equal(ts, tsshape_features(changed, raw=raw))
    pd.testing.assert_frame_equal(ts, tsshape_features(table.drop(columns=[TARGET]), raw=raw))
    # level 가족(수준·분위수)은 빠지고 timing·ramp·drift·levels·lag 만 남는다
    fams = {shape_family(c[3:]) for c in ts.columns}
    assert fams <= set(TS_FAMILIES) and len(fams) >= 3 and "level" not in fams
    # 기존 표 컬럼과 |상관| > 0.98 인 컬럼(예: 단계 길이 ↔ ACTIVE_SEC_s0)은 정답 없이 가지치기된다
    assert not any(c.endswith("s0_step_sec") for c in ts.columns)


def test_sequence_tensor_shape_present_and_elapsed_channel():
    raw = make_raw()
    table = make_raw_table(raw)
    seq, present = sequence_tensor(table, raw=raw, n_points=16)
    assert seq.shape == (len(table), 3, len(PROCESS_COLS) + 1, 16) and seq.dtype == np.float32
    assert present.shape == (len(table), 3) and present[:-1].all() and present[-1].tolist() == [True, True, False]
    assert np.isnan(seq[-1, 2]).all() and np.isfinite(seq[present]).all()
    # 마지막 채널은 경과 행 수: 끝값이 그 단계의 행 수와 같다
    n_rows = annotate_raw(raw).groupby(["WAFER_ID", "STAGE", "STEP"]).size()
    assert np.isclose(seq[0, 1, -1, -1], n_rows[(0, "A", 1)]) and seq[0, 1, -1, 0] == 0.0


# ----------------------------------------------------------------------------- 결합 (클리핑 + 레짐별 NNLS, nested)
def test_blend_clips_to_training_range_and_nested_matches_apply():
    import cmp_blend as cb
    rng = np.random.default_rng(0)
    n = 300
    reg = np.array(["1A", "4A", "4B"])[rng.integers(0, 3, n)]
    y = np.where(reg == "1A", 150.0, 70.0) + rng.normal(0, 5, n)
    oof = {"a": y + rng.normal(0, 3, n), "b": y + rng.normal(0, 4, n)}
    oof["b"][0] = -80.0  # 튀는 예측은 레짐 학습 범위로 잘려야 한다
    W, nested, ins = cb.clip_and_weights(oof, y, reg, 0)
    for r, w in W.items():
        assert np.isclose(sum(w.values()), 1.0) and min(w.values()) >= 0
    bounds = cb.clip_bounds(y, reg)
    P = cb.clip_to(np.column_stack([oof["a"], oof["b"]]), reg, bounds)
    assert P[0, 1] == bounds[reg[0]][0] and (P.min(0) >= min(b[0] for b in bounds.values())).all()
    # nested: 폴드 하나를 손으로 계산하면 apply 와 같아야 한다
    fid = cb.fold_ids(0, n)
    a, b = fid != 0, fid == 0
    Wk, _, _ = cb.clip_and_weights({k: v[a] for k, v in oof.items()}, y[a], reg[a], fid[a] % 4)
    p_b = cb.apply({k: v[b] for k, v in oof.items()}, reg[b], Wk, cb.clip_bounds(y[a], reg[a]))
    assert np.allclose(p_b, cb.nested_blend(oof, y, reg, fid)[b])
    assert nested >= 0 and ins <= cb.mse(oof["a"], y)  # NNLS 는 in-sample 에서 단일 멤버보다 나쁠 수 없다


def test_tiny_members_fit_predict_per_regime_and_global():
    """세 멤버(tiny 설정)가 레짐별/전역 프로토콜로 유한한 예측을 내고, 신경망 표 가지는 TS_ 컬럼을 보지 않는다."""
    import cmp_models as cm
    rng = np.random.default_rng(0)
    n = 90
    table = pd.DataFrame({"WAFER_ID": np.arange(n), "STAGE": np.where(np.arange(n) % 3 == 2, "B", "A"),
                          "GROUP": np.where(np.arange(n) % 3 == 0, 1, 4), "T_START": np.arange(n) * 3600.0})
    table["STAGE_B"] = (table["STAGE"] == "B").astype(int)
    reg = (table["GROUP"].astype(str) + table["STAGE"]).to_numpy()
    y = np.where(reg == "1A", 150.0, 70.0) + rng.normal(0, 5, n)
    X = table[["GROUP", "STAGE_B"]].copy()
    for c in cm.GP_STATE_COLS + ["USAGE_OF_BACKING_FILM_start", "ACTIVE_SEC", "NB_time_3", "DURATION"]:
        X[c] = rng.random(n) * 100
    X["NB_E_same_h2_past"] = np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n))
    X["TS_s0_step_sec"] = rng.random(n)
    seq = rng.normal(size=(n, 3, 14, 8)).astype(np.float32)
    present = np.ones((n, 3), bool)
    present[::7, 2] = False
    seq[~present] = np.nan
    models = cm.make_models(table, seq, present, tiny=True)
    train_idx, pred_idx = np.arange(0, 70), np.arange(70, n)
    for name, spec in models.items():
        p = cm.fit_predict(spec, X, y, reg, train_idx, pred_idx)
        assert p.shape == (len(pred_idx),) and np.isfinite(p).all(), name
    cols = cm.compact_cols(list(X.columns))
    assert "NB_E_same_h2_past" in cols and "TS_s0_step_sec" not in cols and "DURATION" not in cols
    assert cm.select_columns(models["gp_state"], X.columns, X) == cm.GP_STATE_COLS
