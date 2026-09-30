"""Phase-II 고정 지연 VM 검사 (가짜 데이터): 계측하지 않은 웨이퍼 정답·D_i 뒤에 도착하는 정답이 예측에 닿지 않는지."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmp2016"))
import fab_phase2 as f2  # noqa: E402
from cmp_data import TARGET, USAGE_COLS  # noqa: E402
from fab_vm import DELTA_COLS  # noqa: E402

L = 4.0
DEV_DAY, WARM_DAY = 5.0, 1.0
SMALL = {**f2.LGB, "n_estimators": 20}
SPECS = {
    "gbm": dict(kind="lgb", resid=False, topk=None),
    "gbm_res": dict(kind="lgb", topk=None),
    "res_pool": dict(kind="lgb", pooled=True, topk=None),
    "res_6h": dict(kind="lgb", step=6.0, topk=None),
    "tfm": dict(kind="tabdpt", ctx=60, topk=8),       # TFM 경로 (열 선별 + 최근 문맥), 모델은 작은 트리로 바꿔 끼운다
}


@pytest.fixture(autouse=True)
def stub_tfm(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    monkeypatch.setattr(f2, "tfm_model", lambda kind, n_est, seed: HistGradientBoostingRegressor(max_iter=15,
                                                                                                  random_state=0))


def make_table(n=540, seed=0):
    """세 레짐이 20분 간격으로 섞여 처리되는 가짜 웨이퍼 표 (약 7.5일). 학습 70%, 테스트·검증 15%씩."""
    rng = np.random.default_rng(seed)
    reg = np.array(["1A", "4A", "4B"])[np.arange(n) % 3]
    split = rng.choice(["train", "test", "val"], n, p=[0.7, 0.15, 0.15])
    t = pd.DataFrame({
        "split": split, "WAFER_ID": np.arange(n) // 3 * 10 + (reg == "1A"), "STAGE": [r[1] for r in reg],
        "GROUP": np.where(reg == "1A", 1, 4), "T_START": np.arange(n) * 1200.0 + rng.uniform(0, 300, n),
        "DURATION": 300.0, "N_ROWS": 300.0, "OUTLIER": False, "STAGE_B": (reg == "4B").astype(int),
    })
    for c in USAGE_COLS:
        t[f"{c}_start"] = np.cumsum(rng.random(n))
        t[f"{c}_delta"] = rng.random(n)
    t["USAGE_OF_DRESSER_start"] = (np.arange(n) * 3.0) % 500
    for c in DELTA_COLS:
        if c not in t:
            t[c] = rng.normal(100, 5, n)
    base = np.where(reg == "1A", 150.0, np.where(reg == "4A", 73.0, 80.0))
    drift = 3 * np.sin(np.arange(n) / 40.0)
    t[TARGET] = base + drift - 0.01 * t["USAGE_OF_DRESSER_start"] + rng.normal(0, 1.5, n)
    return t


def pipeline(table, q=None, fits=None):
    D = f2.PhaseData(table, dev_day=DEV_DAY, warm_day=WARM_DAY, use_ts=False)
    q = f2.select_rts(D, L)[0] if q is None else q
    X = f2.build_X(D, L, q)
    P = {m: f2.fit_member(D, L, X, s, params=SMALL, min_rows=15, fits=None if fits is None else fits.setdefault(m, []))
         for m, s in SPECS.items()}
    P.update(f2.nolearn_members(X))
    names = list(P)
    for mode in ("online", "frozen", "global"):
        log = None if fits is None or mode == "global" else fits.setdefault(f"blend_{mode}", [])
        P[f"blend_{mode}"] = f2.blend(D, L, P, names, mode, min_rows=10, fits=log)[0]
    return D, X, P, q


def _same(a, b):
    return np.array_equal(a, b, equal_nan=True)


@pytest.fixture(scope="module")
def base():
    return make_table()


@pytest.fixture(scope="module")
def run0(base):
    import sklearn.ensemble
    stub = lambda kind, n_est, seed: sklearn.ensemble.HistGradientBoostingRegressor(max_iter=15, random_state=0)  # noqa
    old, f2.tfm_model = f2.tfm_model, stub
    try:
        fits = {}
        out = pipeline(base, fits=fits)
    finally:
        f2.tfm_model = old
    return (*out, fits)


def test_members_actually_predict(run0):
    D, X, P, q, fits = run0
    for m in SPECS:
        assert np.isfinite(P[m]).sum() > 100, m          # 검사가 빈 예측을 비교하지 않도록
        assert len(fits[m]) > 0


def test_unmeasured_labels_change_nothing(base, run0):
    D0, X0, P0, _, _ = run0
    t = base.copy()
    m = t["split"] != "train"
    t.loc[m, TARGET] = np.random.default_rng(1).uniform(0, 5000, m.sum())   # 테스트·검증 정답을 엉터리로
    D1, X1, P1, _ = pipeline(t)
    assert X0.equals(X1)
    for k in P0:
        assert _same(P0[k], P1[k]), k


@pytest.mark.parametrize("c_day, fix_q", [(DEV_DAY + 0.3, False), (DEV_DAY + 0.02, False), (2.4, True), (2.02, True)])
def test_late_labels_do_not_reach_earlier_predictions(base, run0, c_day, fix_q):
    """도착 시각 > C 인 학습 정답을 바꾸면 D_i <= C 인 웨이퍼의 특징·멤버·결합 예측은 그대로여야 한다.

    C 가 배포 기간이면 선택(RTS 잡음비·동결 가중치)까지 포함한 전체 경로, C 가 개발 기간 안이면 개발 기간 선택은
    고정하고(fix_q) 멤버와 online 결합만 본다. C 는 재학습 경계(6·24시간 배수)와 어긋나게(0.5시간·7시간 뒤) 둔다.
    """
    D0, X0, P0, q0, _ = run0
    C = c_day * 24.0
    t = base.copy()
    te = (t["T_START"] + t["DURATION"] - t["T_START"].min()) / 3600.0
    late = (t["split"] == "train") & (te + D0.delay_h > C)
    t.loc[late, TARGET] += np.random.default_rng(2).normal(0, 20, late.sum())
    D1, X1, P1, _ = pipeline(t, q=q0 if fix_q else None)
    early = D0.t + L <= C
    assert early.sum() > 50 and (~early).sum() > 50
    assert X0.loc[early].equals(X1.loc[early])
    keys = list(SPECS) + ["rts", "K1", "INT", "blend_online"] + ([] if fix_q else ["blend_frozen"])
    for k in keys:
        assert _same(P0[k][early], P1[k][early]), k
        assert not _same(P0[k][~early], P1[k][~early]), k   # 바뀐 정답이 실제로 뒤쪽 예측에는 닿는다
    if not fix_q:
        assert not _same(P0["blend_global"][early], P1["blend_global"][early])  # 전체 기간 가중치는 인과가 아니다


def test_training_rows_arrived_before_refit(run0):
    D, X, P, q, fits = run0
    Dt = D.t + L
    for m in SPECS:
        for T, tr, tgt in fits[m]:
            assert D.meas[tr].all(), m                    # 계측 웨이퍼만
            assert (D.arr[tr] <= T).all(), m               # 정답이 재학습 시각 전에 도착
            assert (Dt[tr] <= T).all(), m                  # 학습 행 특징(D_j 기준)도 재학습 시각 전에 계산
            assert (Dt[tgt] > T).all(), m                  # 예측은 재학습 뒤에 낸다
            step = SPECS[m].get("step", 24.0)
            assert (Dt[tgt] <= T + step).all(), m
    for mode in ("online", "frozen"):                     # 결합 가중치 적합 행과 자르기 범위도 같은 규칙
        assert fits[f"blend_{mode}"]
        for T, rows, tg, lab in fits[f"blend_{mode}"]:
            assert D.meas[rows].all() and D.meas[lab].all()
            limit = T if mode == "online" else D.dev_h    # frozen: 개발 기간 끝까지 도착한 선행 예측만
            assert (D.ready(L)[rows] <= limit).all(), mode
            assert (D.arr[lab] <= T).all() and (Dt[tg] > T).all(), mode


def test_avail_matches_definition(run0):
    """D_i 에 쓰는 정답 = 같은 레짐 계측 웨이퍼 j != i 중 a_j <= D_i 인 것 전부."""
    D = run0[0]
    for i in np.random.default_rng(3).choice(D.n, 60, replace=False):
        P, F, _ = D.avail(D.reg[i], i, L)
        want = np.flatnonzero(D.meas & (D.reg == D.reg[i]) & (D.arr <= D.t[i] + L + 1e-9) & (np.arange(D.n) != i))
        assert np.array_equal(np.sort(np.r_[P, F]), want)
        assert (P < i).all() and (F > i).all()


@pytest.mark.parametrize("lag", [0.5, 1.0, 24.0])
def test_features_use_only_arrived_labels(base, lag):
    """특징(과거·앞보기·교차 스테이지·같은 웨이퍼 다른 스테이지·RTS)만 따로: 짧은 L 에서는 다른 스테이지 정답이 아직
    도착하지 않았을 수 있다. 도착 시각 > C 인 정답을 바꿔도 D_i <= C 인 행은 같아야 한다 (C 여러 곳)."""
    D0 = f2.PhaseData(base, dev_day=DEV_DAY, warm_day=WARM_DAY, use_ts=False)
    q = {r: (0.1, 0.3) for r in f2.REGIMES}
    X0 = f2.build_X(D0, lag, q)
    if lag >= D0.delay_h:                                 # L < 지연이면 다른 스테이지 정답은 한 번도 도착하지 않는다
        assert X0["O_y"].notna().any() and X0["O_y"].isna().any()
    else:
        assert "O_y" not in X0
    te = (base["T_START"] + base["DURATION"] - base["T_START"].min()) / 3600.0
    for c_h in (30.2, 61.7, 100.1):
        t = base.copy()
        late = (t["split"] == "train") & (te + D0.delay_h > c_h)
        t.loc[late, TARGET] += 17.0
        X1 = f2.build_X(f2.PhaseData(t, dev_day=DEV_DAY, warm_day=WARM_DAY, use_ts=False), lag, q)
        early = D0.t + lag <= c_h
        assert X0.loc[early].equals(X1.loc[early]), c_h
        assert not X0.loc[~early].equals(X1.loc[~early]), c_h
