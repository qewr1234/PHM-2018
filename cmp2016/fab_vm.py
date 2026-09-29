"""FAB 적용을 가정한 실시간(인과) 가상 계측(VM) 구성 요소.

대회 방식(시험 웨이퍼의 앞·뒤 학습 웨이퍼 정답을 모두 참고)과 달리, 여기서는 웨이퍼가 처리되는 순서대로 흘려보내며
그 시각까지 **도착한 계측값만** 쓴다.

- 계측 지연: 웨이퍼 w 의 정답은 연마가 끝난 뒤 delay 시간이 지나야 도착한다 (도착 시각 = T_END + delay).
- 표본 계측: 모든 웨이퍼를 재지 않는다. 계측할 웨이퍼는 연마 직후에 정한다 (무작위·주기·불확실성 기반).
- 계측값 검증: 같은 조건 과거 계측의 중앙값보다 훨씬 큰 값(측정 오류)은 이력에 넣지 않는다.
- 두 가지 쓰임새
  * vm       : 연마 직후, 이 웨이퍼의 센서 기록까지 써서 연마량을 추정 (품질 감시, 계측 대체)
  * forecast : 연마 전에, 소모품 카운터와 과거 계측만으로 연마량을 예측 (run-to-run 제어의 연마 시간 결정)

구성 요소
- WaferStream       : 웨이퍼 표(시간순), 정답 없이 만드는 웨이퍼 특징(센서 요약·TS_ 모양·일정·드레서 수명)
- History           : 도착·검증된 계측값 (레짐별, 공정 시각 순)
- history_features  : 예측 시점의 이력 특징 (최근 편차, 지수가중 평균, 추세, 같은 드레서 수명 평균, 상태 차이 등)
- 예측기            : EWMA(run-to-run 방식), 칼만 동적 선형 모델, 주기 재학습 LightGBM, 온라인 결합(NNLS),
                      적응형 컨포멀 구간(ACI)
"""
import bisect
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from cmp_data import (CACHE_DIR, EXCLUDE, KEY, TARGET, align_keys, build_table, load_raw_all,  # noqa: E402
                      prune_columns, regime, shape_family, shape_features, TS_FAMILIES)

HOUR = 3600.0
REGIMES = ("1A", "4A", "4B")
CROSS = {"4A": "4B", "4B": "4A"}
DRESSER_RESET = 200          # 드레서 사용량이 이만큼 떨어지면 교체
MEAS_REJECT_RATIO = 3.0      # 과거 계측 중앙값의 이 배수를 넘으면 측정 오류로 보고 버린다
MEAS_REJECT_ABS = 1000.0     # 이력이 없을 때의 절대 기준 (연마량은 50~170 범위)

# 예측 시점에 이 웨이퍼 자신의 센서 기록이 필요한 컬럼 (forecast 모드에서는 뺀다)
FORECAST_OK = ("USAGE_", "GROUP", "STAGE_B", "SCH_", "LIFE_")
EWM_HOURS = (0.5, 2.0, 8.0, 24.0)
EWM_COUNT = (2, 5, 15)
TREND_K = (15, 40)
DELTA_COLS = ["ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum", "SLURRY_FLOW_LINE_C_s0_mean",
              "MAIN_OUTER_AIR_BAG_PRESSURE_s0_mean", "USAGE_OF_DRESSER_start", "USAGE_OF_POLISHING_TABLE_start"]


# ==============================================================================================
# 웨이퍼 흐름과 정답 없는 웨이퍼 특징
# ==============================================================================================
class WaferStream:
    """모든 웨이퍼·스테이지(학습·테스트·검증)를 연마 종료 시각 순으로 늘어놓은 표와 정답 없는 특징.

    split 은 보고용으로만 남긴다 (FAB 에는 split 이 없다). 정답 열(y)은 시뮬레이터가 '도착' 처리할 때만 읽는다.
    ts_fit_mask: TS_ 모양 특징 가지치기(정답 없음)에 쓸 행 — 개발 기간 행만 주면 배포 기간 데이터를 전혀 보지 않는다.
    """

    def __init__(self, table=None, ts_dev_day=None, use_ts=True):
        t = build_table() if table is None else table.copy()
        t["REG"] = regime(t)
        t["T_END"] = t["T_START"] + t["DURATION"]
        t0 = t["T_START"].min()
        t["DAY"] = (t["T_END"] - t0) / 86400.0
        t = t.sort_values(["T_END", "WAFER_ID", "STAGE"]).reset_index(drop=True)
        self.t = t
        self.n = len(t)
        self.y = t[TARGET].to_numpy(dtype=float)
        self.reg = t["REG"].to_numpy()
        self.t_end = t["T_END"].to_numpy(dtype=float)
        self.t_start = t["T_START"].to_numpy(dtype=float)
        self.day = t["DAY"].to_numpy(dtype=float)
        self.split = t["split"].to_numpy()
        self.outlier = t["OUTLIER"].to_numpy() | (self.y > MEAS_REJECT_ABS)
        self.base = self._base_features()
        self.sched = self._schedule_features()
        self.ts = self._ts_features(ts_dev_day) if use_ts else pd.DataFrame(index=t.index)
        # 같은 웨이퍼의 다른 스테이지 (챔버 그룹 4의 A↔B)
        pos = {(w, s): i for i, (w, s) in enumerate(zip(t["WAFER_ID"], t["STAGE"]))}
        other = {"A": "B", "B": "A"}
        self.partner = np.array([pos.get((w, other[s]), -1) for w, s in zip(t["WAFER_ID"], t["STAGE"])])
        self.delta = t[DELTA_COLS].to_numpy(dtype=float)

    def _base_features(self):
        cols = [c for c in self.t.columns if c not in EXCLUDE and c not in ("REG", "T_END", "DAY")]
        return self.t[cols].astype(float)

    def _schedule_features(self):
        """정답 없이 과거 웨이퍼 기록만으로 만드는 일정·소모품 수명 특징 (연마 시작 시각 기준, 모두 인과적)."""
        t = self.t
        out = pd.DataFrame(index=t.index)
        for g, x in t.sort_values("T_START").groupby("GROUP"):
            idx = x.index.to_numpy()
            ts = x["T_START"].to_numpy() / HOUR
            prev_end = np.r_[np.nan, x["T_END"].to_numpy()[:-1] / HOUR]
            out.loc[idx, "SCH_gap_prev_h"] = ts - prev_end
            out.loc[idx, "SCH_n_prev_3h"] = np.arange(len(ts)) - np.searchsorted(ts, ts - 3.0)
            dr = x["USAGE_OF_DRESSER_start"].to_numpy()
            life = np.r_[0, np.cumsum(np.diff(dr) < -DRESSER_RESET)]
            out.loc[idx, "LIFE_id"] = life
            first_t = pd.Series(ts).groupby(life).transform("min").to_numpy()
            out.loc[idx, "LIFE_hours"] = ts - first_t
            out.loc[idx, "LIFE_wafers"] = pd.Series(np.ones(len(ts))).groupby(life).cumsum().to_numpy() - 1
        return out

    def _ts_features(self, dev_day):
        f = CACHE_DIR / "tsshape.parquet"
        df = pd.read_parquet(f) if f.exists() else shape_features(load_raw_all())
        if not f.exists():
            df.to_parquet(f)
        aligned = align_keys(self.t, df)
        rows = np.ones(self.n, bool) if dev_day is None else self.day < dev_day
        names = [c for c in prune_columns(aligned[rows], self.base[rows]) if shape_family(c) in TS_FAMILIES]
        out = aligned[names].astype(float)
        out.columns = [f"TS_{c}" for c in names]
        return out

    def static_matrix(self, mode):
        """정답 없는 특징 표. forecast 모드는 연마 전에 알 수 있는 컬럼(카운터·일정·수명)만 남긴다."""
        X = pd.concat([self.base, self.sched, self.ts], axis=1)
        if mode == "forecast":
            X = X[[c for c in X.columns if c.startswith(FORECAST_OK)]]
        return X


# ==============================================================================================
# 도착한 계측값 이력
# ==============================================================================================
class History:
    """도착·검증된 계측값. 레짐별로 공정(연마 종료) 시각 순서를 유지한다."""

    def __init__(self, stream):
        self.s = stream
        self.idx = {r: [] for r in REGIMES}
        self.tk = {r: [] for r in REGIMES}
        self.sum = {r: 0.0 for r in REGIMES}
        self.have = np.zeros(stream.n, bool)

    def add(self, i):
        r = self.s.reg[i]
        k = bisect.bisect(self.tk[r], self.s.t_end[i])
        self.tk[r].insert(k, self.s.t_end[i])
        self.idx[r].insert(k, i)
        self.sum[r] += self.s.y[i]
        self.have[i] = True

    def accept(self, i):
        """계측값 검증: 같은 레짐 과거 계측 중앙값의 MEAS_REJECT_RATIO 배를 넘으면 측정 오류로 버린다."""
        v, r = self.s.y[i], self.s.reg[i]
        if not np.isfinite(v):
            return False
        ids = self.idx[r]
        if len(ids) >= 10:
            return v < MEAS_REJECT_RATIO * np.median(self.s.y[ids[-200:]])
        return v < MEAS_REJECT_ABS

    def arrays(self, r):
        ids = np.asarray(self.idx[r], dtype=int)
        return ids, self.s.y[ids], self.s.t_end[ids]

    def mean(self, r):
        n = len(self.idx[r])
        return self.sum[r] / n if n else np.nan


def _ewm(d, w):
    s = w.sum()
    return (w * d).sum() / s if s > 1e-3 else np.nan


def _trend(tt, dd, t_now):
    if len(dd) < 5:
        return np.nan
    tc = tt - tt.mean()
    slope = (tc * (dd - dd.mean())).sum() / max((tc ** 2).sum(), 1e-9)
    return float(np.clip(dd.mean() + slope * (t_now - tt.mean()), dd.min(), dd.max()))


def _block(prefix, d, te, p, full=True):
    """편차 배열 d(공정 시각 te 순)에서 요약 특징."""
    out = {}
    age = (p - te) / HOUR
    out[f"{prefix}last1"] = d[-1]
    out[f"{prefix}mean3"] = d[-3:].mean()
    out[f"{prefix}age1"] = age[-1]
    hs = EWM_HOURS if full else (2.0, 8.0)
    for h in hs:
        out[f"{prefix}ewmh{h:g}"] = _ewm(d, 0.5 ** (age / h))
    if full:
        out[f"{prefix}mean10"] = d[-10:].mean()
        out[f"{prefix}n24"] = float((age < 24).sum())
        rank = np.arange(len(d))[::-1]
        for k in EWM_COUNT:
            out[f"{prefix}ewmn{k}"] = _ewm(d, 0.5 ** (rank / k))
        for k in TREND_K:
            out[f"{prefix}trend{k}"] = _trend(te[-k:] / HOUR, d[-k:], p / HOUR)
    return out


def history_features(s, H, i, p, mode="vm"):
    """예측 시점 p(초)에 도착해 있는 계측값만으로 만드는 웨이퍼 i 의 이력 특징 (H_: 같은 레짐, C_: 짝 스테이지)."""
    r = s.reg[i]
    out = {"H_mu": H.mean(r), "H_n": float(len(H.idx[r]))}
    ids, yy, te = H.arrays(r)
    if len(ids):
        d = yy - out["H_mu"]
        out.update(_block("H_", d, te, p))
        # 같은 드레서 수명 안에서 잰 웨이퍼들의 평균 편차 (수명 초반·말기 수준)
        life = s.sched["LIFE_id"].to_numpy()
        m = life[ids] == life[i]
        out["H_life_mean"] = d[m].mean() if m.any() else np.nan
        out["H_life_n"] = float(m.sum())
        # 마지막 계측 웨이퍼 대비 상태 차이 ("그 뒤로 무엇이 달라졌나")
        j = ids[-1]
        for c, name in enumerate(DELTA_COLS):
            if mode == "forecast" and not name.startswith("USAGE_"):
                continue
            out[f"H_delta_{name}"] = s.delta[i, c] - s.delta[j, c]
    rc = CROSS.get(r)
    if rc is not None and len(H.idx[rc]):
        ids2, yy2, te2 = H.arrays(rc)
        out.update(_block("C_", yy2 - H.mean(rc), te2, p, full=False))
    j = s.partner[i]
    if j >= 0 and H.have[j]:
        out["O_dev"] = s.y[j] - H.mean(s.reg[j])
        out["O_age"] = (p - s.t_end[j]) / HOUR
    return out


# ==============================================================================================
# 예측기
# ==============================================================================================
class EWMAPredictor:
    """run-to-run 제어의 EWMA 추정기: 레짐별 수준 = λ·새 계측 + (1-λ)·이전 수준 (계측 도착 순서로 갱신)."""

    name = "ewma"

    def __init__(self, lam=0.3, lam_reset=None):
        self.lam, self.lam_reset, self.level, self.life = lam, lam_reset, {}, {}

    def observe(self, s, i):
        r = s.reg[i]
        life = s.sched["LIFE_id"].iat[i]
        lam = self.lam
        if self.lam_reset is not None and r in self.life and life != self.life[r]:
            lam = self.lam_reset  # 드레서 교체 뒤 첫 계측: 이전 수준을 거의 버린다
        self.level[r] = s.y[i] if r not in self.level else lam * s.y[i] + (1 - lam) * self.level[r]
        self.life[r] = life

    def predict(self, s, i, p, feats):
        return self.level.get(s.reg[i], np.nan), np.nan


class KalmanDLM:
    """레짐별 동적 선형 모델: y = 수준 + b·z + 잡음, 수준과 b 는 시간에 따라 천천히 움직이는 랜덤 워크.

    z 는 드레서 수명 안 위치(사용량/100)와, vm 모드에서는 가공 시간·연마 구간 슬러리량(표준화). 칼만 필터로
    계측이 도착할 때마다 갱신하고, 예측 시점까지 경과 시간만큼 불확실성을 키운다 (계측 지연·표본 계측을 자연스럽게 다룬다).
    """

    name = "kalman"

    def __init__(self, mode="vm", q_level=0.5, q_beta=0.002, r_meas=4.0, reset_var=0.0):
        self.mode, self.q_level, self.q_beta, self.r_meas, self.reset_var = mode, q_level, q_beta, r_meas, reset_var
        self.state = {}

    def _z(self, s, i):
        dres = s.t["USAGE_OF_DRESSER_start"].iat[i] / 100.0 - 4.0
        z = [dres]
        if self.mode == "vm":
            a = s.t["ACTIVE_SEC"].iat[i]
            z.append((a - 200.0) / 60.0 if np.isfinite(a) else 0.0)
        return np.array(z)

    def _init(self, s, i):
        k = 1 + len(self._z(s, i))
        m = np.zeros(k)
        m[0] = s.y[i]
        P = np.diag([25.0] + [4.0] * (k - 1))
        return {"m": m, "P": P, "t": s.t_end[i], "life": s.sched["LIFE_id"].iat[i]}

    def _advance(self, st, t, life=None):
        """시각 t 까지 시간 갱신. 그 사이 드레서가 교체됐으면(수명 번호가 바뀜) 수준의 분산을 reset_var 만큼 키운다
        (개입 분석: 교체 직후 첫 계측이 수준을 크게 옮길 수 있게)."""
        dt = max(t - st["t"], 0.0) / HOUR
        k = len(st["m"])
        Q = np.diag([self.q_level] + [self.q_beta] * (k - 1)) * dt
        if life is not None and life != st["life"] and self.reset_var > 0:
            Q[0, 0] += self.reset_var
        return st["m"], st["P"] + Q

    def observe(self, s, i):
        r = s.reg[i]
        if r not in self.state:
            self.state[r] = self._init(s, i)
            return
        st = self.state[r]
        life = s.sched["LIFE_id"].iat[i]
        m, P = self._advance(st, s.t_end[i], life)
        h = np.r_[1.0, self._z(s, i)]
        S = h @ P @ h + self.r_meas
        K = P @ h / S
        m = m + K * (s.y[i] - h @ m)
        P = P - np.outer(K, h @ P)
        st.update(m=m, P=P, t=max(st["t"], s.t_end[i]), life=max(st["life"], life))

    def predict(self, s, i, p, feats):
        st = self.state.get(s.reg[i])
        if st is None:
            return np.nan, np.nan
        m, P = self._advance(st, p, s.sched["LIFE_id"].iat[i])
        h = np.r_[1.0, self._z(s, i)]
        return float(h @ m), float(h @ P @ h + self.r_meas)


class OnlineGBM:
    """주기적으로 다시 학습하는 LightGBM. 학습 행 = 도착한 계측 웨이퍼, 특징 = 그 웨이퍼를 예측할 당시 만든 행
    (따라서 학습·예측 모두 같은 인과 조건). 특징에는 같은 시점의 EWMA·칼만 예측(M_)도 들어간다.

    per_regime: 레짐별 모델(True) 또는 레짐 표시 컬럼을 둔 전역 모델(False).
    residual  : 목표를 y - 칼만 예측으로 두고, 예측에 칼만 예측을 더한다 (칼만이 수준을, 트리가 웨이퍼별 보정을 맡는다).
    학습 행이 min_rows 보다 적으면 예측하지 않는다(NaN → 결합에서 제외).
    """

    name = "gbm"

    def __init__(self, retrain_hours=24.0, min_rows=60, topk=120, n_estimators=400, learning_rate=0.04,
                 num_leaves=15, min_child_samples=10, window_days=None, per_regime=True, residual=False, n_jobs=1,
                 seed=0):
        self.retrain_hours, self.min_rows, self.topk, self.window_days = retrain_hours, min_rows, topk, window_days
        self.per_regime, self.residual = per_regime, residual
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=n_jobs)
        self.models, self.next_fit, self.fit_seconds = {}, -np.inf, []

    def _offset(self, X):
        if not self.residual:
            return np.zeros(len(X))
        k = X["M_kalman"].to_numpy(dtype=float) if "M_kalman" in X else np.full(len(X), np.nan)
        mu = X["H_mu"].to_numpy(dtype=float)
        return np.where(np.isfinite(k), k, mu)

    def _fit_one(self, X, y):
        X = X.loc[:, X.notna().any()]
        cols = list(X.columns)
        if self.topk and len(cols) > self.topk:
            scr = lgb.LGBMRegressor(**{**self.params, "n_estimators": 120, "learning_rate": 0.1}).fit(X, y)
            g = pd.Series(scr.booster_.feature_importance("gain"), index=cols)
            cols = list(g.sort_values(ascending=False).index[: self.topk])
        return cols, lgb.LGBMRegressor(**self.params).fit(X[cols], y)

    def maybe_fit(self, s, H, rows, p):
        if p < self.next_fit:
            return
        import time
        t0 = time.time()
        self.next_fit = p + self.retrain_hours * HOUR
        groups = [[r] for r in REGIMES] if self.per_regime else [list(REGIMES)]
        for regs in groups:
            ids = np.asarray([j for r in regs for j in H.idx[r]], dtype=int)
            if self.window_days is not None:
                ids = ids[s.t_end[ids] >= p - self.window_days * 86400]
            if len(ids) < self.min_rows * len(regs):
                continue
            X = rows.loc[ids]
            y = s.y[ids] - self._offset(X)
            fitted = self._fit_one(X, y)
            for r in regs:
                self.models[r] = fitted
        self.fit_seconds.append(time.time() - t0)

    def predict(self, s, i, p, feats_row):
        m = self.models.get(s.reg[i])
        if m is None:
            return np.nan, np.nan
        cols, model = m
        X = feats_row.reindex(columns=cols)
        return float(model.predict(X)[0] + self._offset(feats_row)[0]), np.nan


class OnlineGP:
    """레짐별 온라인 가우시안 프로세스 (Han 외 2025 의 online GP 를 단순화).

    입력 = 연마 종료 시각(시간), 드레서 사용량(/100), vm 모드에서는 가공 시간(/60). 목표 = y - 창 평균.
    커널 = C·RBF(ARD) + White. 초매개변수는 retrain_hours 마다 최근 window 개 계측으로 주변가능도 최적화하고,
    그 사이에는 새 계측이 도착할 때마다 초매개변수를 고정한 채 사후분포만 다시 계산한다 (O(window³), 수 ms).
    """

    name = "gp"

    def __init__(self, mode="vm", window=300, retrain_hours=24.0):
        self.mode, self.window, self.retrain_hours = mode, window, retrain_hours
        self.kernel, self.next_opt, self.model, self.dirty = {}, {}, {}, {r: True for r in REGIMES}
        self.H = None

    def _x(self, s, ids):
        ids = np.atleast_1d(ids)
        cols = [s.t_end[ids] / HOUR, s.t["USAGE_OF_DRESSER_start"].to_numpy()[ids] / 100.0]
        if self.mode == "vm":
            a = s.t["ACTIVE_SEC"].to_numpy()[ids]
            cols.append(np.where(np.isfinite(a), a, 0.0) / 60.0)
        return np.column_stack(cols)

    def observe(self, s, i):
        self.dirty[s.reg[i]] = True

    def bind(self, H):
        self.H = H

    def _refresh(self, s, r, p):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        ids = np.asarray(self.H.idx[r][-self.window:], dtype=int)
        if len(ids) < 15:
            return
        X, y = self._x(s, ids), s.y[ids]
        mu = y.mean()
        opt = r not in self.kernel or p >= self.next_opt.get(r, -np.inf)
        if opt:
            k0 = ConstantKernel(10.0, (0.1, 1e3)) * RBF(np.r_[5.0, np.ones(X.shape[1] - 1)], (1e-2, 1e3)) \
                + WhiteKernel(4.0, (0.1, 100.0))
            g = GaussianProcessRegressor(k0, normalize_y=False, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0)
            self.next_opt[r] = p + self.retrain_hours * HOUR
        else:
            g = GaussianProcessRegressor(self.kernel[r], normalize_y=False, optimizer=None)
        g.fit(X, y - mu)
        self.kernel[r] = g.kernel_
        self.model[r] = (g, mu)
        self.dirty[r] = False

    def predict(self, s, i, p, feats):
        r = s.reg[i]
        if self.dirty[r] or p >= self.next_opt.get(r, np.inf):
            self._refresh(s, r, p)
        if r not in self.model:
            return np.nan, np.nan
        g, mu = self.model[r]
        x = self._x(s, i)
        x[0, 0] = p / HOUR if self.mode == "forecast" else x[0, 0]
        m, sd = g.predict(x, return_std=True)
        return float(m[0] + mu), float(sd[0] ** 2)


class OnlineBlend:
    """멤버 예측을 레짐별로 결합한다. 가중치 = 최근 window 개 계측 웨이퍼에서 (당시 저장한) 멤버 예측으로 푼 NNLS,
    예측은 과거 계측 범위로 자른다. 멤버 예측이 없으면(NaN) 남은 멤버로 다시 정규화한다.
    """

    def __init__(self, names, window=150, refit_every=10, prior=None, method="nnls"):
        self.names, self.window, self.refit_every, self.method = names, window, refit_every, method
        self.w = {r: np.full(len(names), 1.0 / len(names)) if prior is None else np.asarray(prior, float) for r in REGIMES}
        self.count = {r: 0 for r in REGIMES}

    def refit(self, r, P, y):
        ok = np.isfinite(P).all(axis=1)
        if ok.sum() < 30:
            return
        A, b = P[ok][-self.window:], y[ok][-self.window:]
        if self.method == "inv3":
            # Di 외(2017)의 결합: 가중치 ∝ 1 / (최근 MSE)^3
            e = ((A - b[:, None]) ** 2).mean(axis=0)
            w = 1.0 / np.maximum(e, 1e-6) ** 3
        else:
            w, _ = nnls(A, b)
        if w.sum() > 0:
            self.w[r] = w / w.sum()

    def combine(self, r, preds, lo, hi):
        p = np.asarray(preds, float)
        w = self.w[r].copy()
        w[~np.isfinite(p)] = 0
        if w.sum() == 0:
            ok = np.isfinite(p)
            if not ok.any():
                return np.nan
            w = ok.astype(float)
        v = float((w * np.nan_to_num(p)).sum() / w.sum())
        return float(np.clip(v, lo, hi)) if np.isfinite(lo) else v


class AdaptiveConformal:
    """적응형 컨포멀 추론(ACI): 레짐별로 최근 절대 오차의 (1-α_t) 분위수를 구간 폭으로 쓰고,
    계측이 도착해 구간을 벗어났는지 확인할 때마다 α_t ← α_t + γ(α - 벗어남) 으로 조정한다.
    """

    def __init__(self, alpha=0.1, gamma=0.01, window=400):
        self.alpha, self.gamma, self.window = alpha, gamma, window
        self.a = {r: alpha for r in REGIMES}
        self.scores = {r: [] for r in REGIMES}

    def width(self, r):
        sc = self.scores[r][-self.window:]
        if len(sc) < 20:
            return np.nan
        q = min(max(1 - self.a[r], 0.0), 1.0)
        return float(np.quantile(sc, q))

    def update(self, r, err, issued_width):
        if np.isfinite(issued_width):
            miss = float(err > issued_width)
            self.a[r] = self.a[r] + self.gamma * (self.alpha - miss)
        self.scores[r].append(err)


# ==============================================================================================
# 서비스: FAB 에 붙인다면 이런 모양 (MES/FDC 가 사건을 넘겨준다)
# ==============================================================================================
STALE_H = 8.0          # 같은 조건 마지막 계측 이후 이 시간 넘게 지났으면 '오래됨'
DISAGREE = 3.0         # 멤버 예측 표준편차가 이보다 크면 '불일치'
COLD_N = 30            # 같은 조건 계측 이력이 이보다 적으면 '초기'


def make_members(mode, cfg):
    m = [EWMAPredictor(**cfg.get("ewma", {})), KalmanDLM(mode=mode, **cfg.get("kalman", {}))]
    if cfg.get("gp"):
        m.append(OnlineGP(mode=mode, **(cfg["gp"] if isinstance(cfg["gp"], dict) else {})))
    if cfg.get("gbm") is not False:
        m.append(OnlineGBM(**cfg.get("gbm", {})))
    return m


class VMService:
    """온라인 VM 서비스.

    on_metrology(j)  : 계측값 도착 (검증 → 이력·멤버·결합 가중치·컨포멀 갱신). 받아들였으면 True.
    maybe_retrain(p) : 재학습 시각이면 LightGBM 을 다시 학습 (야간 배치 같은 주기 작업).
    on_wafer(i, p)   : 웨이퍼 i 의 예측 요청 (vm: 연마 직후, forecast: 연마 직전). 결과 dict:
                       pred, half_width(90% 구간), members, kal_sd, age_h, flags(no_active/stale/disagree/cold)
    웨이퍼는 스트림 행 번호 i 로 가리킨다 (실제 연동에서는 WAFER_ID·STAGE 로 특징을 조회하는 부분만 바뀐다).
    """

    def __init__(self, s, mode="vm", cfg=None, static=None):
        cfg = cfg or {}
        self.s, self.mode = s, mode
        self.members = make_members(mode, cfg)
        self.names = [m.name for m in self.members]
        self.gbm = next((m for m in self.members if m.name == "gbm"), None)
        self.blend = OnlineBlend(self.names, **cfg.get("blend", {}))
        self.aci = AdaptiveConformal(**cfg.get("aci", {}))
        self.H = History(s)
        for m in self.members:
            if hasattr(m, "bind"):
                m.bind(self.H)
        self.static = static if static is not None or self.gbm is None else s.static_matrix(mode)
        self.hist_rows = {}
        self.P = np.full((s.n, len(self.members)), np.nan)
        self.pred = np.full(s.n, np.nan)
        self.width = np.full(s.n, np.nan)
        self.since_refit = {r: 0 for r in REGIMES}

    def on_metrology(self, j):
        s = self.s
        if not self.H.accept(j):
            return False
        self.H.add(j)
        for m in self.members:
            if hasattr(m, "observe"):
                m.observe(s, j)
        r = s.reg[j]
        self.aci.update(r, abs(s.y[j] - self.pred[j]), self.width[j])
        self.since_refit[r] += 1
        if self.since_refit[r] >= self.blend.refit_every:
            ids = np.asarray(self.H.idx[r], dtype=int)
            self.blend.refit(r, self.P[ids], s.y[ids])
            self.since_refit[r] = 0
        return True

    def maybe_retrain(self, p):
        if self.gbm is None or p < self.gbm.next_fit:
            return
        ids = [j for r in REGIMES for j in self.H.idx[r]]
        if not ids:
            return
        hr = pd.DataFrame.from_dict({j: self.hist_rows[j] for j in ids}, orient="index")
        self.gbm.maybe_fit(self.s, self.H, self.static.loc[ids].join(hr), p)

    def on_wafer(self, i, p):
        s, r = self.s, self.s.reg[i]
        feats = history_features(s, self.H, i, p, self.mode)
        kal_sd = np.nan
        for k, m in enumerate(self.members):
            if m.name == "gbm":
                continue
            v, var = m.predict(s, i, p, feats)
            self.P[i, k] = v
            feats[f"M_{m.name}"] = v
            if m.name == "kalman":
                kal_sd = np.sqrt(var) if np.isfinite(var) else np.nan
                feats["M_kalman_sd"] = kal_sd
        if self.gbm is not None:
            self.hist_rows[i] = feats
            row = self.static.loc[[i]].join(pd.DataFrame([feats], index=[i]))
            self.P[i, self.names.index("gbm")] = self.gbm.predict(s, i, p, row)[0]
        _, yy, _ = self.H.arrays(r)
        lo, hi = (yy.min(), yy.max()) if len(yy) else (np.nan, np.nan)
        self.pred[i] = self.blend.combine(r, self.P[i], lo, hi)
        self.width[i] = self.aci.width(r)
        ok = np.isfinite(self.P[i])
        spread = float(np.std(self.P[i][ok])) if ok.sum() > 1 else np.nan
        age = feats.get("H_age1", np.nan)
        active = s.t["ACTIVE_SEC"].iat[i]
        flags = {
            "no_active": bool(self.mode == "vm" and not (np.isfinite(active) and active > 0)),
            "stale": bool(not np.isfinite(age) or age > STALE_H),
            "disagree": bool(np.isfinite(spread) and spread > DISAGREE),
            "cold": bool(feats["H_n"] < COLD_N),
        }
        return {"pred": self.pred[i], "half_width": self.width[i], "members": dict(zip(self.names, self.P[i])),
                "kal_sd": kal_sd, "spread": spread, "age_h": age, "flags": flags}
