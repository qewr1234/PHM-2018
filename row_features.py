"""행(약 4초) 단위 특징 생성.

모든 특징은 현재 행과 그 이전 행만 사용한다(미래 정보 없음).
공식 채점이 고장 직전 약 1시간의 예측으로 결정되므로, 윈도우로 뭉개지 않고 행마다 특징을 만든다.
"""
import numpy as np
import pandas as pd

from phm_data import SENSOR_COLS

# 롤링 통계를 계산할 핵심 센서 (FlowCool 계통 + 챔버 압력 + 빔 전류)
KEY_SENSORS = [
    "FLOWCOOLPRESSURE",
    "FLOWCOOLFLOWRATE",
    "IONGAUGEPRESSURE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORCURRENT",
]
# 행 개수 기준 롤링 윈도우: 약 1분, 5분, 30분, 3시간 (샘플 간격 약 4초)
ROLL_ROWS = {"1m": 15, "5m": 75, "30m": 450, "3h": 2700}
LONG_ROWS = {"1d": 21600}  # 최근 하루 안에 있었던 이상 이탈(이전 고장 흔적)을 잡기 위한 긴 윈도우
GAP_SEC = 60
NO_EVENT = 1e7  # 해당 이벤트가 아직 없었을 때 쓰는 큰 값(초)

# 레시피 단계별 정상값 기준. 단계마다 설정이 달라 같은 센서 값도 정상 범위가 다르다.
STEP_KEYS = ["recipe", "recipe_step"]
Z_ROLL_ROWS = {"1m": 15, "5m": 75, "30m": 450}
EXCURSION_Z = 4.0  # 단계 기준 대비 이 이상 벗어나면 "이탈 이벤트"
EVENT_COUNT_ROWS = {"1h": 900, "1d": 21600}
STOP_GAPS = {"5m": 300, "1h": 3600}  # 장비 정지로 볼 데이터 공백 길이


def fit_step_stats(df):
    """(레시피, 단계)별 핵심 센서의 중앙값과 강건한 산포(IQR 기반)를 구한다. 학습 기간 데이터로만 호출한다."""
    g = df.groupby(STEP_KEYS)[KEY_SENSORS]
    med = g.median()
    iqr = (g.quantile(0.75) - g.quantile(0.25)) / 1.349
    global_std = df[KEY_SENSORS].std().to_numpy()
    # 설정값처럼 단계 안에서 거의 일정한 센서는 산포가 0에 가까워 z가 폭주하므로 하한을 둔다.
    scale = np.maximum(iqr, np.maximum(0.1 * global_std, 1e-3))
    return {
        "med": med,
        "scale": scale,
        "global_med": df[KEY_SENSORS].median().to_numpy(),
        "global_scale": np.maximum(global_std, 1e-3),
    }


def step_zscores(df, stats):
    """각 행의 핵심 센서를 해당 (레시피, 단계) 기준으로 표준화한다. 처음 보는 단계는 장비 전체 기준을 쓴다."""
    idx = pd.MultiIndex.from_arrays([df[k].to_numpy() for k in STEP_KEYS], names=STEP_KEYS)
    med = stats["med"].reindex(idx).to_numpy(copy=True)
    scale = stats["scale"].reindex(idx).to_numpy(copy=True)
    unseen = np.isnan(med)
    med[unseen] = np.broadcast_to(stats["global_med"], med.shape)[unseen]
    scale[unseen] = np.broadcast_to(stats["global_scale"], scale.shape)[unseen]
    return (df[KEY_SENSORS].to_numpy(dtype=np.float64) - med) / scale


def _time_since(event, time):
    """각 행에서 가장 최근 event 행으로부터 지난 시간(초)."""
    last = np.where(event, time, -np.inf)
    last = np.maximum.accumulate(last)
    out = time - last
    out[~np.isfinite(out)] = NO_EVENT
    return out


def make_row_features(df, rows=None, step_stats=None):
    """시간순으로 정렬된 한 장비의 센서 DataFrame에서 행 단위 특징을 만든다.

    롤링 계산은 전체 행으로 하되, rows(불리언 마스크 또는 인덱스)를 주면 그 행만 담아 메모리를 아낀다.
    step_stats(fit_step_stats 결과)를 주면 레시피 단계별 이탈 특징과 반복 고장 흔적 특징을 추가한다.
    """
    time = df["time"].to_numpy(dtype=np.float64)
    feats = {}

    def put(name, values):
        values = np.asarray(values, dtype=np.float32)
        feats[name] = values if rows is None else values[rows]

    for c in SENSOR_COLS:
        put(c, df[c].to_numpy())
    for c in ["stage", "recipe", "recipe_step"]:
        put(c, df[c].to_numpy())

    # 운전 상태 변화 이후 경과 시간: 데이터 공백(정지) 후 재가동, 레시피 단계/런/웨이퍼 전환
    dt = np.diff(time, prepend=time[0])
    put("dt", dt)
    put("since_gap", _time_since(dt > GAP_SEC, time))
    for c in ["recipe_step", "runnum", "Lot"]:
        v = df[c].to_numpy()
        changed = np.r_[True, v[1:] != v[:-1]]
        put(f"since_{c}_change", _time_since(changed, time))

    for c in KEY_SENSORS:
        s = df[c].astype(np.float32)
        for name, w in ROLL_ROWS.items():
            r = s.rolling(w, min_periods=1)
            mean = r.mean()
            put(f"{c}_mean_{name}", mean)
            put(f"{c}_std_{name}", r.std())
            put(f"{c}_min_{name}", r.min())
            put(f"{c}_max_{name}", r.max())
            put(f"{c}_dev_{name}", s - mean)
        if c in ("FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE"):
            for name, w in LONG_ROWS.items():
                r = s.rolling(w, min_periods=1)
                put(f"{c}_min_{name}", r.min())
                put(f"{c}_max_{name}", r.max())

    # 유량 대비 압력: 같은 유량에서 압력이 비정상적으로 낮거나 높은지
    feats["pressure_minus_flow"] = feats["FLOWCOOLPRESSURE"] - feats["FLOWCOOLFLOWRATE"]

    # 장비 정지(데이터 공백) 이력: 고장 후에는 정지와 재가동이 반복되는 경향이 있다.
    for name, sec in STOP_GAPS.items():
        stop = dt > sec
        put(f"since_stop_{name}", _time_since(stop, time))
        put(f"stops_{name}_1d", pd.Series(stop, dtype=np.float32).rolling(21600, min_periods=1).sum())

    if step_stats is not None:
        z = step_zscores(df, step_stats)
        for j, c in enumerate(KEY_SENSORS):
            zc = pd.Series(z[:, j], dtype=np.float32)
            put(f"{c}_z", zc)
            for name, w in Z_ROLL_ROWS.items():
                r = zc.rolling(w, min_periods=1)
                put(f"{c}_z_mean_{name}", r.mean())
                put(f"{c}_z_min_{name}", r.min())
                put(f"{c}_z_max_{name}", r.max())

        # 단계 기준을 1분 이상 크게 벗어난 이탈 이벤트: 이전 고장의 흔적(같은 고장은 절반이 하루 안에 재발).
        # 단계 전환 순간의 짧은 튐을 거르기 위해 1분 이동 평균으로 판단한다.
        zp = pd.Series(z[:, 0]).rolling(Z_ROLL_ROWS["1m"], min_periods=1).mean().to_numpy()
        zf = pd.Series(z[:, 1]).rolling(Z_ROLL_ROWS["1m"], min_periods=1).mean().to_numpy()
        events = {
            "p_low": zp < -EXCURSION_Z,
            "p_high": zp > EXCURSION_Z,
            "f_low": zf < -EXCURSION_Z,
            "f_high": zf > EXCURSION_Z,
        }
        for name, ev in events.items():
            put(f"since_{name}", _time_since(ev, time))
            ev_series = pd.Series(ev, dtype=np.float32)
            for wname, w in EVENT_COUNT_ROWS.items():
                put(f"n_{name}_{wname}", ev_series.rolling(w, min_periods=1).sum())

    return pd.DataFrame(feats)
