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


def _time_since(event, time):
    """각 행에서 가장 최근 event 행으로부터 지난 시간(초)."""
    last = np.where(event, time, -np.inf)
    last = np.maximum.accumulate(last)
    out = time - last
    out[~np.isfinite(out)] = NO_EVENT
    return out


def make_row_features(df, rows=None):
    """시간순으로 정렬된 한 장비의 센서 DataFrame에서 행 단위 특징을 만든다.

    롤링 계산은 전체 행으로 하되, rows(불리언 마스크 또는 인덱스)를 주면 그 행만 담아 메모리를 아낀다.
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

    return pd.DataFrame(feats)
