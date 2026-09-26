"""PHM 2018 Data Challenge (Ion Mill Etching) 공통 유틸리티.

데이터 로딩, 윈도우 단위 특징(feature) 추출, TTF 라벨 생성을 담당한다.
시간(time) 컬럼과 TTF 값은 모두 초(second) 단위라고 가정한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# 특징 추출에 사용할 수치형 센서 컬럼
SENSOR_COLS = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
]

# 예측 대상 고장 모드 3가지. TTF 파일의 컬럼명은 "TTF_" + 고장명이다.
FAULT_NAMES = [
    "FlowCool Pressure Dropped Below Limit",
    "Flowcool Pressure Too High Check Flowcool Pump",
    "Flowcool leak",
]
TTF_COLS = ["TTF_" + name for name in FAULT_NAMES]

STATS = ["mean", "std", "min", "max", "last"]
DEFAULT_WINDOW_SEC = 1800  # 30분 단위 윈도우
TREND_WINDOWS = 48  # 추세 특징의 기준이 되는 직전 윈도우 개수 (기본 30분 윈도우면 24시간)

FEATURE_COLS = (
    ["n_samples"]
    + [f"{c}_{s}" for c in SENSOR_COLS for s in STATS]
    + [f"{c}_trend" for c in SENSOR_COLS]
)


def tool_id_from_path(path):
    """'01_M01_DC_train.csv' -> '01_M01'"""
    return Path(path).stem.split("_DC")[0]


def load_sensor(path):
    """센서 CSV에서 time과 수치형 센서 컬럼만 읽는다. 없는 센서 컬럼은 NaN으로 채운다."""
    wanted = {"time", *SENSOR_COLS}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    if "time" not in df.columns:
        raise ValueError(f"{path}: 'time' 컬럼이 없습니다.")
    for c in SENSOR_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[SENSOR_COLS] = df[SENSOR_COLS].astype(np.float32)
    return df.sort_values("time").reset_index(drop=True)


def load_ttf(path):
    """TTF(Time To Failure) CSV를 읽는다. 마지막 고장 이후처럼 알 수 없는 구간은 NaN이다."""
    df = pd.read_csv(path)
    missing = [c for c in ["time", *TTF_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 필요한 컬럼이 없습니다: {missing}")
    return df[["time", *TTF_COLS]].sort_values("time").reset_index(drop=True)


def make_features(sensor, window_sec=DEFAULT_WINDOW_SEC):
    """원시 센서 데이터를 window_sec 초 단위 윈도우로 묶어 통계 특징을 만든다.

    인덱스는 윈도우 번호(time // window_sec)이고, 'time' 컬럼은 윈도우 내 마지막 샘플 시각이다.
    각 윈도우의 특징은 그 윈도우 끝까지의 데이터만 사용하므로 미래 정보가 섞이지 않는다.
    """
    window = (sensor["time"] // window_sec).astype(np.int64)
    grouped = sensor.groupby(window)

    feats = grouped[SENSOR_COLS].agg(STATS)
    feats.columns = [f"{c}_{s}" for c, s in feats.columns]

    # 직전 윈도우들의 평균 대비 상대 변화율: 서서히 진행되는 열화(drift)를 포착한다.
    # 상대값이라 장비마다 센서 기준값이 달라도 비교할 수 있다.
    for c in SENSOR_COLS:
        mean = feats[f"{c}_mean"]
        base = mean.rolling(TREND_WINDOWS, min_periods=1).mean().shift(1)
        feats[f"{c}_trend"] = (mean - base) / (base.abs() + 1e-6)

    feats["n_samples"] = grouped.size()
    feats["time"] = grouped["time"].max()
    feats.index.name = "window"
    return feats[["time", *FEATURE_COLS]]


def make_labels(ttf, window_sec=DEFAULT_WINDOW_SEC):
    """TTF를 윈도우 단위로 묶어 각 윈도우 끝 시점의 TTF를 라벨로 사용한다."""
    window = (ttf["time"] // window_sec).astype(np.int64)
    labels = ttf.groupby(window)[TTF_COLS].last()
    labels.index.name = "window"
    return labels


def find_train_files(train_dir, ttf_dir=None):
    """train_dir 안의 센서 CSV와 같은 이름의 TTF CSV(기본: train_dir/train_ttf)를 짝지어 반환한다."""
    train_dir = Path(train_dir)
    ttf_dir = Path(ttf_dir) if ttf_dir else train_dir / "train_ttf"
    pairs = []
    for sensor_path in sorted(train_dir.glob("*.csv")):
        ttf_path = ttf_dir / sensor_path.name
        if ttf_path.exists():
            pairs.append((sensor_path, ttf_path))
        else:
            print(f"[warn] TTF 파일이 없어 건너뜀: {ttf_path}")
    if not pairs:
        raise FileNotFoundError(f"{train_dir}에서 (센서, TTF) 파일 쌍을 찾지 못했습니다.")
    return pairs


def build_training_table(train_dir, window_sec=DEFAULT_WINDOW_SEC, ttf_dir=None):
    """모든 장비(tool)의 윈도우 특징과 TTF 라벨을 하나의 표로 합친다."""
    frames = []
    for sensor_path, ttf_path in find_train_files(train_dir, ttf_dir):
        feats = make_features(load_sensor(sensor_path), window_sec)
        labels = make_labels(load_ttf(ttf_path), window_sec)
        table = feats.join(labels, how="inner")
        table.insert(0, "tool", tool_id_from_path(sensor_path))
        frames.append(table)
        print(f"[load] {sensor_path.name}: {len(table)} windows")
    return pd.concat(frames, ignore_index=True)
