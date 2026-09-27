"""PHM 2018 원본 압축 파일(tar.gz)을 행 단위 parquet으로 변환한다.

4초 간격 원본 행을 그대로 유지하면서 형식만 바꿔, 이후 실험에서 CSV를 다시 파싱하지 않고 빠르게 읽게 한다.
    <out>/train/<tool>.parquet      센서 데이터 (수치 컬럼 float32, 범주 컬럼 int)
    <out>/train_ttf/<tool>.parquet  행별 TTF
    <out>/test/<tool>.parquet       테스트 센서 데이터
    <out>/train_faults/*.csv        고장 기록 (원본 그대로)

사용 예:
    python convert_rows.py --archive data/raw/phm_data_challenge_2018.tar.gz --out data/rows
"""
import argparse
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phm_data import SENSOR_COLS, TTF_COLS
from prepare_features import open_member

CATEGORY_COLS = ["stage", "Lot", "runnum", "recipe", "recipe_step"]


def convert_sensor(f):
    df = pd.read_csv(f, usecols=lambda c: c != "Tool")
    df["time"] = df["time"].astype(np.int64)
    for c in CATEGORY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int64)
    df[SENSOR_COLS] = df[SENSOR_COLS].astype(np.float32)
    return df[["time", *CATEGORY_COLS, *SENSOR_COLS]]


def convert_ttf(f):
    df = pd.read_csv(f)
    df["time"] = df["time"].astype(np.int64)
    df[TTF_COLS] = df[TTF_COLS].astype(np.float32)
    return df[["time", *TTF_COLS]]


def convert(archive, out_dir):
    out_dir = Path(out_dir)
    for sub in ["train", "train_ttf", "train_faults", "test"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            parts = Path(member.name).parts
            name = Path(member.name).name
            tool = name.split("_DC")[0]
            f = open_member(tf, member)
            start = time.time()

            if "train_faults" in parts:
                with open(out_dir / "train_faults" / name, "wb") as dst:
                    shutil.copyfileobj(f, dst)
                continue
            if "train_ttf" in parts:
                df, sub = convert_ttf(f), "train_ttf"
            elif "train" in parts or "test" in parts:
                df, sub = convert_sensor(f), ("train" if "train" in parts else "test")
            else:
                continue
            df.to_parquet(out_dir / sub / f"{tool}.parquet", index=False)
            print(f"[conv] {member.name}: {len(df)} rows ({time.time() - start:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description="PHM 2018 원본을 행 단위 parquet으로 변환")
    p.add_argument("--archive", required=True)
    p.add_argument("--out", default="data/rows")
    args = p.parse_args()
    convert(args.archive, args.out)


if __name__ == "__main__":
    main()
