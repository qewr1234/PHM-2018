"""PHM 2018 원본 압축 파일(tar.gz)을 풀지 않고 스트리밍으로 읽어 윈도우 특징을 저장한다.

원본은 압축을 풀면 약 34GB라, 파일을 하나씩 읽어 윈도우 특징과 라벨만 남긴다.
    <out>/train/<tool>_DC_train.csv       학습 센서 데이터의 윈도우 특징
    <out>/train_ttf/<tool>_DC_train.csv   윈도우별 TTF 라벨
    <out>/train_faults/...                고장 기록 (원본 그대로 복사)
    <out>/test/<tool>_DC_test.csv         테스트 센서 데이터의 윈도우 특징

사용 예:
    python prepare_features.py --archive data/raw/phm_data_challenge_2018.tar.gz --out data/features
    python train.py --features-dir data/features
"""
import argparse
import io
import shutil
import tarfile
import time
from pathlib import Path

from phm_data import DEFAULT_WINDOW_SEC, load_sensor, load_ttf, make_features, make_labels


class _StreamReader(io.RawIOBase):
    """tarfile 스트리밍 모드의 파일 객체는 seekable()이 없어 pandas가 바로 읽지 못하므로 감싼다."""

    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def readinto(self, buf):
        data = self.f.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def prepare(archive, out_dir, window_sec=DEFAULT_WINDOW_SEC):
    out_dir = Path(out_dir)
    for sub in ["train", "train_ttf", "train_faults", "test"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # "r|gz": 압축 파일을 앞에서부터 한 번만 읽는 스트리밍 모드
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            parts = Path(member.name).parts
            name = Path(member.name).name
            f = io.BufferedReader(_StreamReader(tf.extractfile(member)), buffer_size=1 << 20)
            start = time.time()

            if "train_faults" in parts:
                with open(out_dir / "train_faults" / name, "wb") as dst:
                    shutil.copyfileobj(f, dst)
                continue
            if "train_ttf" in parts:
                out = make_labels(load_ttf(f), window_sec)
                out_path = out_dir / "train_ttf" / name
            elif "train" in parts or "test" in parts:
                split = "train" if "train" in parts else "test"
                out = make_features(load_sensor(f), window_sec)
                out_path = out_dir / split / name
            else:
                continue

            out.to_csv(out_path)
            print(f"[prep] {member.name}: {len(out)} windows ({time.time() - start:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description="PHM 2018 원본 압축 파일에서 윈도우 특징 추출")
    p.add_argument("--archive", required=True, help="phm_data_challenge_2018.tar.gz 경로")
    p.add_argument("--out", default="data/features")
    p.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC)
    args = p.parse_args()
    prepare(args.archive, args.out, args.window_sec)


if __name__ == "__main__":
    main()
