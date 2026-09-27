"""학습된 모델로 테스트 센서 데이터의 TTF를 예측한다.

각 입력 파일마다 윈도우별 예측을 <out-dir>/<tool>_pred.csv 로 저장하고,
마지막 윈도우(데이터 끝 시점)의 예측값을 화면에 출력한다.

사용 예:
    python predict.py --test-dir data/synthetic/test --model models/baseline.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from phm_data import load_sensor, make_features, tool_id_from_path


def predict(test_dir, model_path, out_dir):
    bundle = joblib.load(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(Path(test_dir).glob("*.csv"))
    if not test_files:
        raise FileNotFoundError(f"{test_dir}에 CSV 파일이 없습니다.")

    results = {}
    for path in test_files:
        tool = tool_id_from_path(path)
        feats = make_features(load_sensor(path), bundle["window_sec"])
        out = feats[["time"]].reset_index(drop=True)
        out.insert(0, "tool", tool)
        for ttf_col, model in bundle["models"].items():
            pred = model.predict(feats[bundle["feature_cols"]])
            out[ttf_col] = np.clip(pred, 0, bundle["max_ttf_sec"])

        out_path = out_dir / f"{tool}_pred.csv"
        out.to_csv(out_path, index=False)
        results[tool] = out

        last = out.iloc[-1]
        summary = ", ".join(f"{c[4:]}={last[c] / 3600:.1f}h" for c in bundle["models"])
        print(f"[pred] {tool} (time={last['time']:.0f}): {summary} -> {out_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="PHM 2018 TTF 예측")
    p.add_argument("--test-dir", required=True, help="테스트 센서 CSV가 있는 폴더")
    p.add_argument("--model", default="models/baseline.joblib")
    p.add_argument("--out-dir", default="predictions")
    args = p.parse_args()
    predict(args.test_dir, args.model, args.out_dir)


if __name__ == "__main__":
    main()
