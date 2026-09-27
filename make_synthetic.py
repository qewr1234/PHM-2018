"""실제 PHM 2018 데이터 없이 파이프라인을 시험해 볼 수 있는 합성 데이터 생성기.

실제 데이터와 같은 폴더 구조와 컬럼명을 사용한다.
    <out>/train/01_M01_DC_train.csv                       센서 데이터
    <out>/train/train_ttf/01_M01_DC_train.csv             고장까지 남은 시간(TTF, 초)
    <out>/train/train_faults/01_M01_train_fault_data.csv  고장 발생 기록
    <out>/test/01_M01_DC_test.csv                         테스트 센서 데이터

각 고장 직전 일정 시간 동안 FlowCool 관련 센서가 점점 벗어나도록(열화) 만들어,
모델이 학습할 수 있는 신호를 넣는다.

사용 예:
    python make_synthetic.py --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phm_data import FAULT_NAMES, SENSOR_COLS, TTF_COLS

DAY = 86400
HOUR = 3600

# 고장 모드별 열화 패턴: (센서, 기준값 대비 최대 변화율)
DEGRADATION = {
    FAULT_NAMES[0]: [("FLOWCOOLPRESSURE", -0.4)],
    FAULT_NAMES[1]: [("FLOWCOOLPRESSURE", +0.4)],
    FAULT_NAMES[2]: [("FLOWCOOLFLOWRATE", -0.4), ("IONGAUGEPRESSURE", +0.3)],
}


def sample_fault_times(rng, span):
    """3~10일 간격으로 고장 시각을 뽑는다."""
    times, t = [], rng.uniform(2 * DAY, 6 * DAY)
    while t < span:
        times.append(t)
        t += rng.uniform(3 * DAY, 10 * DAY)
    return np.array(times)


def generate_tool(rng, tool, base_levels, days, dt):
    """장비 하나의 센서 데이터, TTF, 고장 기록을 만든다."""
    time = np.arange(0, days * DAY, dt, dtype=np.float64)
    n = len(time)

    # 같은 기종이라 장비 간 기준값 차이는 ±3% 정도로 작다.
    base = {c: level * rng.uniform(0.97, 1.03) for c, level in base_levels.items()}
    sensor = pd.DataFrame({"time": time, "Tool": tool, "stage": "A",
                           "Lot": time // (6 * HOUR), "runnum": time // HOUR,
                           "recipe": 1, "recipe_step": 1})
    for c in SENSOR_COLS:
        sensor[c] = base[c] * (1 + 0.02 * rng.standard_normal(n))

    ttf = pd.DataFrame({"time": time})
    fault_rows = []
    for name, ttf_col in zip(FAULT_NAMES, TTF_COLS):
        fault_times = sample_fault_times(rng, time[-1])
        for t_fault in fault_times:
            # 고장 전 1~3일 동안 센서 값이 점점(제곱 형태로) 벗어난다.
            length = rng.uniform(1 * DAY, 3 * DAY)
            mask = (time > t_fault - length) & (time <= t_fault)
            progress = (time[mask] - (t_fault - length)) / length
            for col, rate in DEGRADATION[name]:
                sensor.loc[mask, col] += base[col] * rate * progress**2
            fault_rows.append({"time": t_fault, "Tool": tool, "fault_name": name})

        # 다음 고장까지 남은 시간. 마지막 고장 이후는 알 수 없으므로 NaN.
        idx = np.searchsorted(fault_times, time, side="left")
        next_fault = np.append(fault_times, np.nan)[idx]
        ttf[ttf_col] = next_fault - time

    faults = pd.DataFrame(fault_rows, columns=["time", "Tool", "fault_name"]).sort_values("time")
    return sensor, ttf, faults


def generate(out_dir, n_tools=4, days=30, test_days=10, dt=60, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    for d in [train_dir / "train_ttf", train_dir / "train_faults", test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_levels = {c: rng.uniform(1, 100) for c in SENSOR_COLS}
    for i in range(n_tools):
        tool = f"{i + 1:02d}_M01"
        sensor, ttf, faults = generate_tool(rng, tool, base_levels, days, dt)
        sensor.to_csv(train_dir / f"{tool}_DC_train.csv", index=False)
        ttf.to_csv(train_dir / "train_ttf" / f"{tool}_DC_train.csv", index=False)
        faults.to_csv(train_dir / "train_faults" / f"{tool}_train_fault_data.csv", index=False)

        test_sensor, _, _ = generate_tool(rng, tool, base_levels, test_days, dt)
        test_sensor.to_csv(test_dir / f"{tool}_DC_test.csv", index=False)
        print(f"[gen] {tool}: train {len(sensor)} rows, {len(faults)} faults / test {len(test_sensor)} rows")
    return train_dir, test_dir


def main():
    p = argparse.ArgumentParser(description="PHM 2018 형식의 합성 데이터 생성")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-tools", type=int, default=4)
    p.add_argument("--days", type=int, default=30, help="장비당 학습 데이터 기간(일)")
    p.add_argument("--test-days", type=int, default=10, help="장비당 테스트 데이터 기간(일)")
    p.add_argument("--dt", type=int, default=60, help="샘플링 간격(초)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.out, args.n_tools, args.days, args.test_days, args.dt, args.seed)


if __name__ == "__main__":
    main()
