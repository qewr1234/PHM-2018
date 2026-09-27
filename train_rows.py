"""행 단위 PHM 2018 모델: 남은 시간 구간 확률 예측 + 공식 채점 기준 기대 벌점 최소화.

1) sample: 장비별로 행 단위 특징을 만들고 학습용 표본을 뽑는다.
   고장 3시간 이내 행은 near_step 행마다 하나, 나머지는 far_frac 비율만 뽑고 가중치로 보정한다.
   장비마다 앞 split_q(기본 75%) 기간은 학습, 뒤 기간은 검증으로 표시한다.
   (테스트 데이터가 같은 장비의 바로 다음 기간이므로 시간 순서로 나눈다.)
2) fit: 고장 모드별로 LightGBM 다중 분류(남은 시간 구간 + "3시간 이상/없음")를 학습하고,
   예측 확률로 "NaN으로 답하기"와 "S초로 답하기"의 기대 벌점을 계산해 가장 작은 쪽을 고른다.
   검증 구간에서 공식 채점식(행 단위, 가중 평균)으로 점수를 계산해 전부 NaN 제출과 비교한다.

사용 예:
    python train_rows.py sample --rows-dir data/rows --out data/samples/rows.parquet
    python train_rows.py fit --sample data/samples/rows.parquet
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from phm_data import (FAULT_NAMES, TTF_COLS, phm_subscores, phm_subscores_final,
                      phm_subscores_secondary)
from row_features import fit_step_stats, make_row_features

BIN_EDGES = np.array([0, 30, 60, 120, 240, 480, 900, 1500, 2400, 3600, 5400, 7200, 10800])
NEAR_SEC = BIN_EDGES[-1]
NORMAL_SEC = 50000  # 이보다 고장이 멀면 정상 구간으로 본다 (Hitachi 팀 논문의 기준)
FAR_CLASS = len(BIN_EDGES) - 1  # 0..11: 가까운 구간, 12: 3시간 이상 또는 고장 없음(NaN)
N_CLASSES = FAR_CLASS + 1
ACTION_GRID = np.arange(0, 7201, 20, dtype=float)  # 숫자로 답할 때 후보 TTF(초)
META_COLS = ["tool", "time", "is_val", "weight", *TTF_COLS]


def ttf_to_class(ttf):
    ttf = np.asarray(ttf, dtype=float)
    cls = np.searchsorted(BIN_EDGES, np.nan_to_num(ttf, nan=NEAR_SEC), side="right") - 1
    cls[np.isnan(ttf) | (ttf >= NEAR_SEC)] = FAR_CLASS
    return cls


def fault_rates(faults_dir, tool, until=None):
    """장비의 고장 모드별 하루당 고장 횟수 (until 이전 기록만 사용)."""
    path = Path(faults_dir) / f"{tool}_train_fault_data.csv"
    f = pd.read_csv(path).drop_duplicates(subset=["time", "fault_name"])
    if until is not None:
        f = f[f["time"] < until]
    span_days = max((f["time"].max() - f["time"].min()) / 86400, 1) if len(f) else 1
    return {f"rate_{i}": (f["fault_name"] == name).sum() / span_days for i, name in enumerate(FAULT_NAMES)}


def build_sample(rows_dir, out_path, split_q=0.75, near_step=2, far_frac=0.02, seed=0):
    rows_dir = Path(rows_dir)
    rng = np.random.default_rng(seed)
    frames = []
    for sensor_path in sorted((rows_dir / "train").glob("*.parquet")):
        start = time.time()
        tool = sensor_path.stem
        sensor = pd.read_parquet(sensor_path)
        ttf = pd.read_parquet(rows_dir / "train_ttf" / sensor_path.name)
        if not np.array_equal(sensor["time"].to_numpy(), ttf["time"].to_numpy()):
            raise ValueError(f"{tool}: 센서와 TTF의 time이 맞지 않습니다.")

        y = ttf[TTF_COLS].to_numpy()
        near = (np.nan_to_num(y, nan=np.inf) < NEAR_SEC).any(axis=1)
        idx = np.arange(len(y))
        keep = (near & (idx % near_step == 0)) | (~near & (rng.random(len(y)) < far_frac))

        boundary = np.quantile(sensor["time"].to_numpy(), split_q)
        # 단계별 정상값 기준은 학습 기간 데이터로만 만든다(검증 기간 정보가 섞이지 않게).
        in_train = sensor["time"].to_numpy() < boundary
        # 압력 -> 유량 곡선은 모든 고장까지 NORMAL_SEC 이상 남았거나 고장이 없는 정상 구간으로만 학습한다.
        normal = (np.nan_to_num(y, nan=np.inf) >= NORMAL_SEC).all(axis=1)
        step_stats = fit_step_stats(sensor[in_train], normal=normal[in_train])
        sample = make_row_features(sensor, rows=keep, step_stats=step_stats)
        for k, v in fault_rates(rows_dir / "train_faults", tool, until=boundary).items():
            sample[k] = np.float32(v)
        sample["tool"] = tool
        sample["time"] = sensor["time"].to_numpy()[keep]
        sample["is_val"] = sample["time"].to_numpy() >= boundary
        sample["weight"] = np.where(near[keep], near_step, 1 / far_frac).astype(np.float32)
        for c in TTF_COLS:
            sample[c] = ttf[c].to_numpy()[keep]
        frames.append(sample)
        print(f"[sample] {tool}: {len(sensor)} rows -> {len(sample)} ({time.time() - start:.0f}s)", flush=True)

    out = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[save] {out_path}: {len(out)} rows")


def cost_matrix(ttf, weight, n_samples=5000, seed=0):
    """C[k, a]: 실제 TTF가 구간 k일 때 행동 a(첫 열은 NaN, 나머지는 ACTION_GRID의 숫자)의 평균 벌점.

    벌점은 최종 점수 (S1 + S2) / 2 기준이다. 동점이면 NaN(첫 열)이 먼저 선택된다.
    """
    rng = np.random.default_rng(seed)
    cls = ttf_to_class(ttf)
    actions = np.r_[np.nan, ACTION_GRID]
    C = np.zeros((N_CLASSES, len(actions)))
    for k in range(N_CLASSES):
        members = np.flatnonzero(cls == k)
        if len(members) == 0:
            continue
        p = weight[members] / weight[members].sum()
        gt = ttf[rng.choice(members, size=min(n_samples, len(members)), p=p)]
        for j, a in enumerate(actions):
            C[k, j] = phm_subscores_final(gt, np.full(len(gt), a)).mean()
    return C, actions


def decide(proba, C, actions, chunk=200_000):
    """기대 벌점이 가장 작은 행동을 고른다. 반환값은 TTF 예측(초) 또는 NaN."""
    out = np.empty(len(proba))
    for s in range(0, len(proba), chunk):
        expected = proba[s:s + chunk] @ C
        out[s:s + chunk] = actions[np.argmin(expected, axis=1)]
    return out


def file_scores(tools, weight, subscores):
    """장비(파일)별 점수 = 가중 평균 부분 점수 (행 x 고장 모드 3개). 반환: 장비별 Series."""
    df = pd.DataFrame({"tool": tools, "w": weight, "s": subscores.sum(axis=1) * weight})
    g = df.groupby("tool")
    return g["s"].sum() / (g["w"].sum() * subscores.shape[1])


def fit(sample_path, n_estimators=300, learning_rate=0.05, seed=0, save_proba=None):
    data = pd.read_parquet(sample_path)
    feature_cols = [c for c in data.columns if c not in META_COLS]
    train, val = data[~data["is_val"]], data[data["is_val"]]
    print(f"[data] train {len(train)} rows / val {len(val)} rows / {len(feature_cols)} features",
          flush=True)
    saved = {}

    preds = np.full((len(val), len(TTF_COLS)), np.nan)
    for i, c in enumerate(TTF_COLS):
        start = time.time()
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=63, min_child_samples=50,
            subsample=0.5, subsample_freq=1, colsample_bytree=0.5, max_bin=63,
            # 클래스가 극도로 불균형하고 가중치가 커서 규제가 없으면 잎 값이 폭주해 확률이 0/1로 발산한다.
            min_child_weight=10.0, reg_lambda=10.0, max_delta_step=1.0,
            random_state=seed, verbose=-1,
        )
        weight = train["weight"] / train["weight"].mean()
        model.fit(train[feature_cols], ttf_to_class(train[c]), sample_weight=weight)
        C, actions = cost_matrix(train[c].to_numpy(dtype=float), train["weight"].to_numpy())
        proba = model.predict_proba(val[feature_cols])
        preds[:, i] = decide(proba, C, actions)
        saved[f"proba_{i}"], saved[f"cost_{i}"] = proba.astype(np.float32), C
        answered = ~np.isnan(preds[:, i])
        print(f"[fit] {c}: {time.time() - start:.0f}s, 숫자로 답한 검증 행 {answered.mean():.3%}", flush=True)

    gt = val[TTF_COLS].to_numpy(dtype=float)
    w = val["weight"].to_numpy()
    if save_proba:
        # 결정 규칙을 재학습 없이 분석/조정할 수 있도록 검증 확률을 저장한다.
        np.savez(save_proba, tool=val["tool"].to_numpy(), weight=w, gt=gt, actions=actions, **saved)
    nan_preds = np.full_like(gt, np.nan)
    tools = val["tool"]
    rows = []
    for name, fn in [("S1", phm_subscores), ("S2", phm_subscores_secondary), ("final", phm_subscores_final)]:
        model = file_scores(tools, w, fn(gt, preds)).sum()
        base = file_scores(tools, w, fn(gt, nan_preds)).sum()
        rows.append((name, model, base))
        print(f"[score] {name}: model {model:.3f} / all NaN {base:.3f}")
    for i, c in enumerate(TTF_COLS):
        m = file_scores(tools, w, phm_subscores_final(gt[:, [i]], preds[:, [i]])).sum()
        n = file_scores(tools, w, phm_subscores_final(gt[:, [i]], nan_preds[:, [i]])).sum()
        print(f"[score] final {c}: model {m:.3f} / all NaN {n:.3f}")
    _, total, base = rows[-1]
    print(f"[score] 검증 최종 점수 (S1+S2)/2 (낮을수록 좋음): model {total:.3f} / all NaN {base:.3f} "
          f"-> 벌점 {1 - total / base:+.2%} 감소")
    return total, base


def main():
    p = argparse.ArgumentParser(description="PHM 2018 행 단위 모델")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="행 단위 특징 표본 생성")
    s.add_argument("--rows-dir", default="data/rows")
    s.add_argument("--out", default="data/samples/rows.parquet")
    s.add_argument("--split-q", type=float, default=0.75)
    s.add_argument("--near-step", type=int, default=2)
    s.add_argument("--far-frac", type=float, default=0.02)
    f = sub.add_parser("fit", help="학습 + 검증 점수 계산")
    f.add_argument("--sample", default="data/samples/rows.parquet")
    f.add_argument("--n-estimators", type=int, default=300)
    f.add_argument("--learning-rate", type=float, default=0.05)
    f.add_argument("--save-proba", default=None, help="검증 확률을 저장할 .npz 경로")
    args = p.parse_args()

    if args.cmd == "sample":
        build_sample(args.rows_dir, args.out, args.split_q, args.near_step, args.far_frac)
    else:
        fit(args.sample, args.n_estimators, args.learning_rate, save_proba=args.save_proba)


if __name__ == "__main__":
    main()
