"""PHM 2016 Data Challenge (웨이퍼 CMP 연마량 예측) 데이터 로딩과 특징 생성.

웨이퍼·스테이지(WAFER_ID, STAGE) 하나가 예측 단위다. 한 기록 안에서 웨이퍼는 챔버 세 곳
(4→5→6 또는 1→2→3)을 차례로 거치므로 특징은 챔버 단계별로 따로 요약한다.

특징 묶음
- 공정: 단계별 압력·회전·슬러리 평균, 가공 시간
- 소모품: 패드(연마 테이블), 드레서, 멤브레인, 백킹 필름 등 사용량
- Preston: 압력 × 회전 속도의 시간 적분 (연마량 ∝ 압력 × 상대 속도 × 시간)
- 이웃: 시간·소모품 상태가 가까운 학습 웨이퍼들의 연마량 (학습 행은 자기 자신 제외)
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["WAFER_ID", "STAGE"]
TARGET = "AVG_REMOVAL_RATE"
USAGE_COLS = [
    "USAGE_OF_BACKING_FILM",
    "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE",
    "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET",
]
PRESSURE_COLS = [
    "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "RIPPLE_AIR_BAG_PRESSURE",
    "EDGE_AIR_BAG_PRESSURE",
]
FLOW_COLS = ["SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B", "SLURRY_FLOW_LINE_C"]
ROTATION_COLS = ["WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION"]
PROCESS_COLS = PRESSURE_COLS + FLOW_COLS + ROTATION_COLS + ["DRESSING_WATER_STATUS"]
# 학습 연마량 중 수천 단위 값은 측정 오류로 본다 (테스트·검증 최댓값은 164).
OUTLIER_RATE = 1000

RAW = Path(__file__).resolve().parents[1] / "data" / "cmp2016" / "raw"
SPLIT_GLOBS = {
    "train": "train_test/*/CMP-data/training/*.csv",
    "test": "train_test/*/CMP-data/test/*.csv",
    "val": "validation/*/validation/*.csv",
}
LABEL_GLOBS = {
    "train": "train_test/*/CMP-training-removalrate.csv",
    "test": "answers/*/orig_CMP-test-removalrate.csv",
    "val": "answers/*/orig_CMP-validation-removalrate.csv",
}


def load_split(split, raw_dir=RAW):
    files = sorted(Path(raw_dir).glob(SPLIT_GLOBS[split]))
    if not files:
        raise FileNotFoundError(f"{raw_dir}/{SPLIT_GLOBS[split]} 에 파일이 없습니다.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_labels(split, raw_dir=RAW):
    files = sorted(Path(raw_dir).glob(LABEL_GLOBS[split]))
    if not files:
        raise FileNotFoundError(f"{raw_dir}/{LABEL_GLOBS[split]} 에 파일이 없습니다.")
    return pd.read_csv(files[0])


def wafer_features(raw):
    """센서 행을 웨이퍼·스테이지 단위 특징으로 요약한다."""
    raw = raw.sort_values(KEY + ["TIMESTAMP"])
    # 챔버 그룹(1~3 또는 4~6)과 그룹 안에서의 단계(0, 1, 2)
    group = raw.groupby(KEY)["CHAMBER"].transform("min")
    raw = raw.assign(GROUP=np.where(group <= 3, 1, 4))
    raw = raw.assign(STEP=(raw["CHAMBER"] - raw["GROUP"]).clip(0, 2).astype(int))
    dt = raw.groupby(KEY)["TIMESTAMP"].diff().fillna(0).clip(0, 5)

    # Preston: 압력 × 회전 속도를 시간으로 적분한 값 (행 간격 약 1초)
    for p in ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE"]:
        for r in ["WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION"]:
            raw[f"PV_{p[:12]}_{r[:5]}"] = raw[p] * raw[r] * dt
    pv_cols = [c for c in raw.columns if c.startswith("PV_")]
    raw = raw.assign(ACTIVE=(raw["PRESSURIZED_CHAMBER_PRESSURE"] > 0) & (raw["SLURRY_FLOW_LINE_C"] > 0))

    g = raw.groupby(KEY)
    out = pd.DataFrame({
        "GROUP": g["GROUP"].first(),
        "T_START": g["TIMESTAMP"].min(),
        "DURATION": g["TIMESTAMP"].max() - g["TIMESTAMP"].min(),
        "N_ROWS": g.size(),
        "ACTIVE_SEC": raw.loc[raw["ACTIVE"]].groupby(KEY).size(),
    })
    for c in USAGE_COLS:
        out[f"{c}_start"] = g[c].min()
        out[f"{c}_delta"] = g[c].max() - g[c].min()
    for c in pv_cols:
        out[f"{c}_sum"] = g[c].sum()

    # 챔버 단계별 공정 변수 요약 (가압 중인 행만)
    active = raw.loc[raw["ACTIVE"]]
    step_mean = active.groupby(KEY + ["STEP"])[PROCESS_COLS].mean().unstack("STEP")
    step_mean.columns = [f"{c}_s{s}_mean" for c, s in step_mean.columns]
    step_std = active.groupby(KEY + ["STEP"])[PRESSURE_COLS].std().unstack("STEP")
    step_std.columns = [f"{c}_s{s}_std" for c, s in step_std.columns]
    step_len = active.groupby(KEY + ["STEP"]).size().unstack("STEP")
    step_len.columns = [f"ACTIVE_SEC_s{s}" for s in step_len.columns]
    out = out.join([step_mean, step_std, step_len])
    out["ACTIVE_SEC"] = out["ACTIVE_SEC"].fillna(0)
    return out.reset_index()


def build_table(raw_dir=RAW):
    """train/test/val 특징과 정답을 하나의 표로 합친다 (split 컬럼으로 구분)."""
    frames = []
    for split in ["train", "test", "val"]:
        feats = wafer_features(load_split(split, raw_dir))
        labels = load_labels(split, raw_dir)
        table = feats.merge(labels, on=KEY, how="left")
        table.insert(0, "split", split)
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    # split마다 없는 단계 컬럼이 있으면 concat 후 object 타입이 되므로 숫자로 되돌린다.
    for c in table.columns.difference(["split", "STAGE"]):
        table[c] = pd.to_numeric(table[c])
    table["STAGE_B"] = (table["STAGE"] == "B").astype(int)
    table["OUTLIER"] = (table["split"] == "train") & (table[TARGET] > OUTLIER_RATE)
    return table


NEIGHBOR_K = (1, 3, 10)
TREND_K = (15, 40)
STATE_COLS = [f"{c}_start" for c in USAGE_COLS]


def neighbor_features(table, ref_mask):
    """시간·소모품 상태가 가까운 참조(학습) 웨이퍼들의 연마량 평균.

    같은 챔버 그룹과 스테이지 안에서만 찾는다. 참조 집합에 속한 행은 자기 자신을 빼고 계산해
    정답이 특징으로 새지 않게 한다 (leave-one-out).
    """
    out = pd.DataFrame(index=table.index)
    ref_mask = np.asarray(ref_mask)
    scale = table.loc[ref_mask, STATE_COLS].std().replace(0, 1)
    for (grp, stage), idx in table.groupby(["GROUP", "STAGE"]).groups.items():
        idx = np.asarray(idx)
        ref = idx[ref_mask[idx]]
        if len(ref) < 2:
            continue
        y_ref = table.loc[ref, TARGET].to_numpy()
        spaces = {
            "time": table.loc[:, ["T_START"]].to_numpy(dtype=float) / 3600.0,
            "state": (table.loc[:, STATE_COLS] / scale).to_numpy(dtype=float),
        }
        for name, X in spaces.items():
            d = np.linalg.norm(X[idx][:, None, :] - X[ref][None, :, :], axis=2)
            d[idx[:, None] == ref[None, :]] = np.inf  # 자기 자신 제외
            order = np.argsort(d, axis=1)
            for k in NEIGHBOR_K:
                kk = min(k, len(ref) - 1)
                out.loc[idx, f"NB_{name}_{k}"] = y_ref[order[:, :kk]].mean(axis=1)
            out.loc[idx, f"NB_{name}_dist1"] = np.take_along_axis(d, order[:, :1], axis=1)[:, 0]
            if name == "time":
                # 국소 추세: 시간상 가까운 k개 웨이퍼로 직선을 맞춰 이 웨이퍼 시점의 값을 추정한다
                # (패드·드레서가 닳으며 연마량이 천천히 변하는 흐름을 반영).
                t_ref = X[ref, 0]
                for k in TREND_K:
                    nn = order[:, :min(k, len(ref) - 1)]
                    tt, yy = t_ref[nn], y_ref[nn]
                    tc = tt - tt.mean(axis=1, keepdims=True)
                    slope = (tc * (yy - yy.mean(axis=1, keepdims=True))).sum(axis=1) / np.maximum((tc ** 2).sum(axis=1), 1e-9)
                    est = yy.mean(axis=1) + slope * (X[idx, 0] - tt.mean(axis=1))
                    # 멀리 외삽해 튀지 않도록 이웃 범위 안으로 자른다
                    out.loc[idx, f"NB_trend_{k}"] = np.clip(est, yy.min(axis=1), yy.max(axis=1))
                    out.loc[idx, f"NB_slope_{k}"] = slope

    # 같은 웨이퍼의 다른 스테이지(A↔B) 연마량이 참조 집합에 있으면 쓴다.
    ref_rates = table.loc[ref_mask, ["WAFER_ID", "STAGE", TARGET]]
    other = table[["WAFER_ID", "STAGE"]].assign(STAGE=table["STAGE"].map({"A": "B", "B": "A"}))
    out["OTHER_STAGE_RATE"] = other.merge(ref_rates, on=["WAFER_ID", "STAGE"], how="left")[TARGET].to_numpy()
    return out


def regime(table):
    """연마 조건(챔버 그룹 + 스테이지): '1A', '4A', '4B'."""
    return (table["GROUP"].astype(int).astype(str) + table["STAGE"]).to_numpy()


def sequence_features(table, ref_mask, n=3):
    """처리 순서상 바로 앞·뒤 참조 웨이퍼들의 연마량이 레짐 평균보다 얼마나 높거나 낮았는지.

    same: 같은 레짐, cross: 같은 챔버 그룹의 다른 스테이지(4A↔4B, 같은 패드·슬러리를 공유).
    참조 집합에 속한 행은 자기 자신을 빼고 계산한다.
    """
    ref_mask = np.asarray(ref_mask)
    reg = regime(table)
    mu = table.loc[ref_mask].groupby(reg[ref_mask])[TARGET].mean()
    dev = (table[TARGET] - pd.Series(reg).map(mu).to_numpy()).to_numpy()
    hours = table["T_START"].to_numpy() / 3600.0
    out = pd.DataFrame(index=table.index)
    out["regime_mu"] = pd.Series(reg).map(mu).to_numpy()
    for r in np.unique(reg):
        idx = np.flatnonzero(reg == r)
        sources = {"same": [r], "cross": [q for q in np.unique(reg) if q != r and q[0] == r[0]]}
        for name, regs in sources.items():
            cand = np.flatnonzero(ref_mask & np.isin(reg, regs))
            if len(cand) < n + 1:
                continue
            order = np.argsort(hours[cand])
            ct, cd, cid = hours[cand][order], dev[cand][order], cand[order]
            for i in idx:
                pos = np.searchsorted(ct, hours[i])
                prev = [j for j in range(pos - 1, max(pos - n - 2, -1), -1) if cid[j] != i][:n]
                nxt = [j for j in range(pos, min(pos + n + 1, len(ct))) if cid[j] != i][:n]
                if prev:
                    out.loc[i, f"{name}_prev1"] = cd[prev[0]]
                    out.loc[i, f"{name}_prev{n}"] = cd[prev].mean()
                    out.loc[i, f"{name}_prev_gap"] = hours[i] - ct[prev[0]]
                if nxt:
                    out.loc[i, f"{name}_next1"] = cd[nxt[0]]
                    out.loc[i, f"{name}_next{n}"] = cd[nxt].mean()
                    out.loc[i, f"{name}_next_gap"] = ct[nxt[0]] - hours[i]
    return out
