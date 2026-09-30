"""PHM 2016 Data Challenge (웨이퍼 CMP 연마량 예측) 데이터 로딩과 특징 생성.

웨이퍼·스테이지(WAFER_ID, STAGE) 하나가 예측 단위다. 한 기록 안에서 웨이퍼는 챔버 세 곳
(4→5→6 또는 1→2→3)을 차례로 거치므로 특징은 챔버 단계별로 따로 요약한다.

특징 묶음
- 공정: 단계별 압력·회전·슬러리 평균, 가공 시간
- 소모품: 패드(연마 테이블), 드레서, 멤브레인, 백킹 필름 등 사용량
- Preston: 압력 × 회전 속도의 시간 적분 (연마량 ∝ 압력 × 상대 속도 × 시간)
- 시퀀스 모양(TS_*, 정답 미사용): 단계별 채널의 램프업/다운·안정화 시간, 정체 구간 드리프트, 수준 변화 횟수,
  채널 간 시작·종료 지연, 단계 앞뒤 비활성 시간. 원본 센서 행에서 만들어 data/cmp2016/ 에 캐시한다.
- 이웃: 시간·소모품 상태가 가까운 학습 웨이퍼들의 연마량 (학습 행은 자기 자신 제외)
- 이웃 확장(NB_E_/NB_D_/NB_O_): 과거·미래 참조 웨이퍼 편차의 지수 가중 이동평균, 직전·직후 참조 웨이퍼 대비 공정
  상태 변화량, 같은 웨이퍼 다른 스테이지의 편차. 참조 행은 inner 5-fold 로 자기 폴드 밖 참조 행만 써서 계산한다.
- 시퀀스 텐서: 단계별 13 채널 궤적을 64 점으로 보간한 배열 (융합 신경망 입력, 정답 미사용)
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
# 특징이 아닌 표 컬럼 (정답·식별자·시각). 정적 특징 가지치기와 모델 입력 모두 이 집합을 뺀다.
EXCLUDE = {"split", "WAFER_ID", "STAGE", TARGET, "OUTLIER", "T_START"}

RAW = Path(__file__).resolve().parents[1] / "data" / "cmp2016" / "raw"
CACHE_DIR = RAW.parent  # 원본에서 만든 정적 특징·시퀀스 캐시 (data/ 는 git 에 넣지 않는다)
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


def annotate_raw(raw):
    """센서 행을 KEY·TIMESTAMP 순으로 정렬하고 챔버 그룹(GROUP 1|4), 그룹 안 단계(STEP 0~2), 가압 여부(ACTIVE)를 붙인다."""
    raw = raw.sort_values(KEY + ["TIMESTAMP"])
    group = raw.groupby(KEY)["CHAMBER"].transform("min")
    raw = raw.assign(GROUP=np.where(group <= 3, 1, 4))
    raw = raw.assign(STEP=(raw["CHAMBER"] - raw["GROUP"]).clip(0, 2).astype(int))
    return raw.assign(ACTIVE=(raw["PRESSURIZED_CHAMBER_PRESSURE"] > 0) & (raw["SLURRY_FLOW_LINE_C"] > 0))


def _annotated(raw):
    return raw if "STEP" in raw.columns and "ACTIVE" in raw.columns else annotate_raw(raw)


def load_raw_all(raw_dir=RAW):
    """세 split 의 센서 행 전체 (split 컬럼 + annotate_raw 컬럼). 정적 특징·시퀀스 텐서의 입력."""
    return pd.concat([annotate_raw(load_split(s, raw_dir)).assign(split=s) for s in ["train", "test", "val"]],
                     ignore_index=True)


def wafer_features(raw):
    """센서 행을 웨이퍼·스테이지 단위 특징으로 요약한다."""
    raw = annotate_raw(raw)
    dt = raw.groupby(KEY)["TIMESTAMP"].diff().fillna(0).clip(0, 5)

    # Preston: 압력 × 회전 속도를 시간으로 적분한 값 (행 간격 약 1초)
    for p in ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE"]:
        for r in ["WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION"]:
            raw[f"PV_{p[:12]}_{r[:5]}"] = raw[p] * raw[r] * dt
    pv_cols = [c for c in raw.columns if c.startswith("PV_")]

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


# ==============================================================================================
# 시퀀스 모양 특징 (TS_*): 원본 1 Hz 센서 행에서 만드는 정적 특징. 정답은 읽지 않는다.
# ==============================================================================================
SHAPE_CHANNELS = PRESSURE_COLS + FLOW_COLS + ROTATION_COLS  # 12 채널
CHANNEL_SHORT = {
    "PRESSURIZED_CHAMBER_PRESSURE": "pc", "MAIN_OUTER_AIR_BAG_PRESSURE": "mo", "CENTER_AIR_BAG_PRESSURE": "ce",
    "RETAINER_RING_PRESSURE": "rr", "RIPPLE_AIR_BAG_PRESSURE": "ri", "EDGE_AIR_BAG_PRESSURE": "ed",
    "SLURRY_FLOW_LINE_A": "sa", "SLURRY_FLOW_LINE_B": "sb", "SLURRY_FLOW_LINE_C": "sc",
    "WAFER_ROTATION": "wr", "STAGE_ROTATION": "sr", "HEAD_ROTATION": "hr",
}
# 특징 가족 (컬럼 접미사). 최종 모델은 TS_FAMILIES 만 쓴다: level(수준·분위수)은 기존 단계별 평균과 겹치고
# 레시피 지문(k-means·PCA)은 교차검증을 나쁘게 해 뺐다. 가지치기는 모든 가족 위에서 하고 그 뒤 가족을 고른다.
SHAPE_FAMILIES = {
    "timing": ("_step_sec", "_n_rows", "_dt_mean", "_active_frac", "_pre_active_sec", "_post_active_sec",
               "_active_span_sec", "_active_gaps", "_dw_frac", "_dw_active_frac", "_dw_toggles"),
    "ramp": ("_ramp_up", "_ramp_down", "_hi_sec", "_hi_frac", "_t_stable", "_t_on", "_t_off"),
    "level": ("_plateau", "_pstd", "_q05", "_q25", "_q75", "_q95", "_iqr", "_first", "_last", "_max",
              "_min_active", "_zero_frac", "_integral"),
    "drift": ("_slope", "_half_diff", "_n_jumps", "_abs_diff_mean"),
    "levels": ("_n_levels", "_n_changes", "_longest_run", "_n_levels_active"),
    "lag": ("_lag_on", "_lag_off"),
}
TS_FAMILIES = ("timing", "ramp", "drift", "levels", "lag")


def _runs(x):
    """값이 바뀌는 지점으로 나눈 런 길이 배열."""
    if len(x) == 0:
        return np.zeros(0, int)
    change = np.flatnonzero(np.diff(x) != 0) + 1
    return np.diff(np.concatenate([[0], change, [len(x)]]))


def _channel_stats(t, v, a, dt, prefix, out):
    """채널 하나의 모양 통계를 out(dict)에 채운다. t: 단계 시작 기준 초, v: 값, a: ACTIVE, dt: 행 간격."""
    n = len(v)
    va = v[a] if a.any() else v
    ta = t[a] if a.any() else t
    plateau = float(np.median(va))
    out[f"{prefix}_plateau"] = plateau
    out[f"{prefix}_pstd"] = float(va.std())
    q = np.quantile(va, [0.05, 0.25, 0.75, 0.95])
    out[f"{prefix}_q05"], out[f"{prefix}_q25"], out[f"{prefix}_q75"], out[f"{prefix}_q95"] = map(float, q)
    out[f"{prefix}_iqr"] = float(q[2] - q[1])
    out[f"{prefix}_first"] = float(va[0])
    out[f"{prefix}_last"] = float(va[-1])
    out[f"{prefix}_max"] = float(v.max())
    out[f"{prefix}_min_active"] = float(va.min())
    out[f"{prefix}_zero_frac"] = float((v <= 0).mean())
    out[f"{prefix}_integral"] = float((v * dt).sum())
    # 램프업/램프다운: 단계 시작 → 정체 수준의 90% 도달, 마지막 90% 이상 → 단계 끝
    if plateau > 0:
        hi = v >= 0.9 * plateau
        if hi.any():
            i0, i1 = np.flatnonzero(hi)[[0, -1]]
            out[f"{prefix}_ramp_up"] = float(t[i0])
            out[f"{prefix}_ramp_down"] = float(t[-1] - t[i1])
            out[f"{prefix}_hi_sec"] = float(t[i1] - t[i0])
            out[f"{prefix}_hi_frac"] = float(hi.mean())
        # 안정화: 정체 수준 ±5% 밴드에 처음 들어간 시각
        band = np.abs(v - plateau) <= 0.05 * abs(plateau)
        if band.any():
            out[f"{prefix}_t_stable"] = float(t[np.flatnonzero(band)[0]])
        # 시작·종료(정체 수준의 절반 넘는 첫/마지막 시각) → 채널 간 지연 계산에 씀
        on = v >= 0.5 * plateau
        if on.any():
            j0, j1 = np.flatnonzero(on)[[0, -1]]
            out[f"{prefix}_t_on"] = float(t[j0])
            out[f"{prefix}_t_off"] = float(t[j1])
    # 정체 구간 기울기 (드리프트, 단계 전체 변화량)
    if len(va) >= 3 and ta[-1] > ta[0]:
        tc = ta - ta.mean()
        slope = float((tc * (va - va.mean())).sum() / max((tc ** 2).sum(), 1e-9))
        out[f"{prefix}_slope"] = slope * (ta[-1] - ta[0])
        h = len(va) // 2
        out[f"{prefix}_half_diff"] = float(va[h:].mean() - va[:h].mean())
    # 수준 변화 횟수: 정체 수준의 2% 해상도로 반올림한 런렝스 (길이 3 이상인 런만 레시피 하위 단계로 센다)
    res = max(abs(plateau), 1e-6) * 0.02
    r = _runs(np.round(v / res))
    out[f"{prefix}_n_levels"] = int((r >= 3).sum())
    out[f"{prefix}_n_changes"] = int(len(r) - 1)
    out[f"{prefix}_longest_run"] = float(r.max() / n) if len(r) else np.nan
    ra = _runs(np.round(va / res))
    out[f"{prefix}_n_levels_active"] = int((ra >= 3).sum())
    # 활성 구간 안 큰 변화 횟수(정체 수준의 10% 넘게 점프)
    if len(va) >= 2:
        out[f"{prefix}_n_jumps"] = int((np.abs(np.diff(va)) > 0.1 * max(abs(plateau), 1e-6)).sum())
        out[f"{prefix}_abs_diff_mean"] = float(np.abs(np.diff(va)).mean())


def shape_features(raw):
    """(WAFER_ID, STAGE) 별 단계 0,1,2 채널 모양 특징. raw 는 annotate_raw 를 거친 센서 행 (정답 컬럼 없음)."""
    raw = _annotated(raw)
    cols = ["TIMESTAMP", "ACTIVE", "DRESSING_WATER_STATUS"] + SHAPE_CHANNELS
    rows = {}
    for (w, s, step), g in raw.groupby(KEY + ["STEP"], sort=False):
        arr = g[cols].to_numpy(dtype=float)
        t = arr[:, 0] - arr[0, 0]
        a = arr[:, 1] > 0
        dt = np.diff(arr[:, 0], prepend=arr[0, 0]).clip(0, 5)
        out = rows.setdefault((w, s), {})
        p = f"s{step}"
        n = len(t)
        out[f"{p}_step_sec"] = float(t[-1])
        out[f"{p}_n_rows"] = n
        out[f"{p}_dt_mean"] = float(dt[1:].mean()) if n > 1 else np.nan
        out[f"{p}_active_frac"] = float(a.mean())
        if a.any():
            ia = np.flatnonzero(a)
            out[f"{p}_pre_active_sec"] = float(t[ia[0]])
            out[f"{p}_post_active_sec"] = float(t[-1] - t[ia[-1]])
            out[f"{p}_active_span_sec"] = float(t[ia[-1]] - t[ia[0]])
            out[f"{p}_active_gaps"] = int(len(_runs(a.astype(int))) // 2)  # 활성 구간이 끊긴 횟수
        dw = arr[:, 2]
        out[f"{p}_dw_frac"] = float((dw > 0).mean())
        out[f"{p}_dw_active_frac"] = float((dw[a] > 0).mean()) if a.any() else np.nan
        out[f"{p}_dw_toggles"] = int((np.diff(dw) != 0).sum())
        for j, c in enumerate(SHAPE_CHANNELS):
            _channel_stats(t, arr[:, 3 + j], a, dt, f"{p}_{CHANNEL_SHORT[c]}", out)
        # 채널 간 지연: 가압 챔버 압력 시작·종료 기준
        pc_on, pc_off = out.get(f"{p}_pc_t_on"), out.get(f"{p}_pc_t_off")
        for c in SHAPE_CHANNELS[1:]:
            k = CHANNEL_SHORT[c]
            if pc_on is not None and f"{p}_{k}_t_on" in out:
                out[f"{p}_{k}_lag_on"] = out[f"{p}_{k}_t_on"] - pc_on
            if pc_off is not None and f"{p}_{k}_t_off" in out:
                out[f"{p}_{k}_lag_off"] = out[f"{p}_{k}_t_off"] - pc_off
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=KEY)
    return df.reset_index()


def shape_family(col):
    for fam, sufs in SHAPE_FAMILIES.items():
        if col.endswith(sufs):
            return fam
    return None


def align_keys(table, df):
    """(WAFER_ID, STAGE) 키가 있는 DataFrame 을 table 행 순서(index)에 맞춘다."""
    m = table[KEY].merge(df, on=KEY, how="left")
    m.index = table.index
    return m.drop(columns=KEY)


def prune_columns(df, base=None, min_notna=0.05, max_corr=0.98):
    """정답 없이 가지치기: 결측 비율·상수·기존 표(base)나 앞선 컬럼과 |상관| > max_corr 인 컬럼을 뺀 이름 목록."""
    keep = []
    X = df.copy()
    ok = X.columns[(X.notna().mean() >= min_notna)]
    X = X[ok]
    X = X.loc[:, X.std(ddof=0) > 1e-9]
    # 상관 (결측은 열 평균으로 채워 계산)
    Z = X.fillna(X.mean())
    Z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-12)
    ref = None
    if base is not None:
        B = base.select_dtypes(include=[np.number])
        B = B.loc[:, B.std(ddof=0) > 1e-9]
        B = B.fillna(B.mean())
        ref = ((B - B.mean()) / (B.std(ddof=0) + 1e-12)).to_numpy()
    Zn = Z.to_numpy()
    n = len(Zn)
    if ref is not None:
        c_base = np.abs(Zn.T @ ref) / n  # [p, b]
        dup_base = (c_base > max_corr).any(axis=1)
    else:
        dup_base = np.zeros(Zn.shape[1], bool)
    C = np.abs(Zn.T @ Zn) / n
    for j, col in enumerate(X.columns):
        if dup_base[j]:
            continue
        if any(C[j, i] > max_corr for i in keep):
            continue
        keep.append(j)
    return list(X.columns[keep])


def tsshape_features(table, raw_dir=RAW, cache_dir=CACHE_DIR, raw=None, families=TS_FAMILIES):
    """table 행 순서의 TS_* 정적 특징 (최종 418 컬럼). 원본 센서 행에서 계산해 cache_dir/tsshape.parquet 에 캐시한다.

    정답은 읽지 않는다: 가지치기의 기준 표는 EXCLUDE(정답·식별자·시각)를 뺀 표 컬럼이고, 캐시는 KEY 로만 맞춘다.
    raw 를 주면 캐시를 쓰지 않고 그 센서 행에서 계산한다 (테스트용).
    """
    if raw is not None:
        df = shape_features(_annotated(raw))
    else:
        cache = Path(cache_dir) / "tsshape.parquet" if cache_dir is not None else None
        if cache is not None and cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = shape_features(load_raw_all(raw_dir))
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache)
    aligned = align_keys(table, df)
    base = table[[c for c in table.columns if c not in EXCLUDE]]
    names = [c for c in prune_columns(aligned, base) if shape_family(c) in families]
    out = aligned[names].astype(float)
    out.columns = [f"TS_{c}" for c in names]
    return out


# ==============================================================================================
# 이웃 확장 특징 (NB_E_/NB_D_/NB_O_): 참조 웨이퍼 정답에서 만든다 (교차검증에서는 학습 폴드 안에서만).
# ==============================================================================================
EWM_HOURS = (0.5, 2.0, 8.0)      # 지수 가중 이동평균 반감기 (시간)
EWM_COUNT = (2, 5, 15)           # 반감기 (웨이퍼 수)
DELTA_COLS = {"ACTIVE_SEC": "ACTIVE_SEC", "PV_PRESSURIZED__HEAD__sum": "PV", "SLURRY_FLOW_LINE_C_s0_mean": "SLURRY",
              "MAIN_OUTER_AIR_BAG_PRESSURE_s0_mean": "AIRBAG", "USAGE_OF_DRESSER_start": "DRESSER",
              "USAGE_OF_POLISHING_TABLE_start": "PAD"}
MIN_W = 1e-3                     # 가중 합이 이보다 작으면 결측
INNER_FOLDS, INNER_SEED = 5, 12345


def _wmean(w, v):
    sw = w.sum(axis=1)
    m = (w @ v) / np.maximum(sw, 1e-12)
    m[sw < MIN_W] = np.nan
    return m


def _ext_compute(idx_all, cand_mask, families, prefix, cols, n, ctx):
    """idx_all 행들의 특징을 cand_mask 참조 행들로 계산해 cols 에 채운다 (자기 자신은 뺀다)."""
    reg, dev, hours, state = ctx
    regs = list(np.unique(reg))

    def put(name, idx, val):
        cols.setdefault(prefix + name, np.full(n, np.nan))[idx] = val

    for r in regs:
        idx = idx_all[reg[idx_all] == r]
        if len(idx) == 0:
            continue
        sources = {"same": [r], "cross": [q for q in regs if q != r and q[0] == r[0]]}
        for sname, sregs in sources.items():
            if not sregs:
                continue
            cand = np.flatnonzero(cand_mask & np.isin(reg, sregs))
            if len(cand) < 5:
                continue
            dt = hours[idx][:, None] - hours[cand][None, :]   # > 0 이면 참조가 과거
            notself = idx[:, None] != cand[None, :]           # 자기 자신 제외
            dv = dev[cand]
            past, fut = (dt > 0) & notself, (dt < 0) & notself
            if "E" in families:
                # 과거·미래 참조 편차의 지수 가중 이동평균과 그 차(국소 기울기): 시간 반감기, 웨이퍼 수 반감기
                adt = np.minimum(np.abs(dt), 1e4)
                for hl in EWM_HOURS:
                    p = _wmean(np.where(past, 0.5 ** (adt / hl), 0.0), dv)
                    f = _wmean(np.where(fut, 0.5 ** (adt / hl), 0.0), dv)
                    put(f"E_{sname}_h{hl:g}_past", idx, p)
                    put(f"E_{sname}_h{hl:g}_fut", idx, f)
                    put(f"E_{sname}_h{hl:g}_slope", idx, f - p)
                rank_p = np.argsort(np.argsort(np.where(past, dt, np.inf), axis=1), axis=1)
                rank_f = np.argsort(np.argsort(np.where(fut, -dt, np.inf), axis=1), axis=1)
                for hl in EWM_COUNT:
                    p = _wmean(np.where(past, 2.0 ** (-rank_p / hl), 0.0), dv)
                    f = _wmean(np.where(fut, 2.0 ** (-rank_f / hl), 0.0), dv)
                    put(f"E_{sname}_c{hl}_past", idx, p)
                    put(f"E_{sname}_c{hl}_fut", idx, f)
                    put(f"E_{sname}_c{hl}_slope", idx, f - p)
            if "D" in families and sname == "same":
                # 시간상 바로 앞·뒤 참조 웨이퍼 대비 이 웨이퍼의 공정 상태 변화량 ("마지막 측정 이후 무엇이 달라졌나")
                for dname, mask, sign in (("prev", past, 1.0), ("next", fut, -1.0)):
                    dd = np.where(mask, sign * dt, np.inf)
                    j = dd.argmin(axis=1)
                    ok = np.isfinite(dd[np.arange(len(idx)), j])
                    delta = state[idx] - state[cand[j]]
                    delta[~ok] = np.nan
                    for c, short in enumerate(DELTA_COLS.values()):
                        put(f"D_{dname}_{short}", idx, delta[:, c])


def neighbor_ext_features(table, ref_mask, families=("E", "D", "O"), prefix="NB_", inner_folds=INNER_FOLDS):
    """참조(학습) 웨이퍼 정답으로 만드는 이웃 확장 특징 50개 (레짐 평균 대비 편차 dev 기준).

    E: 과거·미래 참조 편차의 지수 가중 이동평균(같은 레짐 / 같은 그룹 다른 스테이지), D: 직전·직후 참조 웨이퍼 대비
    상태 변화량, O: 같은 웨이퍼 다른 스테이지의 편차와 시간 간격.
    누수 방지: 참조 행을 leave-one-out 만으로 처리하면 넓은 창에서 이웃 집합이 거의 같아 평균이 자기 정답의 단조
    함수가 된다. 그래서 참조 행은 inner K-fold 로 나눠 자기 폴드 밖의 참조 행만으로 계산하고, 참조가 아닌 행은
    참조 전체로 계산한다. 자기 자신은 항상 뺀다.
    """
    ref_mask = np.asarray(ref_mask, bool)
    families = set(families)
    reg = regime(table)
    y = table[TARGET].to_numpy(float)
    mu = {r: y[ref_mask & (reg == r)].mean() for r in np.unique(reg)}
    dev = y - np.array([mu[r] for r in reg])
    hours = table["T_START"].to_numpy(float) / 3600.0
    ctx = (reg, dev, hours, table[list(DELTA_COLS)].to_numpy(float))
    n, cols = len(table), {}
    _ext_compute(np.flatnonzero(~ref_mask), ref_mask, families, prefix, cols, n, ctx)
    if inner_folds and inner_folds > 1:
        inner = np.random.RandomState(INNER_SEED).randint(0, inner_folds, n)
        for f in range(inner_folds):
            own = ref_mask & (inner == f)
            _ext_compute(np.flatnonzero(own), ref_mask & ~own, families, prefix, cols, n, ctx)
    else:
        _ext_compute(np.flatnonzero(ref_mask), ref_mask, families, prefix, cols, n, ctx)
    out = pd.DataFrame(cols, index=table.index)
    if "O" in families:
        # 같은 웨이퍼 다른 스테이지(참조 집합에 있을 때)의 편차와 시간 간격 (다른 행이므로 자기 정답과 무관)
        ref = table.loc[ref_mask, ["WAFER_ID", "STAGE", "T_START"]].assign(dev=dev[ref_mask])
        other = table[["WAFER_ID", "STAGE", "T_START"]].assign(STAGE=table["STAGE"].map({"A": "B", "B": "A"}))
        m = other.merge(ref, on=["WAFER_ID", "STAGE"], how="left", suffixes=("", "_o"))
        out[prefix + "O_dev"] = m["dev"].to_numpy()
        out[prefix + "O_gap"] = ((m["T_START_o"] - m["T_START"]) / 3600.0).to_numpy()
    return out


# ==============================================================================================
# 시퀀스 텐서: 단계별 센서 궤적을 고정 길이로 보간한 배열 (융합 신경망 입력, 정답 미사용)
# ==============================================================================================
SEQ_POINTS = 64


def sequences(raw, n_points=SEQ_POINTS, cols=PROCESS_COLS, active_only=False):
    """웨이퍼·스테이지마다 챔버 단계(STEP 0,1,2)별 센서 구간을 n_points 길이로 선형 보간한다.

    반환: keys(DataFrame WAFER_ID, STAGE), X(float32 [n, 3, n_points, len(cols)], 없는 단계는 NaN),
    step_len(float32 [n, 3], 단계별 행 수; 없는 단계는 0). active_only=False: 가압 구간만이 아니라 기록 전체
    (챔버 그룹 1 웨이퍼의 절반은 가압 행이 없다).
    """
    r = raw[raw["ACTIVE"]] if active_only else raw
    keys = raw[KEY].drop_duplicates().reset_index(drop=True)
    pos = {tuple(k): i for i, k in enumerate(keys.itertuples(index=False))}
    X = np.full((len(keys), 3, n_points, len(cols)), np.nan, dtype=np.float32)
    step_len = np.zeros((len(keys), 3), dtype=np.float32)
    grid = np.linspace(0, 1, n_points)
    for (w, s, step), g in r.groupby(KEY + ["STEP"]):
        step_len[pos[(w, s)], step] = len(g)
        v = g[cols].to_numpy(dtype=float)
        if len(v) < 2:
            continue
        src = np.linspace(0, 1, len(v))
        X[pos[(w, s)], step] = np.column_stack([np.interp(grid, src, v[:, j]) for j in range(v.shape[1])])
    return keys, X, step_len


def sequence_tensor(table, raw_dir=RAW, cache_dir=CACHE_DIR, raw=None, n_points=SEQ_POINTS):
    """table 행 순서의 시퀀스 텐서 (seq float32 [n, 3, 14, n_points], present bool [n, 3]).

    채널 = PROCESS_COLS 13 + 경과 행 수(정규화 격자 × 단계 행 수; 보간으로 사라지는 시간 정보 복원).
    없는 단계는 NaN. cache_dir/seq_{n_points}_all.npz 에 캐시한다. table 의 index 는 RangeIndex 여야 한다
    (모델이 X.index 로 행을 찾는다).
    """
    assert (table.index.to_numpy() == np.arange(len(table))).all(), "table 의 index 가 RangeIndex 가 아닙니다"
    cache = Path(cache_dir) / f"seq_{n_points}_all.npz" if (cache_dir is not None and raw is None) else None
    if cache is not None and cache.exists():
        d = np.load(cache)
        keys, X, step_len = pd.DataFrame({"WAFER_ID": d["wafer"], "STAGE": d["stage"]}), d["X"], d["step_len"]
    else:
        keys, X, step_len = sequences(load_raw_all(raw_dir) if raw is None else _annotated(raw), n_points, active_only=False)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, wafer=keys["WAFER_ID"].to_numpy(dtype=np.int64),
                                stage=np.array(list(keys["STAGE"]), dtype="U4"), X=X, step_len=step_len)
    pos = table[KEY].merge(keys.reset_index().rename(columns={"index": "pos"}), on=KEY, how="left")["pos"].to_numpy()
    assert not np.isnan(pos.astype(float)).any(), "시퀀스가 없는 웨이퍼가 있습니다"
    pos = pos.astype(int)
    X, step_len = X[pos], step_len[pos]  # [n, 3, L, 13], [n, 3]
    L = X.shape[2]
    elapsed = np.linspace(0, 1, L, dtype=np.float32)[None, None, :, None] * step_len[:, :, None, None]
    X = np.concatenate([X, elapsed], axis=3)
    present = ~np.isnan(X[:, :, 0, 0])
    X[~present] = np.nan
    return np.ascontiguousarray(X.transpose(0, 1, 3, 2)), present  # [n, 3, C, L]
