"""물리(트라이볼로지·Preston·CMP 공정) 지식 기반 정적 특징 실험.

아이디어: 표와 센서 원본 행에서 정답을 쓰지 않는 물리 특징을 만들어 static_fn 으로 붙인다.
특징 묶음(접두사로 구분, 가족별 절제 실험용)
- PH_V_ 상대속도: |HEAD-STAGE|, HEAD+STAGE, WAFER/HEAD 비 등을 각 존 압력과 곱해 활성 시간으로 적분(Preston, 단계별)
- PH_P_ 압력 분포: 존/챔버 압력 비, 존 불균일도, 리테이너/챔버 비, 압력 가중 시간
- PH_S_ 슬러리: 라인별 부피(유량×dt), A/B/C 비율, 단위 P·V·시간당 슬러리, Sommerfeld 유사수 V·Q/P, 드레싱 워터 비율
- PH_C_ 컨디셔닝·소모품: 웨이퍼 중 드레서 사용량 증가, 수명 위치·초기 길들이기, 교체 후 경과 웨이퍼 수·시간, 마모율, 교호작용
- PH_T_ 스케줄·열: 같은 챔버 그룹 직전 웨이퍼와의 유휴 시간, 처리량, 로트 내 위치, 시각
- PH_K_ 단계 구조: 활성 단계 수, 챔버 패턴, 단계별 시간 비율, 레시피 중앙값 대비 연마 시간, 회전 중 비율

결과 (5-fold CV 앙상블, fold_seed=0, 기준선 7.596 / 테스트 6.969 / 검증 6.850)
- 최종(여섯 가족 전부, 새 특징 187개): CV 7.144 (Δ-0.452), 테스트 6.660, 검증 6.540. fold_seed=1 에서 CV 6.868 (기준선 범위 7.51~7.67).
  이득은 전부 lgb 멤버(7.718→7.241)에서 온다. 나머지 멤버는 compact 컬럼만 써서 새 특징을 보지 않는다.
- 가족 하나씩만 추가한 절제: S(슬러리) 7.299, K(단계 구조) 7.356, T(스케줄) 7.424, V(상대속도) 7.468,
  C(컨디셔닝) 7.545, P(압력 분포) 7.600. 가족들은 서로 보완적이라 전부 넣은 것이 가장 좋다.
- 레짐별 LightGBM gain 상위 새 특징: PH_S_B_vol_s0, PH_S_C_vol_s0 (dt 가중 단계 0 슬러리 부피 = 정밀한 연마 시간),
  PH_T_prev_active_sec, PH_K_head_mean_all, PH_P_PC_int_s0 (압력×시간 도즈), PH_S_dw_frac_inact, PH_T_gap_prev(_end),
  PH_T_hod_cos/sin, PH_K_lead_sec, PH_C_dresser_x_pad, PH_C_pad_hours_since. 새 특징의 gain 비중은 1A 26%, 4A 11%, 4B 15%.

실행: python cmp2016/exp/physics.py          # 최종 변형만 (혼자 실행 시 약 4분), physics.json 저장
      python cmp2016/exp/physics.py --all    # 탐색 변형들 (전부 + 가족별 하나씩)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 다른 실험과 CPU 를 나눌 때 OpenMP 스핀 대기 낭비를 줄인다

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, get_raw, get_table, run, summary  # noqa: E402
from cmp_data import FLOW_COLS, KEY, PRESSURE_COLS, USAGE_COLS  # noqa: E402

CACHE = HERE / "cache_physics.parquet"
FAMILIES = ("PH_V_", "PH_P_", "PH_S_", "PH_C_", "PH_T_", "PH_K_")
ZONES = ["MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "EDGE_AIR_BAG_PRESSURE"]
STEP_MEDIAN_SEC = {0: 127.0, 1: 78.0, 2: 46.0}  # 레시피 단계별 활성 시간 중앙값


def _unstack(df, name):
    """(KEY, STEP) 인덱스 시리즈를 KEY 인덱스의 _s0/_s1/_s2 컬럼으로 편다."""
    u = df.unstack("STEP")
    u.columns = [f"{name}_s{s}" for s in u.columns]
    for s in range(3):
        if f"{name}_s{s}" not in u.columns:
            u[f"{name}_s{s}"] = np.nan
    return u[[f"{name}_s{s}" for s in range(3)]]


def raw_features(raw):
    """센서 행에서 만드는 가족(V, P, S, K 일부). KEY(WAFER_ID, STAGE) 인덱스."""
    r = raw.sort_values(KEY + ["TIMESTAMP"]).reset_index(drop=True)
    dt = r.groupby(KEY)["TIMESTAMP"].diff().fillna(0).clip(0, 5).to_numpy()
    r["dt"] = dt
    act = r["ACTIVE"].to_numpy()
    head, stage, wafer = (r[c].to_numpy(dtype=float) for c in ["HEAD_ROTATION", "STAGE_ROTATION", "WAFER_ROTATION"])
    pc = r["PRESSURIZED_CHAMBER_PRESSURE"].to_numpy(dtype=float)
    r["V_REL"] = np.abs(head - stage)
    r["V_SUM"] = head + stage
    r["V_WH"] = wafer / np.maximum(head, 1.0)
    r["V_WS"] = wafer / np.maximum(stage, 1.0)
    r["ROT_ON"] = (stage > 0).astype(float)
    r["WROT_ON"] = (wafer > 0).astype(float)
    r["Q_TOT"] = r[FLOW_COLS].sum(axis=1)
    r["P_ALL"] = r[PRESSURE_COLS].sum(axis=1)
    a = r[act].copy()
    ga, gs = a.groupby(KEY), a.groupby(KEY + ["STEP"])
    out = {}

    # --- PH_V_ 상대속도 · Preston 적분 ---
    for v in ["V_REL", "V_SUM", "V_WH", "V_WS", "ROT_ON", "WROT_ON"]:
        out[f"PH_V_{v}_mean"] = ga[v].mean()
    for v in ["V_REL", "V_SUM", "ROT_ON"]:
        out.update(_unstack(gs[v].mean(), f"PH_V_{v}").to_dict("series"))
    for v in ["V_REL", "V_SUM"]:
        a[f"{v}_dt"] = a[v] * a["dt"]
        for p in PRESSURE_COLS:
            a[f"PV_{p}_{v}"] = a[p] * a[v] * a["dt"]
            out[f"PH_V_PV_{p[:10]}_{v[2:]}"] = a.groupby(KEY)[f"PV_{p}_{v}"].sum()
        out[f"PH_V_{v}_int"] = a.groupby(KEY)[f"{v}_dt"].sum()
        for p in ["PRESSURIZED_CHAMBER_PRESSURE", "MAIN_OUTER_AIR_BAG_PRESSURE"]:
            out.update(_unstack(a.groupby(KEY + ["STEP"])[f"PV_{p}_{v}"].sum(), f"PH_V_PV_{p[:10]}_{v[2:]}").to_dict("series"))
    # 회전 중(실제 연마) 구간만의 Preston 적분
    a["PV_ROT"] = a["PRESSURIZED_CHAMBER_PRESSURE"] * a["V_SUM"] * a["dt"] * a["ROT_ON"]
    out["PH_V_PV_rot_int"] = a.groupby(KEY)["PV_ROT"].sum()
    out["PH_V_rot_sec"] = (a["ROT_ON"] * a["dt"]).groupby([a[k] for k in KEY]).sum()

    # --- PH_P_ 압력 분포 ---
    for z in ZONES + ["RETAINER_RING_PRESSURE"]:
        ratio = a[z] / np.maximum(a["PRESSURIZED_CHAMBER_PRESSURE"], 1e-3)
        out[f"PH_P_{z[:8]}_ratio"] = ratio.groupby([a[k] for k in KEY]).mean()
        out.update(_unstack(ratio.groupby([a[k] for k in KEY + ["STEP"]]).mean(), f"PH_P_{z[:8]}_ratio").to_dict("series"))
    zone = a[ZONES].to_numpy(dtype=float)
    a["ZONE_CV"] = zone.std(axis=1) / np.maximum(zone.mean(axis=1), 1e-3)
    a["EDGE_CENTER"] = a["EDGE_AIR_BAG_PRESSURE"] / np.maximum(a["CENTER_AIR_BAG_PRESSURE"], 1e-3)
    a["RET_ZONE"] = a["RETAINER_RING_PRESSURE"] / np.maximum(zone.mean(axis=1), 1e-3)
    a["P_ALL_dt"] = a["P_ALL"] * a["dt"]
    a["PC_dt"] = a["PRESSURIZED_CHAMBER_PRESSURE"] * a["dt"]
    for c in ["ZONE_CV", "EDGE_CENTER", "RET_ZONE"]:
        out[f"PH_P_{c}"] = a.groupby(KEY)[c].mean()
        out.update(_unstack(a.groupby(KEY + ["STEP"])[c].mean(), f"PH_P_{c}").to_dict("series"))
    out["PH_P_P_ALL_int"] = a.groupby(KEY)["P_ALL_dt"].sum()
    out["PH_P_PC_int"] = a.groupby(KEY)["PC_dt"].sum()
    out.update(_unstack(a.groupby(KEY + ["STEP"])["PC_dt"].sum(), "PH_P_PC_int").to_dict("series"))
    out["PH_P_P_ALL_mean"] = a.groupby(KEY)["P_ALL"].mean()
    out["PH_P_PC_cv"] = a.groupby(KEY)["PRESSURIZED_CHAMBER_PRESSURE"].std() / np.maximum(a.groupby(KEY)["PRESSURIZED_CHAMBER_PRESSURE"].mean(), 1e-3)

    # --- PH_S_ 슬러리 ---
    g_all = r.groupby(KEY)
    for f in FLOW_COLS:
        a[f"{f}_vol"] = a[f] * a["dt"]
        r[f"{f}_vol"] = r[f] * r["dt"]
        out[f"PH_S_{f[-1]}_vol"] = a.groupby(KEY)[f"{f}_vol"].sum()
        out[f"PH_S_{f[-1]}_vol_all"] = g_all[f"{f}_vol"].sum()
        out.update(_unstack(a.groupby(KEY + ["STEP"])[f"{f}_vol"].sum(), f"PH_S_{f[-1]}_vol").to_dict("series"))
    tot = sum(out[f"PH_S_{f[-1]}_vol"] for f in FLOW_COLS)
    out["PH_S_tot_vol"] = tot
    for f in FLOW_COLS:
        out[f"PH_S_{f[-1]}_frac"] = out[f"PH_S_{f[-1]}_vol"] / tot.replace(0, np.nan)
    out["PH_S_vol_per_sec"] = tot / ga["dt"].sum().replace(0, np.nan)
    a["PV_HEAD"] = a["PRESSURIZED_CHAMBER_PRESSURE"] * a["HEAD_ROTATION"] * a["dt"]
    out["PH_S_vol_per_PV"] = tot / a.groupby(KEY)["PV_HEAD"].sum().replace(0, np.nan) * 1e4
    a["SOMM"] = a["V_SUM"] * a["Q_TOT"] / np.maximum(a["PRESSURIZED_CHAMBER_PRESSURE"], 1e-3)
    out["PH_S_sommerfeld"] = a.groupby(KEY)["SOMM"].mean()
    out.update(_unstack(a.groupby(KEY + ["STEP"])["SOMM"].mean(), "PH_S_sommerfeld").to_dict("series"))
    out["PH_S_A_pre_vol"] = r.loc[~act].groupby(KEY)["SLURRY_FLOW_LINE_A_vol"].sum()  # 비가압 중 슬러리(프리웻·린스)
    out["PH_S_C_pre_vol"] = r.loc[~act].groupby(KEY)["SLURRY_FLOW_LINE_C_vol"].sum()
    out["PH_S_dw_frac_all"] = g_all["DRESSING_WATER_STATUS"].mean()
    out["PH_S_dw_frac_act"] = ga["DRESSING_WATER_STATUS"].mean()
    out["PH_S_dw_sec_all"] = (r["DRESSING_WATER_STATUS"] * r["dt"]).groupby([r[k] for k in KEY]).sum()
    out["PH_S_dw_frac_inact"] = r.loc[~act].groupby(KEY)["DRESSING_WATER_STATUS"].mean()

    # --- PH_K_ 단계 구조(센서 행 기반) ---
    t0 = g_all["TIMESTAMP"].min()
    out["PH_K_rows_all"] = g_all.size()
    out.update(_unstack(r.groupby(KEY + ["STEP"]).size(), "PH_K_rows").to_dict("series"))
    out["PH_K_lead_sec"] = ga["TIMESTAMP"].min() - t0
    out["PH_K_tail_sec"] = g_all["TIMESTAMP"].max() - ga["TIMESTAMP"].max()
    out["PH_K_press_nosl_rows"] = pd.Series((pc > 0) & ~act, index=r.index).groupby([r[k] for k in KEY]).sum()
    out["PH_K_head_mean_all"] = g_all["HEAD_ROTATION"].mean()
    out["PH_K_rot_frac_all"] = (r["STAGE_ROTATION"] > 0).groupby([r[k] for k in KEY]).mean()
    out["PH_K_gap_max"] = g_all["TIMESTAMP"].diff().groupby([r[k] for k in KEY]).max()  # 기록 중 최대 끊김
    return pd.DataFrame(out)


def table_features(table):
    """표에서 만드는 가족(C, T, K). 정답을 쓰지 않고 T_START·소모품 카운터는 모든 split 을 쓴다."""
    t = table
    out = pd.DataFrame(index=t.index)
    act = t["ACTIVE_SEC"].clip(lower=1)
    dr, pad, mem = t["USAGE_OF_DRESSER_start"], t["USAGE_OF_POLISHING_TABLE_start"], t["USAGE_OF_MEMBRANE_start"]

    # --- PH_C_ 컨디셔닝·소모품 ---
    out["PH_C_dresser_per_sec"] = t["USAGE_OF_DRESSER_delta"] / act
    out["PH_C_dtable_per_sec"] = t["USAGE_OF_DRESSER_TABLE_delta"] / act
    out["PH_C_pad_per_sec"] = t["USAGE_OF_POLISHING_TABLE_delta"] / act
    out["PH_C_dresser_breakin"] = (dr < 100).astype(int)
    out["PH_C_pad_breakin"] = (pad < 30).astype(int)
    out["PH_C_dresser_x_pad"] = dr * pad / 1e4
    out["PH_C_dresser_x_mem"] = dr * mem / 1e3
    out["PH_C_pad_x_mem"] = pad * mem / 1e3
    out["PH_C_dresser_over_pad"] = dr / pad.clip(lower=1)
    for c in USAGE_COLS:
        out[f"PH_C_{c[9:14]}_rate"] = t[f"{c}_delta"] / t["DURATION"].clip(lower=1)
    # 교체(카운터 초기화) 후 경과: 그룹별 시간순, 모든 split
    for c, name, drop in [("USAGE_OF_DRESSER_start", "dresser", 200), ("USAGE_OF_POLISHING_TABLE_start", "pad", 100),
                          ("USAGE_OF_MEMBRANE_start", "mem", 30)]:
        for g, idx in t.groupby("GROUP").groups.items():
            x = t.loc[idx].sort_values("T_START")
            life = (x[c].diff() < -drop).cumsum()
            since = x.groupby(life).cumcount()
            t_reset = x["T_START"] - x.groupby(life)["T_START"].transform("min")
            life_max = x.groupby(life)[c].transform("max")
            out.loc[x.index, f"PH_C_{name}_life"] = life.to_numpy()
            out.loc[x.index, f"PH_C_{name}_wafers_since"] = since.to_numpy()
            out.loc[x.index, f"PH_C_{name}_hours_since"] = t_reset.to_numpy() / 3600.0
            out.loc[x.index, f"PH_C_{name}_life_frac"] = (x[c] / life_max.clip(lower=1)).to_numpy()
            if name == "dresser":
                # 최근 10장 동안의 컨디셔닝 강도(드레서·드레서 테이블 카운터 증가)
                for cc, nn in [("USAGE_OF_DRESSER_delta", "dresser"), ("USAGE_OF_DRESSER_TABLE_delta", "dtable")]:
                    roll = x[cc].rolling(10, min_periods=1).sum()
                    out.loc[x.index, f"PH_C_{nn}_delta_roll10"] = roll.to_numpy()
                out.loc[x.index, "PH_C_dresser_rate_roll10"] = (
                    x["USAGE_OF_DRESSER_delta"].rolling(10, min_periods=1).sum()
                    / (x["T_START"].diff().rolling(10, min_periods=1).sum().clip(lower=60) / 3600.0)).to_numpy()

    # --- PH_T_ 스케줄·열 ---
    t_end = t["T_START"] + t["DURATION"]
    for g, idx in t.groupby("GROUP").groups.items():
        x = t.loc[idx].sort_values("T_START")
        ts, te = x["T_START"].to_numpy(), t_end.loc[x.index].to_numpy()
        prev_end = pd.Series(np.r_[np.nan, te[:-1]]).cummax().to_numpy()
        gap_prev = ts - np.r_[np.nan, ts[:-1]]
        gap_prev_end = ts - prev_end
        gap_next = np.r_[ts[1:], np.nan] - ts
        gap_prev2 = ts - np.r_[np.nan, np.nan, ts[:-2]]
        out.loc[x.index, "PH_T_gap_prev"] = gap_prev
        out.loc[x.index, "PH_T_gap_prev_end"] = gap_prev_end
        out.loc[x.index, "PH_T_gap_next"] = gap_next
        out.loc[x.index, "PH_T_gap_prev2"] = gap_prev2
        out.loc[x.index, "PH_T_log_gap_prev"] = np.log1p(np.maximum(gap_prev, 0))
        out.loc[x.index, "PH_T_log_gap_prev_end"] = np.log1p(np.maximum(gap_prev_end, 0))
        out.loc[x.index, "PH_T_log_gap_next"] = np.log1p(np.maximum(gap_next, 0))
        long_idle = np.nan_to_num(gap_prev_end, nan=1e9) > 1800
        out.loc[x.index, "PH_T_idle_long"] = long_idle.astype(int)
        run_id = np.cumsum(long_idle)
        s = pd.Series(np.arange(len(x)), index=x.index)
        out.loc[x.index, "PH_T_since_idle"] = s.groupby(run_id).cumcount().to_numpy()
        out.loc[x.index, "PH_T_run_len"] = s.groupby(run_id).transform("size").to_numpy()
        last_idle_t = pd.Series(np.where(long_idle, ts, np.nan)).ffill().to_numpy()
        out.loc[x.index, "PH_T_hours_since_idle"] = (ts - last_idle_t) / 3600.0
        # 로트(연속 처리, 10분 이내) 내 위치
        lot = np.cumsum(np.nan_to_num(gap_prev, nan=1e9) > 600)
        out.loc[x.index, "PH_T_lot_pos"] = s.groupby(lot).cumcount().to_numpy()
        out.loc[x.index, "PH_T_lot_len"] = s.groupby(lot).transform("size").to_numpy()
        # 처리량: 지난 1·3시간, 다음 1시간 동안 같은 그룹 웨이퍼 수
        for h in [1, 3]:
            out.loc[x.index, f"PH_T_n_prev_{h}h"] = np.searchsorted(ts, ts, side="left") - np.searchsorted(ts, ts - h * 3600, side="left")
        out.loc[x.index, "PH_T_n_next_1h"] = np.searchsorted(ts, ts + 3600, side="right") - np.searchsorted(ts, ts, side="right")
        # 직전 웨이퍼의 연마 시간(같은 그룹, 정답 아님) 과 스테이지
        out.loc[x.index, "PH_T_prev_active_sec"] = np.r_[np.nan, x["ACTIVE_SEC"].to_numpy()[:-1]]
        out.loc[x.index, "PH_T_prev_stage_B"] = np.r_[np.nan, x["STAGE_B"].to_numpy()[:-1]]
    hod = (t["T_START"] % 86400) / 86400.0 * 2 * np.pi
    out["PH_T_hod_sin"], out["PH_T_hod_cos"] = np.sin(hod), np.cos(hod)
    out["PH_T_day"] = (t["T_START"] - t["T_START"].min()) / 86400.0
    other = t[["WAFER_ID", "STAGE", "T_START"]].assign(STAGE=t["STAGE"].map({"A": "B", "B": "A"}))
    ot = other.merge(t[["WAFER_ID", "STAGE", "T_START"]], on=["WAFER_ID", "STAGE"], how="left", suffixes=("", "_o"))
    out["PH_T_other_stage_gap"] = (ot["T_START_o"].to_numpy() - t["T_START"].to_numpy()) / 3600.0

    # --- PH_K_ 단계 구조(표 기반) ---
    present = t[[f"ACTIVE_SEC_s{s}" for s in range(3)]].notna()
    out["PH_K_n_steps"] = present.sum(axis=1)
    out["PH_K_pattern"] = present.astype(int).to_numpy() @ np.array([4, 2, 1])
    for s in range(3):
        out[f"PH_K_frac_s{s}"] = t[f"ACTIVE_SEC_s{s}"] / act
        out[f"PH_K_dev_s{s}"] = t[f"ACTIVE_SEC_s{s}"] - STEP_MEDIAN_SEC[s]
    med = t["ACTIVE_SEC"].where(t["ACTIVE_SEC"] > 0).groupby([t["GROUP"], t["STAGE"]]).transform("median")  # 활성 웨이퍼 기준 레시피 중앙값
    out["PH_K_active_vs_median"] = t["ACTIVE_SEC"] - med
    out["PH_K_active_ratio_median"] = t["ACTIVE_SEC"] / med.clip(lower=1)
    out["PH_K_inactive_sec"] = t["DURATION"] - t["ACTIVE_SEC"]
    out["PH_K_active_frac"] = t["ACTIVE_SEC"] / t["DURATION"].clip(lower=1)
    out["PH_K_no_active"] = (t["ACTIVE_SEC"] == 0).astype(int)
    return out


def physics_static(table, families=FAMILIES, cache=CACHE):
    """모든 물리 특징을 만들고(parquet 캐시) 요청한 가족만 돌려준다."""
    if cache is not None and cache.exists():
        feats = pd.read_parquet(cache)
    else:
        rf = raw_features(get_raw())
        feats = table[KEY].merge(rf, left_on=KEY, right_index=True, how="left").drop(columns=KEY)
        feats.index = table.index
        feats = feats.join(table_features(table))
        feats = feats.astype(float)
        if cache is not None:
            feats.to_parquet(cache)
    keep = [c for c in feats.columns if c.startswith(tuple(families))]
    return feats[keep]


def make_static(families=FAMILIES, cols=None):
    """static_fn 팩토리: 가족 또는 명시한 컬럼 목록만 붙인다."""
    def fn(table):
        f = physics_static(table, families)
        return f[[c for c in cols if c in f.columns]] if cols is not None else f
    return fn


def lgb_gain(table, static_fn, top=30):
    """레짐마다 학습 행 전체로 LightGBM 을 한 번씩 학습해 새 특징의 gain 비중(레짐 평균)을 본다 (선택은 CV 로만 한다)."""
    from harness import base_features
    from cmp_data import TARGET, regime
    train_mask = ((table["split"] == "train") & ~table["OUTLIER"]).to_numpy()
    X = base_features(table, train_mask, "full").join(static_fn(table))
    y, reg = table[TARGET].to_numpy(), regime(table)
    tot, per_regime = pd.Series(0.0, index=X.columns), {}
    for r in np.unique(reg):
        m = train_mask & (reg == r)
        use = [c for c in X.columns if X.loc[m, c].notna().any()]
        model = BASE_MODELS["lgb"]["make"]().fit(X.loc[m, use], y[m])
        g = pd.Series(model.booster_.feature_importance("gain"), index=use)
        g = g / g.sum()
        per_regime[r] = float(g[g.index.str.startswith("PH_")].sum())
        tot = tot.add(g, fill_value=0)
    tot = (tot / len(per_regime)).sort_values(ascending=False)
    return tot[tot.index.str.startswith("PH_")].head(top), tot, per_regime


# 최종 변형: 여섯 가족 전부 (가족별 절제 결과는 모듈 docstring 참고)
FINAL_FAMILIES = FAMILIES
FINAL_COLS = None


def final_static():
    return make_static(FINAL_FAMILIES, FINAL_COLS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="탐색 변형 실행")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    table = get_table()
    if args.all:
        results = {}
        res = run(static_fn=make_static(FAMILIES), label="ph_all", fold_seed=args.seed)
        results["ph_all"] = summary(res)
        for fam in FAMILIES:
            res = run(static_fn=make_static((fam,)), label=f"ph_only_{fam}", fold_seed=args.seed)
            results[f"only_{fam}"] = summary(res)
        for k, v in results.items():
            print(k, v)
        return
    res = run(static_fn=final_static(), label="physics", fold_seed=args.seed, out=str(HERE / "physics.json"))
    print(summary(res))
    top, gain, share = lgb_gain(table, final_static())
    print("[gain] 레짐별 새 특징 gain 비중:", {k: round(v, 3) for k, v in share.items()})
    print("[gain] 가족별 gain 비중(레짐 평균):", {f: round(float(gain[gain.index.str.startswith(f)].sum()), 4) for f in FAMILIES})
    print("[gain] 상위 새 특징(레짐 평균):")
    print(top.round(4).to_string())


if __name__ == "__main__":
    main()
