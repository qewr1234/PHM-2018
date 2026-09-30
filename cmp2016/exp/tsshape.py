"""tsshape: 원시 1 Hz 센서 시퀀스의 모양(shape) 특징 (정답 미사용 static 특징).

아이디어
- 웨이퍼·스테이지의 챔버 단계(STEP 0,1,2)마다 압력 6종·슬러리 3종·회전 3종 채널의
  램프업/램프다운 시간, 안정화 시간, 정체 구간(plateau) 수준·표준편차·기울기, 분위수, 처음/끝 값,
  수준 변화 횟수(런렝스), 적분, 채널 간 시작·종료 지연(슬러리가 압력보다 먼저?), 단계 앞뒤 비활성 시간
- 레시피 지문: harness.sequences(32점)로 단계별 궤적을 표준화해 k-means(k=8) 군집 id·중심 거리·군집 크기,
  레짐 중앙 궤적과의 잔차 노름(채널별), PCA 상위 8성분
- 정답 없이 가지치기(결측 95% 이상, 상수, 기존 표·앞선 컬럼과 |상관| > 0.98 제거): 1181 → 795 컬럼
- 특징은 static_fn으로 붙으므로 cols="all"인 lgb 멤버만 본다 (compact 멤버는 그대로).

시도한 것 (5-fold CV 앙상블, fold_seed=0, 기준선 7.596)
- fp (지문만, 65)                          7.670  (+0.074, 나빠짐)
- lean (도메인 기준 27개)                   7.330  (-0.266)
- lean_member (기본 lgb 유지 + TS lgb 멤버) 7.330  (NNLS가 기본 lgb를 0으로 → 멤버 추가는 의미 없음)
- all (7가족 전부, 795)                     7.144  (-0.452)
- lite (timing·ramp·drift·levels·lag, 418) 7.135  (-0.461)  ← 최종 (level·fp 가족 제외, 절반 비용)
- nofp (fp 제외, 730)                       7.119  (-0.477; lite와 차이 0.016 < 잡음 0.05 → 더 싼 lite 유지)
- lite, fold_seed=1                         7.000  (테스트 6.703, 검증 6.378; 같은 seed 기준선 CV 범위 7.51~7.67)
lgb 멤버 CV 7.718 → 7.241. gain 상위 TS 특징은 s0_mo_hi_sec(주 에어백이 정체 수준 90% 이상인 시간),
s0_active_span_sec, 압력·슬러리 적분 등 "실제 연마 시간" 계열 (README의 가공 시간 ↔ 연마량 관계와 일치).

최종 (lite): CV 앙상블 7.135, 공식 테스트 6.673, 검증 6.438 (기준선 7.596 / 6.969 / 6.850).

결과는 cmp2016/exp/cache_tsshape.parquet 에 캐시한다 (계산 약 1.5분).
실행: python cmp2016/exp/tsshape.py            # 최종 변형만 → cmp2016/exp/tsshape.json
      python cmp2016/exp/tsshape.py --all      # 탐색 변형 전부
      python cmp2016/exp/tsshape.py --variants lite --seed 1 --outdir DIR
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 다른 실험과 CPU를 나눠 쓸 때 OpenMP 스핀 대기로 LightGBM이 수십 배 느려지는 것을 막는다 (lightgbm import 전에 설정).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, get_raw, get_table, run, sequences, summary  # noqa: E402
from cmp_data import FLOW_COLS, KEY, PRESSURE_COLS, ROTATION_COLS, regime  # noqa: E402
from run_cmp import COMPACT_PREFIX, EXCLUDE  # noqa: E402

CACHE = HERE / "cache_tsshape.parquet"
CHANNELS = PRESSURE_COLS + FLOW_COLS + ROTATION_COLS  # 12 채널
SHORT = {
    "PRESSURIZED_CHAMBER_PRESSURE": "pc", "MAIN_OUTER_AIR_BAG_PRESSURE": "mo", "CENTER_AIR_BAG_PRESSURE": "ce",
    "RETAINER_RING_PRESSURE": "rr", "RIPPLE_AIR_BAG_PRESSURE": "ri", "EDGE_AIR_BAG_PRESSURE": "ed",
    "SLURRY_FLOW_LINE_A": "sa", "SLURRY_FLOW_LINE_B": "sb", "SLURRY_FLOW_LINE_C": "sc",
    "WAFER_ROTATION": "wr", "STAGE_ROTATION": "sr", "HEAD_ROTATION": "hr", "DRESSING_WATER_STATUS": "dw",
}


# ----------------------------------------------------------------------------------------------
# 1. 단계별 채널 모양 통계
# ----------------------------------------------------------------------------------------------
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
    """(WAFER_ID, STAGE) 별 단계 0,1,2 채널 모양 특징 DataFrame."""
    t0 = time.time()
    cols = ["TIMESTAMP", "ACTIVE", "DRESSING_WATER_STATUS"] + CHANNELS
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
        for j, c in enumerate(CHANNELS):
            _channel_stats(t, arr[:, 3 + j], a, dt, f"{p}_{SHORT[c]}", out)
        # 채널 간 지연: 가압 챔버 압력 시작·종료 기준
        pc_on, pc_off = out.get(f"{p}_pc_t_on"), out.get(f"{p}_pc_t_off")
        for c in CHANNELS[1:]:
            k = SHORT[c]
            if pc_on is not None and f"{p}_{k}_t_on" in out:
                out[f"{p}_{k}_lag_on"] = out[f"{p}_{k}_t_on"] - pc_on
            if pc_off is not None and f"{p}_{k}_t_off" in out:
                out[f"{p}_{k}_lag_off"] = out[f"{p}_{k}_t_off"] - pc_off
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=KEY)
    print(f"[tsshape] shape features {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    return df.reset_index()


# ----------------------------------------------------------------------------------------------
# 2. 레시피 지문 (k-means, 레짐 중앙 궤적 잔차, PCA)
# ----------------------------------------------------------------------------------------------
def fingerprint_features(raw, table, n_points=32, k=8, n_pca=8, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    t0 = time.time()
    keys, X = sequences(raw, n_points=n_points, cols=CHANNELS)  # [n, 3, n_points, C]
    reg_by_key = dict(zip(map(tuple, table[KEY].itertuples(index=False)), regime(table)))
    reg = np.array([reg_by_key[tuple(r)] for r in keys.itertuples(index=False)])
    out = pd.DataFrame(index=keys.index)
    for step in range(3):
        Z = X[:, step]  # [n, n_points, C]
        ok = ~np.isnan(Z).any(axis=(1, 2))
        if ok.sum() < 20:
            continue
        # 채널별 전역 표준화 (레짐 무관)
        mu = np.nanmean(Z[ok], axis=(0, 1))
        sd = np.nanstd(Z[ok], axis=(0, 1)) + 1e-6
        S = (Z - mu) / sd  # [n, n_points, C]
        F = S.reshape(len(S), -1)
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(F[ok])
        p = f"fp{step}"
        lab = np.full(len(F), np.nan)
        lab[ok] = km.labels_
        out[f"{p}_cluster"] = lab
        d = np.full(len(F), np.nan)
        d[ok] = np.linalg.norm(F[ok] - km.cluster_centers_[km.labels_], axis=1)
        out[f"{p}_cdist"] = d
        # 군집 크기(드문 레시피인가)
        size = np.bincount(km.labels_, minlength=k)
        cs = np.full(len(F), np.nan)
        cs[ok] = size[km.labels_] / ok.sum()
        out[f"{p}_csize"] = cs
        # 레짐 중앙 궤적과의 잔차 (전체·채널별)
        resid = np.full((len(F), len(CHANNELS)), np.nan)
        for r in np.unique(reg):
            m = ok & (reg == r)
            if m.sum() < 5:
                continue
            med = np.median(S[m], axis=0)  # [n_points, C]
            resid[m] = np.sqrt(((S[m] - med) ** 2).mean(axis=1))
        out[f"{p}_resid_all"] = np.sqrt(np.nanmean(resid ** 2, axis=1)) if ok.any() else np.nan
        for j, c in enumerate(CHANNELS):
            out[f"{p}_resid_{SHORT[c]}"] = resid[:, j]
        pca = PCA(n_components=n_pca, random_state=seed).fit(F[ok])
        P = np.full((len(F), n_pca), np.nan)
        P[ok] = pca.transform(F[ok])
        for i in range(n_pca):
            out[f"{p}_pca{i}"] = P[:, i]
    out = pd.concat([keys, out], axis=1)
    print(f"[tsshape] fingerprint features {out.shape} ({time.time() - t0:.0f}s)", flush=True)
    return out


def compute_all(force=False):
    """모양 + 지문 특징을 계산해 parquet에 캐시하고 (WAFER_ID, STAGE) 키를 포함한 DataFrame을 돌려준다."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    raw, table = get_raw(), get_table()
    df = shape_features(raw).merge(fingerprint_features(raw, table), on=KEY, how="outer")
    df.to_parquet(CACHE)
    return df


def align(table, df):
    """캐시 DataFrame을 table 행 순서(index)에 맞춘다."""
    m = table[KEY].merge(df, on=KEY, how="left")
    m.index = table.index
    return m.drop(columns=KEY)


# ----------------------------------------------------------------------------------------------
# 3. 비지도 가지치기 + 특징 가족(family) 선택
# ----------------------------------------------------------------------------------------------
FAMILIES = {
    "timing": ("_step_sec", "_n_rows", "_dt_mean", "_active_frac", "_pre_active_sec", "_post_active_sec",
               "_active_span_sec", "_active_gaps", "_dw_frac", "_dw_active_frac", "_dw_toggles"),
    "ramp": ("_ramp_up", "_ramp_down", "_hi_sec", "_hi_frac", "_t_stable", "_t_on", "_t_off"),
    "level": ("_plateau", "_pstd", "_q05", "_q25", "_q75", "_q95", "_iqr", "_first", "_last", "_max",
              "_min_active", "_zero_frac", "_integral"),
    "drift": ("_slope", "_half_diff", "_n_jumps", "_abs_diff_mean"),
    "levels": ("_n_levels", "_n_changes", "_longest_run", "_n_levels_active"),
    "lag": ("_lag_on", "_lag_off"),
    "fp": ("fp0_", "fp1_", "fp2_"),
}


def family_of(col):
    if col.startswith("fp"):
        return "fp"
    for fam, sufs in FAMILIES.items():
        if fam != "fp" and col.endswith(sufs):
            return fam
    return None


def prune(df, base=None, min_notna=0.05, max_corr=0.98):
    """정답 없이 가지치기: 결측 비율·상수·기존 표/앞선 컬럼과 |상관| > max_corr 인 컬럼 제거."""
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


def select(cols, families):
    return [c for c in cols if family_of(c) in families]


# ----------------------------------------------------------------------------------------------
# 4. 실험
# ----------------------------------------------------------------------------------------------
def make_static(families, extra_prune=None):
    """static_fn 팩토리: 캐시 특징 중 지정 가족만 골라 table 순서로 돌려준다."""
    def static_fn(table):
        df = align(table, compute_all())
        base = table[[c for c in table.columns if c not in EXCLUDE]]
        cols = prune(df, base)
        cols = select(cols, families)
        if extra_prune is not None:
            cols = extra_prune(cols, df)
        out = df[cols].astype(float)
        out.columns = [f"TS_{c}" for c in out.columns]
        print(f"[tsshape] static features {out.shape[1]} ({', '.join(families)})", flush=True)
        return out
    return static_fn


ALL_FAM = tuple(FAMILIES)
N_JOBS = {"lgb": 1, "lgb_small": 1, "extra_trees": 4}  # LightGBM 기본(n_jobs=None)은 코어 전부를 써서 경합 시 매우 느리다


def _threaded(name, spec):
    """모델 팩토리를 감싸 n_jobs를 명시한다 (결과는 같고 스레드 수만 다르다)."""
    make, nj = spec["make"], N_JOBS.get(name)
    if nj is None:
        return spec

    def mk():
        m = make()
        m.set_params(n_jobs=nj)
        return m
    return {**spec, "make": mk}


BASE_MODELS = {k: _threaded(k, v) for k, v in BASE_MODELS.items()}


def _base_only(cols):
    return [c for c in cols if not c.startswith("TS_")]


def _compact_plus_ts(cols):
    return [c for c in cols if c.startswith(COMPACT_PREFIX) or c.startswith("TS_")]


# 기본 멤버는 그대로 두고(lgb는 TS_ 제외) TS 특징을 보는 lgb 멤버를 하나 더 붙인다 → NNLS가 가중치를 정한다
MODELS_MEMBER = {**{k: ({**v, "cols": _base_only} if v["cols"] == "all" else v) for k, v in BASE_MODELS.items()},
                 "lgb_ts": {**BASE_MODELS["lgb"], "cols": "all"}}
# compact 멤버(lgb_small, extra_trees, ridge)에도 TS 특징을 준다
MODELS_COMPACT_TS = {k: ({**v, "cols": _compact_plus_ts} if v["cols"] == "compact" else v) for k, v in BASE_MODELS.items()}

# 도메인 기준으로 고른 작은 집합(정답 미사용): 가압 램프·안정화·드리프트·하위 단계 수, 슬러리/헤드 회전 지연,
# 단계 앞뒤 비활성 시간, 드레싱 워터, 레시피 지문(군집·중심 거리·레짐 잔차)
LEAN = tuple(f"s{k}_{c}" for k in range(3) for c in (
    "pc_ramp_up", "pc_ramp_down", "pc_t_stable", "pc_slope", "pc_n_levels", "sc_lag_on", "sc_lag_off", "hr_lag_on",
    "pre_active_sec", "post_active_sec", "dw_active_frac")) + tuple(
    f"fp{k}_{c}" for k in range(3) for c in ("cluster", "cdist", "resid_all"))


def _lean(cols, df):
    return [c for c in cols if c in LEAN]


VARIANTS = {
    "nofp": dict(families=tuple(f for f in ALL_FAM if f != "fp")),
    "lite": dict(families=("timing", "ramp", "drift", "levels", "lag")),
    "lean": dict(families=ALL_FAM, extra_prune=_lean),
    "lean_member": dict(families=ALL_FAM, extra_prune=_lean, models=MODELS_MEMBER),
    "fp": dict(families=("fp",)),
    "all": dict(families=ALL_FAM),
    "timing": dict(families=("timing",)),
    "ramp": dict(families=("ramp",)),
    "lag": dict(families=("lag",)),
    "levels": dict(families=("levels",)),
    "drift": dict(families=("drift",)),
    "level": dict(families=("level",)),
    "all_member": dict(families=ALL_FAM, models=MODELS_MEMBER),
    "fp_member": dict(families=("fp",), models=MODELS_MEMBER),
    "fp_compact": dict(families=("fp",), models=MODELS_COMPACT_TS),
    "all_compact": dict(families=ALL_FAM, models=MODELS_COMPACT_TS),
}
FINAL = "lite"


def run_variant(name, fold_seed=0, out=None, **kw):
    spec = dict(VARIANTS[name]); spec.update(kw)
    models = spec.pop("models", BASE_MODELS)
    res = run(static_fn=make_static(**spec), models=models, fold_seed=fold_seed, label=f"tsshape_{name}", out=out)
    print(summary(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="탐색 변형 전부 실행")
    ap.add_argument("--variants", default="", help="쉼표로 구분한 변형 이름 (탐색용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="캐시 다시 계산")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    t0 = time.time()
    compute_all(force=args.force)
    if args.all or args.variants:
        names = args.variants.split(",") if args.variants else list(VARIANTS)
        results = {}
        for n in names:
            out = (Path(args.outdir) / f"tsshape_{n}_s{args.seed}.json") if args.outdir else None
            results[n] = run_variant(n, fold_seed=args.seed, out=out)
        print("\n=== summary ===")
        for n, r in results.items():
            print(summary(r))
    else:
        res = run_variant(FINAL, out=str(HERE / "tsshape.json"))
    print(f"[done] {time.time() - t0:.0f}s")
