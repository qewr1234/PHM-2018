"""대시보드용 요약 수치를 만든다.

train_rows.py로 만든 표본과 검증 확률(--save-proba)에서 대시보드가 쓰는 값만 뽑아 JSON으로 저장한다.
    - 데이터 개요 (장비, 행 수, 고장 건수)
    - 고장 직전 센서 변화 (레시피 단계 기준 이탈, 압력->유량 잔차)
    - 검증 구간의 고장 임박(1시간 이내) 감지 성능: ROC, 오경보 1%에서의 감지율, 남은 시간별 감지율
    - 감지 위험도 타임라인 예시, 중요 특징
    - 대회 채점식(S1, S2, 최종) 점수

사용 예:
    python build_dashboard.py --sample data/samples/rows_v3.parquet --proba data/val_proba_v5.npz \\
        --out dashboard/data.json
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from phm_data import (FAULT_NAMES, TTF_COLS, phm_subscores, phm_subscores_final,
                      phm_subscores_secondary)
from train_rows import META_COLS, decide, file_scores, ttf_to_class

NEAR_1H_CLASSES = ttf_to_class([3599])[0] + 1  # 0..8 구간이 1시간 이내
FAULT_SHORT = ["압력 저하", "압력 과다", "누설"]
PROFILE_BINS = [-10800, -7200, -3600, -1800, -900, -300, -60, 0]
PROFILE_LABELS = ["3~2시간", "2~1시간", "60~30분", "30~15분", "15~5분", "5~1분", "1분 이내"]
LEAD_BINS = [0, 300, 900, 1800, 3600, 10800]
LEAD_LABELS = ["5분 이내", "5~15분", "15~30분", "30~60분", "1~3시간"]
PROFILE_FEATURES = {
    "FLOWCOOLPRESSURE_z_mean_1m": "FlowCool 압력 (단계 기준 z)",
    "FLOWCOOLFLOWRATE_z_mean_1m": "FlowCool 유량 (단계 기준 z)",
    "flow_resid_mean_1m": "압력 대비 유량 잔차",
}


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return values[order][np.searchsorted(cw, q * cw[-1])]


def overview(rows_dir):
    rows_dir = Path(rows_dir)
    faults = pd.concat(
        pd.read_csv(p).assign(tool=p.name[:6]).drop_duplicates(subset=["time", "fault_name"])
        for p in sorted((rows_dir / "train_faults").glob("*.csv"))
    )
    faults = faults[faults["fault_name"].isin(FAULT_NAMES)]
    per_tool = (faults.groupby(["tool", "fault_name"]).size().unstack(fill_value=0)
                .reindex(columns=FAULT_NAMES, fill_value=0))
    n_rows = {split: int(sum(pd.read_parquet(p, columns=["time"]).shape[0]
                             for p in (rows_dir / split).glob("*.parquet")))
              for split in ["train", "test"]}
    return {
        "tools": len(per_tool),
        "train_rows": n_rows["train"],
        "test_rows": n_rows["test"],
        "faults_by_type": [int(faults["fault_name"].eq(n).sum()) for n in FAULT_NAMES],
        "faults_by_tool": {t: [int(v) for v in r] for t, r in per_tool.iterrows()},
    }


def profiles(data):
    """고장까지 남은 시간 구간별 센서 평균 (정상 구간 평균을 기준선으로)."""
    out = {}
    w = data["weight"].to_numpy()
    for i, c in enumerate(TTF_COLS):
        gt = data[c].to_numpy(dtype=float)
        far = np.isnan(gt) | (gt > 50000)
        bins = pd.cut(-gt, bins=PROFILE_BINS, labels=PROFILE_LABELS, right=False)
        prof = {}
        for f, label in PROFILE_FEATURES.items():
            x = data[f].to_numpy(dtype=float)
            base = np.average(x[far], weights=w[far])
            means = pd.Series(x - base).groupby(bins, observed=False).mean()
            prof[label] = [round(float(v), 3) for v in means.to_numpy()]
        out[FAULT_SHORT[i]] = prof
    return {"bins": PROFILE_LABELS, "by_fault": out}


def detection(val, proba_npz, fa_rate=0.01):
    d = np.load(proba_npz, allow_pickle=True)
    w = d["weight"]
    gt = d["gt"]
    results = []
    for i in range(len(TTF_COLS)):
        risk = d[f"proba_{i}"][:, :NEAR_1H_CLASSES].sum(axis=1).astype(float)
        g = np.nan_to_num(gt[:, i], nan=np.inf)
        pos = g < 3600
        far = g >= 10800
        fpr, tpr, _ = roc_curve(pos, risk, sample_weight=w)
        keep = np.unique(np.r_[0, np.searchsorted(fpr, np.linspace(0, 1, 80)).clip(0, len(fpr) - 1), len(fpr) - 1])
        thr = weighted_quantile(risk[far], w[far], 1 - fa_rate)
        flagged = risk >= thr
        lead = []
        for lo, hi in zip(LEAD_BINS[:-1], LEAD_BINS[1:]):
            m = (g >= lo) & (g < hi)
            lead.append(round(float(np.average(flagged[m], weights=w[m])), 4) if m.any() else None)
        base_rate = float(np.average(pos, weights=w))
        results.append({
            "fault": FAULT_SHORT[i],
            "auc": round(float(roc_auc_score(pos, risk, sample_weight=w)), 3),
            "ap": round(float(average_precision_score(pos, risk, sample_weight=w)), 4),
            "base_rate": round(base_rate, 5),
            "recall_at_fa": round(float(np.average(flagged[pos], weights=w[pos])), 4),
            "fa_rate": fa_rate,
            "roc": [[round(float(fpr[k]), 4), round(float(tpr[k]), 4)] for k in keep],
            "recall_by_lead": lead,
            "threshold": float(thr),
        })
    return {"lead_bins": LEAD_LABELS, "faults": results}


def timeline(val, proba_npz, detections, hours=48, max_points=1500):
    """검증 구간에서 고장이 가장 많은 장비/고장 조합의 위험도 추이 예시."""
    d = np.load(proba_npz, allow_pickle=True)
    best = None
    for i, c in enumerate(TTF_COLS):
        g = val[c].to_numpy(dtype=float)
        fault_time = np.round(val["time"].to_numpy() + g)[g < 60]
        for tool in val["tool"].unique():
            ft = np.unique(fault_time[(val["tool"].to_numpy()[g < 60] == tool)])
            if best is None or len(ft) > best[0]:
                best = (len(ft), i, tool, ft)
    _, i, tool, ft = best
    # 고장이 가장 몰린 hours 시간 창을 고른다.
    counts = [np.sum((ft >= t) & (ft < t + hours * 3600)) for t in ft]
    start = ft[int(np.argmax(counts))] - 6 * 3600
    end = start + hours * 3600
    m = (val["tool"].to_numpy() == tool) & (val["time"].to_numpy() >= start) & (val["time"].to_numpy() < end)
    risk = d[f"proba_{i}"][m][:, :NEAR_1H_CLASSES].sum(axis=1)
    t = val["time"].to_numpy()[m]
    order = np.argsort(t)
    t, risk = t[order], risk[order]
    if len(t) > max_points:
        # 구간별 최댓값으로 줄여 짧은 위험도 급등이 사라지지 않게 한다.
        edges = np.linspace(start, end, max_points + 1)
        idx = np.searchsorted(edges, t, side="right") - 1
        s = pd.DataFrame({"i": idx, "t": t, "r": risk}).groupby("i").agg(t=("t", "mean"), r=("r", "max"))
        t, risk = s["t"].to_numpy(), s["r"].to_numpy()
    return {
        "tool": tool,
        "fault": FAULT_SHORT[i],
        "hours": hours,
        "points": [[round(float((a - start) / 3600), 3), round(float(b), 4)] for a, b in zip(t, risk)],
        "faults": [round(float((f - start) / 3600), 3) for f in ft if start <= f < end],
        "threshold": detections["faults"][i]["threshold"],
    }


def importance(train, top=8, seed=0):
    """고장 모드별로 '1시간 이내 고장' 이진 분류기를 가볍게 학습해 중요 특징(gain)을 뽑는다."""
    feature_cols = [c for c in train.columns if c not in META_COLS]
    sub = train.sample(frac=0.3, random_state=seed)
    out = {}
    for i, c in enumerate(TTF_COLS):
        y = np.nan_to_num(sub[c].to_numpy(dtype=float), nan=np.inf) < 3600
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
                                   min_child_weight=10.0, reg_lambda=10.0, max_bin=63, random_state=seed,
                                   verbose=-1, importance_type="gain")
        model.fit(sub[feature_cols], y, sample_weight=sub["weight"] / sub["weight"].mean())
        gain = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        share = gain / gain.sum()
        out[FAULT_SHORT[i]] = [[k, round(float(v), 4)] for k, v in share.head(top).items()]
    return out


def competition(val, proba_npz):
    d = np.load(proba_npz, allow_pickle=True)
    w, gt, tools = d["weight"], d["gt"], d["tool"]
    preds = np.column_stack([decide(d[f"proba_{i}"].astype(float), d[f"cost_{i}"], d["actions"])
                             for i in range(len(TTF_COLS))])
    nan = np.full_like(gt, np.nan)
    rows = {}
    for name, fn in [("S1", phm_subscores), ("S2", phm_subscores_secondary), ("final", phm_subscores_final)]:
        rows[name] = {
            "model": round(float(file_scores(tools, w, fn(gt, preds)).sum()), 3),
            "all_nan": round(float(file_scores(tools, w, fn(gt, nan)).sum()), 3),
        }
    return rows


def main():
    p = argparse.ArgumentParser(description="대시보드용 요약 수치 생성")
    p.add_argument("--rows-dir", default="data/rows")
    p.add_argument("--sample", default="data/samples/rows_v3.parquet")
    p.add_argument("--proba", default="data/val_proba_v5.npz")
    p.add_argument("--out", default="dashboard/data.json")
    args = p.parse_args()

    data = pd.read_parquet(args.sample)
    val = data[data["is_val"]].reset_index(drop=True)
    det = detection(val, args.proba)
    result = {
        "overview": overview(args.rows_dir),
        "profiles": profiles(data),
        "detection": det,
        "timeline": timeline(val, args.proba, det),
        "importance": importance(data[~data["is_val"]]),
        "competition": competition(val, args.proba),
        "split": {"train_rows": int((~data["is_val"]).sum()), "val_rows": int(len(val))},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False))
    print(f"[save] {args.out}")
    render(args.out)


def render(data_path, template="dashboard/template.html", out="dashboard/index.html"):
    """템플릿에 요약 수치를 넣어 한 파일짜리 대시보드 HTML을 만든다."""
    data = Path(data_path).read_text().replace("</", "<\\/")
    Path(out).write_text(Path(template).read_text().replace("__DATA__", data))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
