"""기계공학 관점 분석: 드레서 마모(수명 주기·교체 전후), Preston 식의 P·V 설명력.

예측 모델과 별개로, 정답이 공개된 모든 웨이퍼(학습·테스트·검증, 측정 오류 4개 제외)를 써서
장비 물리에 관한 질문에 답한다.

1. 드레서 교체 시점(사용량 카운터 초기화)을 찾고, 교체 직전·직후 N장의 연마량을 비교한다 (Welch t-검정).
2. 한 드레서 수명 안에서 사용량에 따른 연마량 기울기를 수명별 절편(고정효과) 회귀로 구한다.
3. 수명 안 사용량 구간별 연마량 편차로 마모 곡선 모양(초기·정상·말기)을 본다.
4. 드레서 교체 때 다른 소모품 카운터도 같이 초기화됐는지 확인한다 (동시 정비 여부).
5. 실제 연마 구간(가압 + 슬러리 공급)의 압력 P, 헤드 회전 V, P·V가 웨이퍼마다 얼마나 다른지와
   연마량과의 순위 상관을 구해, Preston 식 MRR = Kp·P·V에서 P·V가 연마량 차이를 설명하는지 본다.

사용 예:
    python cmp2016/mech_analysis.py --out cmp2016/mech_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_data import KEY, TARGET, build_table, load_labels, load_split, regime  # noqa: E402

RESET_DROP = 200  # 드레서 사용량이 이만큼 이상 떨어지면 교체로 본다 (사용량 범위 약 5~772)
N_AROUND = 20  # 교체 전후 비교에 쓰는 웨이퍼 수
USAGE_BINS = [0, 100, 200, 300, 400, 500, 600, 700, 800]
OTHER_COUNTERS = ["USAGE_OF_POLISHING_TABLE", "USAGE_OF_MEMBRANE", "USAGE_OF_BACKING_FILM"]


def labeled_table():
    t = build_table()
    t = t[~t["OUTLIER"] & t[TARGET].notna()].copy()
    t["REG"] = regime(t)
    t["DAY"] = (t["T_START"] - t["T_START"].min()) / 86400
    t = t.sort_values("T_START")
    for grp, x in t.groupby("GROUP"):
        t.loc[x.index, "LIFE"] = (x["USAGE_OF_DRESSER_start"].diff() < -RESET_DROP).cumsum()
    return t


def reset_days(x, col, drop):
    return x.loc[x[col].diff() < -drop, "DAY"].to_numpy()


def dresser_events(t):
    out = []
    for reg, x in t.groupby("REG"):
        for day in reset_days(t[t["GROUP"] == x["GROUP"].iloc[0]], "USAGE_OF_DRESSER_start", RESET_DROP):
            before = x[x["DAY"] < day].tail(N_AROUND)[TARGET]
            after = x[x["DAY"] >= day].head(N_AROUND)[TARGET]
            test = stats.ttest_ind(after, before, equal_var=False)
            out.append({"regime": reg, "day": float(day), "before": float(before.mean()),
                        "after": float(after.mean()), "jump": float(after.mean() - before.mean()),
                        "p_value": float(test.pvalue)})
    return out


def within_life_slope(t):
    """수명별 절편을 둔 OLS: MRR = a_life + b · (드레서 사용량 / 100)."""
    out = {}
    for reg, x in t.groupby("REG"):
        X = pd.get_dummies(x["LIFE"].astype(int), prefix="life", dtype=float)
        X["use100"] = x["USAGE_OF_DRESSER_start"] / 100
        A, y = X.to_numpy(), x[TARGET].to_numpy()
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        se = np.sqrt(resid.var(ddof=A.shape[1]) * np.linalg.inv(A.T @ A)[-1, -1])
        out[reg] = {"slope_per_100": float(coef[-1]), "se": float(se), "n": int(len(x)),
                    "lives": int(x["LIFE"].nunique())}
    return out


def wear_shape(t):
    out = {}
    for reg, x in t.groupby("REG"):
        dev = x[TARGET] - x.groupby("LIFE")[TARGET].transform("mean")
        b = pd.cut(x["USAGE_OF_DRESSER_start"], USAGE_BINS)
        g = dev.groupby(b, observed=True).agg(["mean", "count"])
        out[reg] = [{"from": float(i.left), "to": float(i.right), "dev": float(r["mean"]), "n": int(r["count"])}
                    for i, r in g.iterrows()]
    return out


def coincident_resets(t):
    out = {}
    for grp, x in t.groupby("GROUP"):
        dresser = reset_days(x, "USAGE_OF_DRESSER_start", RESET_DROP)
        row = {}
        for c in OTHER_COUNTERS:
            col = f"{c}_start"
            days = reset_days(x, col, 0.3 * x[col].std())
            row[c] = [float(days[np.argmin(abs(days - d))] - d) if len(days) else None for d in dresser]
        out[int(grp)] = row
    return out


def preston_variability():
    """실제 연마 구간의 P, V, P·V 변동과 연마량의 순위 상관 (학습·테스트·검증 전체)."""
    frames = []
    for split in ["train", "test", "val"]:
        raw = load_split(split).sort_values(KEY + ["TIMESTAMP"])
        group = np.where(raw.groupby(KEY)["CHAMBER"].transform("min") <= 3, 1, 4)
        act = raw.assign(GROUP=group)
        act = act[(act["PRESSURIZED_CHAMBER_PRESSURE"] > 0) & (act["SLURRY_FLOW_LINE_C"] > 0)]
        act = act.assign(PV=act["PRESSURIZED_CHAMBER_PRESSURE"] * act["HEAD_ROTATION"])
        w = act.groupby(KEY).agg(GROUP=("GROUP", "first"), P=("PRESSURIZED_CHAMBER_PRESSURE", "mean"),
                                 V=("HEAD_ROTATION", "mean"), PV=("PV", "mean"),
                                 active_sec=("TIMESTAMP", "size"), slurry=("SLURRY_FLOW_LINE_C", "mean"))
        frames.append(w.reset_index().merge(load_labels(split), on=KEY))
    w = pd.concat(frames, ignore_index=True)
    w = w[w[TARGET] < 1000]
    w["REG"] = w["GROUP"].astype(str) + w["STAGE"]

    def cv(s):
        return float(s.std() / s.mean() * 100)

    out = {}
    for reg, x in w.groupby("REG"):
        out[reg] = {
            "n": int(len(x)),
            "cv_mrr": cv(x[TARGET]), "cv_p": cv(x["P"]), "cv_v": cv(x["V"]), "cv_pv": cv(x["PV"]),
            "cv_active_sec": cv(x["active_sec"]), "cv_slurry": cv(x["slurry"]),
            "rho_mrr_pv": float(stats.spearmanr(x[TARGET], x["PV"]).statistic),
            "rho_mrr_p": float(stats.spearmanr(x[TARGET], x["P"]).statistic),
            "rho_mrr_slurry": float(stats.spearmanr(x[TARGET], x["slurry"]).statistic),
        }
    return out


def main():
    p = argparse.ArgumentParser(description="CMP 기계공학 관점 분석")
    p.add_argument("--out", default="cmp2016/mech_results.json")
    args = p.parse_args()

    t = labeled_table()
    res = {
        "n_wafers": int(len(t)),
        "dresser_events": dresser_events(t),
        "within_life_slope": within_life_slope(t),
        "wear_shape": wear_shape(t),
        "coincident_resets_days": coincident_resets(t),
        "preston": preston_variability(),
    }
    print(f"[data] labeled wafer-stages (outliers removed): {res['n_wafers']}")
    print(pd.DataFrame(res["dresser_events"]).round(3).to_string(index=False))
    for reg, s in res["within_life_slope"].items():
        print(f"[slope] {reg}: {s['slope_per_100']:+.3f} ± {s['se']:.3f} per 100 dresser use (n {s['n']}, lives {s['lives']})")
    print("[coincident resets, days from each dresser reset]", res["coincident_resets_days"])
    print(pd.DataFrame(res["preston"]).T.round(2).to_string())
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
