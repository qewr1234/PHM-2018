"""Phase-II 고정 지연 VM 실행: 특징 → 멤버(병렬) → 결합·채점·짝 붓스트랩 → cmp2016/fab_phase2_results.json

사용 예:
    python cmp2016/fab_phase2_run.py --lags 24,4,1 --procs 2                 # LightGBM 멤버 (L 하나에 CPU 약 40분)
    python cmp2016/fab_phase2_run.py --lags 24,4 --tfm tabdpt --procs 2      # + TabDPT 잔차 멤버 (L 하나에 약 11분)
    python cmp2016/fab_phase2_run.py --lags 24,4 --tfm tabpfn_c300,tabicl    # 선택: TabPFN v2·TabICLv2 (L 하나에 각 약 25분)
    (보고한 결과: L = 24·4 시간은 LightGBM + TabDPT·TabPFN v2·TabICLv2, L = 1 시간은 LightGBM + TabDPT. TFM 은 2 스레드)
    python cmp2016/fab_phase2_run.py --stage report --lags 24,4,1           # 캐시된 멤버로 결합·채점만
    FAB2_TFM_THREADS=1 python cmp2016/fab_phase2_run.py --stage leak --lags 24 --procs 1   # 실제 데이터 누설 검사
                                                   # (줄인 설정, 변형 3개 × 약 13분) → 결과 파일의 leak_check

캐시: data/cmp2016/fab_phase2/ (X_{L}.parquet, members/{L}_{멤버}.npy, preds_{L}.parquet). 캐시는 특징·멤버 코드 열쇠
(fab_phase2.code_key) 가 같을 때만 다시 쓴다 (--reuse). 멤버 조합 선택·동결 가중치·RTS 잡음비는 개발 기간 선행 예측으로만 정한다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
_TFM_CACHE = HERE.parent / "data" / "cmp2016" / "tfm_cache"   # = fab_phase2.TFM_CACHE (가중치만 여기서 읽는다)
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(_TFM_CACHE / "tabpfn"))
os.environ.setdefault("HF_HOME", str(_TFM_CACHE / "hf"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fab_phase2 as f2  # noqa: E402
from cmp_data import CACHE_DIR, TARGET  # noqa: E402

OUT_DIR = CACHE_DIR / "fab_phase2"
RESULTS = HERE / "fab_phase2_results.json"
CONTEST_REF = "3742cdc"        # 대회 방식 앙상블(lgb+seqdl+gp_state) 예측이 든 results.json 의 커밋 (작업 시작 때 HEAD 와 같은 내용)
REF_MSE = 6.36                 # 대회 방식(비인과) 최고 발표치 (Appl. Sci. 2022)
_D = {}


def key_of(L):
    return f"{L:g}h"


def data():
    if "D" not in _D:
        _D["D"] = f2.PhaseData()
    return _D["D"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- 특징
def features(L, reuse=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fx, fm = OUT_DIR / f"X_{key_of(L)}.parquet", OUT_DIR / f"X_{key_of(L)}.json"
    if reuse and fx.exists() and fm.exists() and json.loads(fm.read_text())["code"] == f2.code_key():
        return pd.read_parquet(fx), json.loads(fm.read_text())
    D = data()
    t0 = time.time()
    q, mse = f2.select_rts(D, L)
    X = f2.build_X(D, L, q)
    meta = {"code": f2.code_key(), "rts_q": {r: list(v) for r, v in q.items()}, "rts_dev_preq_mse": mse,
            "n_cols": int(X.shape[1]), "seconds": round(time.time() - t0, 1)}
    X.to_parquet(fx)
    fm.write_text(json.dumps(meta, indent=1))
    log(f"L={key_of(L)} X {X.shape} RTS q {meta['rts_q']} ({meta['seconds']}s)")
    return X, meta


# ----------------------------------------------------------------------------- 멤버 (프로세스 하나에 작업 하나)
def _member_path(L, name):
    return OUT_DIR / "members" / f"{key_of(L)}_{name}"


def member_job(job):
    L, name, reuse = job
    base = _member_path(L, name)
    spec = f2.MEMBERS[name]
    meta_f = base.with_suffix(".json")
    if reuse and meta_f.exists() and json.loads(meta_f.read_text())["code"] == f2.code_key():
        return name, L, "cached"
    X, _ = features(L, reuse=True)
    D = data()
    t0 = time.time()
    p = f2.fit_member(D, L, X, spec, log=lambda s: log(f"{key_of(L)} {name}{s}"))
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(base.with_suffix(".npy"), p)
    meta_f.write_text(json.dumps({"code": f2.code_key(), "spec": spec, "seconds": round(time.time() - t0, 1),
                                  "tfm_threads": f2._tfm_threads() if spec["kind"] != "lgb" else None}))
    log(f"{key_of(L)} {name} done {time.time() - t0:.0f}s")
    return name, L, round(time.time() - t0, 1)


def load_members(L):
    P, secs = {}, {}
    for name in f2.MEMBERS:
        base = _member_path(L, name)
        if base.with_suffix(".json").exists():
            meta = json.loads(base.with_suffix(".json").read_text())
            if meta["code"] == f2.code_key():
                P[name] = np.load(base.with_suffix(".npy"))
                secs[name] = meta["seconds"] if meta.get("tfm_threads") is None else \
                    {"seconds": meta["seconds"], "torch_threads": meta["tfm_threads"]}
    return P, secs


# ----------------------------------------------------------------------------- 결합 후보·선택
def candidate_sets(P):
    have = lambda names: [m for m in names if m in P]  # noqa: E731
    lgb_all = have(["gbm", "gbm_res_bag", "rts", "res_pool_bag", "res_6h", "res_pool6h", "K1", "INT"])
    res_only = have(["gbm_res_bag", "rts", "res_pool_bag", "res_6h", "res_pool6h"])
    base3 = have(["gbm", "gbm_res_bag", "rts"])
    tfm = have(f2.TFM_MEMBERS)
    C = {"base3": base3, "lgb_all": lgb_all, "res_only": res_only}
    if tfm:
        C["lgb_all+tfm"] = lgb_all + tfm
        C["base3+tfm"] = base3 + tfm
        C["res_only+tfm"] = res_only + tfm
        C["rts+tfm"] = ["rts"] + tfm
        if len(tfm) > 1:
            for m in tfm:
                C[f"lgb_all+{m}"] = lgb_all + [m]
    for m in have(["gbm", "gbm_res_bag", "res_pool_bag", "res_6h", "res_pool6h", "rts"] + tfm):
        C[f"single:{m}"] = [m]
    return C


def contest_errors(D):
    """대회 방식 앙상블 예측(고정 커밋의 results.json)을 이 흐름의 웨이퍼 순서에 맞춘 제곱오차 (test·val)."""
    txt = subprocess.run(["git", "show", f"{CONTEST_REF}:cmp2016/results.json"], cwd=HERE.parent, check=True,
                         capture_output=True, text=True).stdout
    cp = pd.DataFrame(json.loads(txt)["predictions"])
    T = D.s.t
    key = pd.DataFrame({"split": D.split, "group": T["GROUP"].to_numpy(), "stage": T["STAGE"].to_numpy(),
                        "t": ((T["T_START"] - T["T_START"].min()) / 86400).round(3).to_numpy(), "i": np.arange(D.n)})
    m = key.merge(cp, on=["split", "group", "stage", "t"], how="inner", validate="one_to_one")
    assert len(m) == len(cp) and np.allclose(m["y"], D.y_score[m["i"]], atol=2e-3), (len(m), len(cp))
    pred = np.full(D.n, np.nan)
    pred[m["i"].to_numpy()] = m["pred"].to_numpy()
    return pred


def report(lags, only_members=None):
    D = data()
    contest = contest_errors(D)
    res = {
        "protocol": {
            "measured": "공식 train − 큰 이상치 4개 (계측 웨이퍼, 약 70%); test·val 은 계측하지 않은 웨이퍼 (정답은 채점만)",
            "label_arrival_h": D.delay_h, "issue": "D_i = T_END_i + L, a_j <= D_i 인 계측 정답(j != i)만 사용",
            "refit": "멤버는 T_k = k·step 에 max(D_j, a_j) <= T_k 인 행으로 재학습, T_k < D_i <= T_k + step 예측",
            "dev_period": f"선택(RTS 잡음비·멤버 조합·동결 가중치) = {D.warm_day:g}일 이후 & max(D_j, a_j) < {D.dev_day:g}일 선행 예측",
            "deploy_period": f"day >= {D.dev_day:g} (T_END 기준)",
            "headline": "online = 레짐별 NNLS 가중치를 매일 그때까지의 선행 예측으로 다시 맞춤 (완전 인과)",
            "dev_frozen": "개발 기간 선행 예측으로 맞춘 가중치를 고정 (배포 기간 인과)",
            "global_reference": "7일 이후 전체 선행 예측으로 맞춘 가중치 — 미래 정답 사용, 인과 아님, 낙관적 참고치",
            "contest_reference": f"git {CONTEST_REF}:cmp2016/results.json predictions (대회 방식 앙상블, 비인과)",
        },
        "config": {"lgb": f2.LGB, "topk": f2.TOPK, "tfm_topk": f2.TFM_TOPK, "min_rows": f2.MIN_ROWS,
                   "blend_min_rows": f2.BLEND_MIN_ROWS, "members": f2.MEMBERS, "bags": f2.BAGS,
                   "rts_grid": {"qh": f2.QH_GRID, "qj": f2.QJ_GRID, "gap_h": f2.GAP_H}},
        "code_key": f2.code_key(),
        "n": {"wafers": int(D.n), "measured": int(D.meas.sum()), "test": int((D.split == "test").sum()),
              "val": int((D.split == "val").sum())},
        "lags": {},
    }
    ce = {}
    y = D.y_score
    for split in ("test", "val"):
        m = D.split == split
        ce[split] = {"all": f2._mse(contest, y, m), "after7": f2._mse(contest, y, m & (D.day >= D.warm_day)),
                     "dev7_30": f2._mse(contest, y, m & (D.day >= D.warm_day) & (D.day < D.dev_day)),
                     "deploy": f2._mse(contest, y, m & (D.day >= D.dev_day))}
    res["contest_reference"] = ce
    frames = {}
    for L in lags:
        k = key_of(L)
        X, meta = features(L, reuse=True)
        P, secs = load_members(L)
        if only_members is not None:
            P = {m: v for m, v in P.items() if m in only_members}
        P = f2.add_bags(P)
        P.update(f2.nolearn_members(X))
        out = {"L_h": L, "rts_q": meta["rts_q"], "rts_dev_preq_mse": meta["rts_dev_preq_mse"], "n_cols": meta["n_cols"],
               "member_seconds": secs, "members": {}, "candidates": {}}
        preds = {}
        for m, p in P.items():
            q, nfb = f2.finalize(D, L, p)
            out["members"][m] = f2.score(D, L, q, nfb)
            preds[f"m:{m}"] = q
        dev = D.dev_rows(L)
        C = candidate_sets(P)
        B = {}
        for name, names in C.items():
            b, W = f2.blend(D, L, P, names, "online")
            q, _ = f2.finalize(D, L, b)
            ok = dev & np.isfinite(q)
            out["candidates"][name] = {"names": names, "dev_preq_online": f2._mse(q, D.ym, ok)}
            B[name] = q
        sel = min(C, key=lambda s: out["candidates"][s]["dev_preq_online"])
        names = C[sel]
        out["selected_set"] = sel
        out["selected_names"] = names
        for mode, label in (("online", "headline_online"), ("frozen", "dev_frozen"), ("global", "global_reference_optimistic")):
            b, W = f2.blend(D, L, P, names, mode)
            q, nfb = f2.finalize(D, L, b)
            out[label] = {**f2.score(D, L, q, nfb), ("weights_last" if mode == "online" else "weights"): W}
            preds[f"b:{mode}"] = q
        # 선택과 무관한 비교: 가장 큰 조합의 online (TFM 이 있으면 lgb_all+tfm)
        big = "lgb_all+tfm" if "lgb_all+tfm" in C else "lgb_all"
        out["largest_set_online"] = {"set": big, **f2.score(D, L, B[big])}
        boot = {}
        for mode in ("online", "frozen"):
            for split in ("test", "val"):
                for per, pm in (("", np.ones(D.n, bool)), ("_deploy", D.day >= D.dev_day)):
                    if per and mode != "online":
                        continue
                    m = (D.split == split) & pm
                    e1 = (preds[f"b:{mode}"][m] - y[m]) ** 2
                    e0 = (contest[m] - y[m]) ** 2
                    boot[f"{mode}_{split}{per}"] = f2.paired_bootstrap(e1, e0, ref=REF_MSE)
        out["bootstrap"] = boot
        res["lags"][k] = out
        summ = {"selected_set": sel}
        for label in ("headline_online", "dev_frozen", "global_reference_optimistic"):
            v = out[label]
            summ[label] = {f"{s}_{p}": v[s][p] for s in ("test", "val") for p in ("all", "after7", "dev7_30", "deploy")}
            summ[label].update({f"preq_{p}": v["preq"][p] for p in ("after7", "dev", "deploy")})
        summ["p_test_mse_below_6.36_online"] = boot["online_test"][f"p_mse_below_{REF_MSE}"]
        summ["test_diff_vs_contest_online_ci95"] = boot["online_test"]["diff_vs_ref_model_ci95"]
        res.setdefault("summary", {})[k] = summ
        T = D.s.t
        frames[k] = pd.DataFrame({"WAFER_ID": T["WAFER_ID"].to_numpy(), "STAGE": T["STAGE"].to_numpy(),
                                  "split": D.split, "reg": D.reg, "day": D.day, "t_h": D.t, "D_h": D.t + L,
                                  "meas": D.meas, "y": y, "contest": contest, **preds})
        h = out["headline_online"]
        log(f"L={k} selected {sel}: online test {h['test']['all']} val {h['val']['all']} | deploy "
            f"{h['test']['deploy']}/{h['val']['deploy']} | preq {h['preq']['after7']} | frozen "
            f"{out['dev_frozen']['test']['all']}/{out['dev_frozen']['val']['all']} | global "
            f"{out['global_reference_optimistic']['test']['all']}/{out['global_reference_optimistic']['val']['all']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, df in frames.items():
        df.to_parquet(OUT_DIR / f"preds_{k}.parquet")
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if "leak_check" in old:
        res["leak_check"] = old["leak_check"]
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"saved {RESULTS}")
    return res


# ----------------------------------------------------------------------------- 실제 데이터 누설 검사
LEAK_C_DAY = 30.5              # 이 시각 뒤에 도착하는 학습 정답을 바꾼다 (일 단위 재학습 경계와 어긋나게)
LEAK_MEMBERS = ("gbm", "gbm_res", "res_pool", "res_6h", "tabdpt")
LEAK_TFM_TMAX_DAY = 33


def _leak_table(variant):
    t = pd.read_parquet(CACHE_DIR / "table.parquet")
    if variant == "junk":                                 # 테스트·검증 정답을 모두 엉터리 값으로
        m = t["split"].isin(["test", "val"])
        t.loc[m, TARGET] = np.random.default_rng(1).uniform(0, 5000, m.sum())
    elif variant == "late":                               # 도착 시각이 C 뒤인 학습 정답에 잡음
        te = (t["T_START"] + t["DURATION"] - t["T_START"].min()) / 3600.0
        late = (t["split"] == "train") & (te + f2.DEFAULT_DELAY_H > LEAK_C_DAY * 24)
        t.loc[late, TARGET] += np.random.default_rng(2).normal(0, 20, late.sum())
    return t


def leak_job(job):
    variant, L = job
    D = f2.PhaseData(table=_leak_table(variant))
    q, _ = f2.select_rts(D, L)
    X = f2.build_X(D, L, q)
    P = {}
    for m in LEAK_MEMBERS:
        spec = dict(f2.MEMBERS[m])
        if spec["kind"] != "lgb":
            spec["n_est"] = 1
        P[m] = f2.fit_member(D, L, X, spec, params={**f2.LGB, "n_estimators": 150},
                             t_max=LEAK_TFM_TMAX_DAY * 24 if spec["kind"] != "lgb" else None)
    P.update(f2.nolearn_members(X))
    names = list(P)
    for mode in ("online", "frozen", "global"):
        P[f"blend_{mode}"], _ = f2.blend(D, L, P, names, mode)
    f = OUT_DIR / f"leak_{variant}_{key_of(L)}.npz"
    np.savez(f, X=X.to_numpy(float), t=D.t, **P)
    log(f"leak {variant} L={key_of(L)} done")
    return str(f)


def leak(lags, procs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(v, L) for L in lags for v in ("base", "junk", "late")]
    with Pool(procs) as pool:
        pool.map(leak_job, jobs, chunksize=1)
    out = {}
    nn = lambda a: np.nan_to_num(a, nan=-1.0)  # noqa: E731
    for L in lags:
        Z = {v: np.load(OUT_DIR / f"leak_{v}_{key_of(L)}.npz") for v in ("base", "junk", "late")}
        b, j, lt = Z["base"], Z["junk"], Z["late"]
        keys = [k for k in b.files if k not in ("X", "t")]
        early = b["t"] + L <= LEAK_C_DAY * 24
        r = {"C_day": LEAK_C_DAY, "tfm_n_estimators": 1, "tfm_refits_up_to_day": LEAK_TFM_TMAX_DAY,
             "lgb_n_estimators": 150, "n_wafers_D_le_C": int(early.sum()),
             "junk_testval_labels": {"X_identical": bool(np.array_equal(b["X"], j["X"], equal_nan=True)),
                                     "max_abs_pred_diff": {k: float(np.max(np.abs(nn(b[k]) - nn(j[k])))) for k in keys}},
             "late_train_labels": {"X_identical_D_le_C": bool(np.array_equal(b["X"][early], lt["X"][early],
                                                                              equal_nan=True))}}
        for k in keys:
            a, c = nn(b[k]), nn(lt[k])
            fin = np.isfinite(b[k]) & ~early
            r["late_train_labels"][k] = {"max_abs_diff_D_le_C": float(np.max(np.abs(a[early] - c[early]))),
                                         "frac_changed_D_gt_C": round(float(np.mean(np.abs(a[fin] - c[fin]) > 1e-9)), 3)}
        out[key_of(L)] = r
        log(f"leak L={key_of(L)}: {json.dumps(r)}")
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["leak_check"] = out
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False))


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "features", "members", "report", "leak"])
    ap.add_argument("--lags", default="24,4")
    ap.add_argument("--members", default=",".join(f2.LGB_MEMBERS), help="LightGBM 멤버 (쉼표)")
    ap.add_argument("--tfm", default="", help="TFM 멤버 (쉼표): tabdpt, tabpfn_c300, tabicl")
    ap.add_argument("--tfm-lags", default="", help="TFM 을 돌릴 L (기본 --lags 와 같음)")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    lags = [float(x) for x in a.lags.split(",") if x]
    if a.stage == "leak":
        leak(lags, a.procs)
        return
    if a.stage in ("all", "features", "members"):
        for L in lags:
            features(L, reuse=a.reuse)
    if a.stage in ("all", "members"):
        tfm_lags = [float(x) for x in a.tfm_lags.split(",") if x] or lags
        jobs = [(L, m, a.reuse) for L in tfm_lags for m in a.tfm.split(",") if m]
        jobs += [(L, m, a.reuse) for L in lags for m in a.members.split(",") if m]
        jobs.sort(key=lambda j: (f2.MEMBERS[j[1]]["kind"] == "lgb", f2.MEMBERS[j[1]].get("step", 24.0)))  # 긴 작업 먼저
        with Pool(a.procs) as pool:
            for name, L, sec in pool.imap_unordered(member_job, jobs):
                log(f"member {key_of(L)} {name}: {sec}")
    if a.stage in ("all", "report"):
        report(lags)


if __name__ == "__main__":
    main()
