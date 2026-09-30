"""FAB 적용 시뮬레이션 실행: 개발 기간에서 설정을 고르고(--stage dev), 고정한 설정으로 배포 기간 시나리오를 돌린다(--stage main).

사용 예:
    python cmp2016/fab_run.py --stage dev     # 개발 기간(7~30일) 채점으로 하이퍼파라미터 비교 → cmp2016/fab_dev.json
    python cmp2016/fab_run.py --stage sampling_dev  # 계측 표본 정책 비교(개발 기간) → fab_dev.json 에 추가
    python cmp2016/fab_run.py --stage main    # 시나리오 22개 (CPU 4개 병렬, 약 10분) → cmp2016/fab_results.json
    python cmp2016/fab_run.py --stage tfm_dev --procs 2         # TFM 멤버 유무 비교(개발 기간) → fab_dev.json 'tfm_dev'
    python cmp2016/fab_run.py --stage tfm_main --procs 2 --reuse # TFM 멤버를 넣은 주요 시나리오 5개
                                                                 # → fab_results.json 'scenarios_tfm'·'r2r_tfm'
    (TFM 시뮬레이션 하나: 스레드 1개로 16~64분, 메모리 최대 약 4GB(관찰값) — 메모리가 모자라면 --procs 를 줄인다)

시나리오
- 계측 지연: 0, 0.1, 0.25, 1, 4, 12 시간 (전수 계측)                  → vm·forecast
- 표본 계측: 50·20·10% × 무작위·주기·불확실성 기반 (지연 1시간)       → vm
- run-to-run: forecast 예측으로 연마 시간을 정했을 때의 제거량 오차     → 전수 계측, 20% 표본
- TFM 멤버(선택): 칼만 잔차 TabICLv2 를 결합에 더한 설정(FINAL_CFG_TFM)으로 지연 0·1·4시간, 주기 20%(vm),
  forecast 지연 1시간. 기본 설정(FINAL_CFG)과 결과(scenarios)는 그대로 둔다.
--reuse: 같은 작업 설정·같은 시뮬레이션 코드로 이미 만든 기록(data/cmp2016/fab/*.parquet)이 있으면 다시 돌리지 않는다.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cmp_data import CACHE_DIR, build_table  # noqa: E402
from fab_sim import (DEFAULT_DELAY_H, DEV_END_DAY, coverage, period_mask, r2r_errors, replay_blend,  # noqa: E402
                     score, simulate)
from fab_vm import WaferStream  # noqa: E402

HERE = Path(__file__).resolve().parent
REC_DIR = CACHE_DIR / "fab"
# 개발 기간 비교(fab_dev.json)로 고른 설정. 바꾸면 --stage dev 결과와 맞는지 확인할 것.
FINAL_CFG = {
    "ewma": {"lam": 0.5},
    "kalman": {"q_level": 2.0, "q_beta": 0.0, "r_meas": 8.0, "reset_var": 25.0},
    "gbm": {"retrain_hours": 24.0, "topk": 120, "n_estimators": 400, "learning_rate": 0.04},
    "blend": {"window": 150, "method": "nnls"},
    "aci": {"alpha": 0.1, "gamma": 0.005},
}
# 선택 멤버 TFM 을 더한 설정 (--stage tfm_dev 의 개발 기간 비교로 채택). 느려서 기본 설정과 따로 둔다.
FINAL_CFG_TFM = {**FINAL_CFG, "tfm": {"retrain_hours": 24.0, "topk": 100, "n_estimators": 4, "kv_cache": True,
                                      "threads": 1}}
# 다른 단계가 쓴 키: 단계를 다시 돌려도 지우지 않는다
DEV_KEYS = ("sampling_dev", "tfm_dev")
MAIN_KEYS = ("config_tfm", "scenarios_tfm", "r2r_tfm")
# 기록 재사용 열쇠에 넣는 시뮬레이션 코드
_CODE = hashlib.sha1(b"".join((HERE / f).read_bytes() for f in ("fab_vm.py", "fab_sim.py", "cmp_data.py"))).hexdigest()

_S = {}


def _stream():
    if "s" not in _S:
        f = CACHE_DIR / "table.parquet"
        tab = pd.read_parquet(f) if f.exists() else build_table()
        s = WaferStream(tab, ts_dev_day=DEV_END_DAY)
        _S["s"] = s
        _S["static"] = {m: s.static_matrix(m) for m in ("vm", "forecast")}
    return _S["s"], _S["static"]


def _spec(job):
    _, mode, delay, policy, budget, cfg = job
    return json.dumps({"mode": mode, "delay": delay, "policy": policy, "budget": budget, "cfg": cfg, "code": _CODE},
                      sort_keys=True)


def _run(job, reuse=False):
    name, mode, delay, policy, budget, cfg = job
    f = REC_DIR / f"{name}.parquet"
    if reuse and f.exists():
        rec = pd.read_parquet(f)
        if rec.attrs.get("spec") == _spec(job):
            print(f"[reuse] {name}", flush=True)
            return name, rec
    s, static = _stream()
    st = static[mode]
    drop = cfg.get("_drop_prefix")
    if drop:
        st = st[[c for c in st.columns if not c.startswith(tuple(drop))]]
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    t0 = time.time()
    rec = simulate(s, mode=mode, delay_h=delay, policy=policy, budget=budget, cfg=cfg, static=st)
    rec.attrs["seconds"] = round(time.time() - t0, 1)
    rec.attrs["spec"] = _spec(job)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(f)
    return name, rec


def _run_star(a):
    return _run(*a)


def run_jobs(jobs, procs=4, reuse=False):
    out = {}
    with Pool(procs) as pool:
        for name, rec in pool.imap_unordered(_run_star, [(j, reuse) for j in jobs]):
            out[name] = rec
            print(f"[done] {name}: {rec.attrs['seconds']}s  dev {score(rec, 'dev')['mse']:.3f}", flush=True)
    return out


def r3(x):
    return float(np.round(x, 3)) if isinstance(x, (float, np.floating)) else x


def members(rec, period):
    return {c[2:]: r3(score(rec, period, c)["mse"]) for c in rec.columns if c.startswith("p_")}


# ==============================================================================================
def stage_dev(args):
    """개발 기간(WARMUP~DEV_END 일) 채점으로만 비교한다. 배포 기간 숫자는 계산하지 않는다."""
    s, _ = _stream()
    res = {"period": f"day 7-{DEV_END_DAY:g} (warm-up 7 days excluded)", "delay_h": DEFAULT_DELAY_H}
    # 1) EWMA λ, 칼만 잡음 (LightGBM 없이, 빠름)
    for mode in ("vm", "forecast"):
        rows = []
        for lam in (0.1, 0.2, 0.3, 0.5, 0.7):
            rec = simulate(s, mode=mode, cfg={"gbm": False, "ewma": {"lam": lam}})
            rows.append({"ewma_lam": lam, "dev_mse": r3(score(rec, "dev", "p_ewma")["mse"])})
        for ql, qb, rm, rv in itertools.product((0.05, 0.2, 0.5, 2.0), (0.0, 0.002, 0.02), (2.0, 4.0, 8.0), (0.0, 25.0)):
            rec = simulate(s, mode=mode, cfg={"gbm": False, "kalman": {"q_level": ql, "q_beta": qb, "r_meas": rm,
                                                                         "reset_var": rv}})
            rows.append({"kalman": [ql, qb, rm, rv], "dev_mse": r3(score(rec, "dev", "p_kalman")["mse"])})
        res[f"simple_{mode}"] = rows
        best_e = min((r for r in rows if "ewma_lam" in r), key=lambda r: r["dev_mse"])
        best_k = min((r for r in rows if "kalman" in r), key=lambda r: r["dev_mse"])
        print(f"[dev] {mode}: best EWMA {best_e}, best Kalman {best_k}", flush=True)
    # 2) LightGBM·결합 (vm, forecast)
    base = {k: v for k, v in FINAL_CFG.items()}
    jobs = [
        ("dev_vm_base", "vm", DEFAULT_DELAY_H, "all", 1.0, base),
        ("dev_vm_top60", "vm", DEFAULT_DELAY_H, "all", 1.0, {**base, "gbm": {**base["gbm"], "topk": 60}}),
        ("dev_vm_win14", "vm", DEFAULT_DELAY_H, "all", 1.0, {**base, "gbm": {**base["gbm"], "window_days": 14}}),
        ("dev_vm_noTS", "vm", DEFAULT_DELAY_H, "all", 1.0, {**base, "_drop_prefix": ["TS_"]}),
        ("dev_vm_resid", "vm", DEFAULT_DELAY_H, "all", 1.0, {**base, "gbm": {**base["gbm"], "residual": True}}),
        ("dev_vm_global", "vm", DEFAULT_DELAY_H, "all", 1.0, {**base, "gbm": {**base["gbm"], "per_regime": False}}),
        ("dev_fc_base", "forecast", DEFAULT_DELAY_H, "all", 1.0, base),
        ("dev_fc_nogbm", "forecast", DEFAULT_DELAY_H, "all", 1.0, {**base, "gbm": False}),
    ]
    recs = run_jobs(jobs, args.procs)
    res["gbm_grid"] = {k: {"blend": r3(score(v, "dev")["mse"]), "members": members(v, "dev"),
                           "fit_s": r3(v.attrs["gbm_fit_s"])} for k, v in sorted(recs.items())}
    # 3) 결합 방식 (멤버 예측은 그대로 두고 결합만 인과 순서로 다시 흉내)
    rec = recs["dev_vm_base"]
    res["blend_grid"] = {}
    for method, window in itertools.product(("nnls", "inv3"), (75, 150, 300)):
        p = replay_blend(s, rec, DEFAULT_DELAY_H, {"method": method, "window": window})
        m = period_mask(rec, "dev") & np.isfinite(p)
        res["blend_grid"][f"{method}_{window}"] = r3(float(np.mean((p[m] - rec["y"].to_numpy()[m]) ** 2)))
    print(json.dumps(res["gbm_grid"], indent=1), json.dumps(res["blend_grid"]), flush=True)
    f = Path(args.dev_out)
    prev = json.loads(f.read_text()) if f.exists() else {}
    res.update({k: prev[k] for k in DEV_KEYS if k in prev})
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[save] {args.dev_out}")


# ==============================================================================================
def stage_sampling_dev(args):
    """계측 표본 정책을 개발 기간 채점으로 비교해 fab_dev.json 에 덧붙인다 (지연 1시간, vm)."""
    jobs = [(f"devs_{pol}{int(b * 100)}", "vm", DEFAULT_DELAY_H, pol, b, FINAL_CFG)
            for b in (0.5, 0.2, 0.1) for pol in ("random", "periodic", "smart", "smart_spread", "hybrid")]
    recs = run_jobs(jobs, args.procs)
    f = Path(args.dev_out)
    res = json.loads(f.read_text()) if f.exists() else {}
    res["sampling_dev"] = {k: {"dev_mse": r3(score(v, "dev")["mse"]), "measured_rate": r3(float(v["measured"].mean()))}
                           for k, v in sorted(recs.items())}
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["sampling_dev"], indent=1))


def summarize(rec):
    out = {"deploy": {k: r3(v) for k, v in score(rec, "deploy").items()},
           "members_deploy": members(rec, "deploy"),
           "coverage_deploy": {k: r3(v) for k, v in coverage(rec, "deploy").items()},
           "measured_rate_deploy": r3(float(rec["measured"].to_numpy()[period_mask(rec, "deploy")].mean())),
           "latency_ms": r3(rec.attrs["latency_ms"]), "gbm_fit_s": r3(rec.attrs["gbm_fit_s"]),
           "gbm_fits": rec.attrs["gbm_fits"], "weights_end": rec.attrs["weights"], "seconds": rec.attrs["seconds"]}
    if "tfm_fits" in rec.attrs:
        out.update(tfm_fit_s=r3(rec.attrs["tfm_fit_s"]), tfm_fits=rec.attrs["tfm_fits"])
    unm = ~rec["measured"].to_numpy()
    if unm[period_mask(rec, "deploy")].any():
        out["deploy_unmeasured"] = {k: r3(v) for k, v in score(rec, "deploy", extra=unm).items()}
    for sp in ("test", "val"):
        m = (rec["split"] == sp).to_numpy()
        out[f"{sp}_deploy"] = r3(score(rec, "deploy", extra=m)["mse"])
        out[f"{sp}_after_warmup"] = r3(score(rec, "after_warmup", extra=m)["mse"])
    return out


def flag_table(rec):
    m = period_mask(rec, "deploy") & np.isfinite(rec["pred"].to_numpy())
    e2 = (rec["pred"] - rec["y"]).to_numpy() ** 2
    out = {}
    for f in [c for c in rec.columns if c.startswith("flag_")]:
        v = rec[f].to_numpy()
        out[f[5:]] = {"share": r3(float(v[m].mean())), "mse_flagged": r3(float(e2[m & v].mean())) if (m & v).any() else None,
                      "mse_clear": r3(float(e2[m & ~v].mean()))}
    anyf = rec[[c for c in rec.columns if c.startswith("flag_")]].any(axis=1).to_numpy()
    out["any"] = {"share": r3(float(anyf[m].mean())), "mse_flagged": r3(float(e2[m & anyf].mean())),
                  "mse_clear": r3(float(e2[m & ~anyf].mean()))}
    return out


def fixed_recipe(s):
    """run-to-run 기준: 개발 기간 끝에 고정한 레시피(레짐 평균 연마량)."""
    dev = (s.day < DEV_END_DAY) & ~s.outlier & np.isfinite(s.y)
    return pd.Series(s.y[dev]).groupby(s.reg[dev]).mean()


def stage_main(args):
    s, _ = _stream()
    cfg = FINAL_CFG
    jobs = []
    for d in (0.0, 0.1, 0.25, 1.0, 4.0, 12.0):
        jobs.append((f"vm_delay{d:g}", "vm", d, "all", 1.0, cfg))
    for d in (0.25, 1.0, 4.0):
        jobs.append((f"forecast_delay{d:g}", "forecast", d, "all", 1.0, cfg))
    for b, pol in itertools.product((0.5, 0.2, 0.1), ("random", "periodic", "smart")):
        jobs.append((f"vm_{pol}{int(b * 100)}", "vm", DEFAULT_DELAY_H, pol, b, cfg))
    for pol in ("periodic", "smart"):
        jobs.append((f"forecast_{pol}20", "forecast", DEFAULT_DELAY_H, pol, 0.2, cfg))
    jobs.append(("vm_delay1_nogbm", "vm", DEFAULT_DELAY_H, "all", 1.0, {**cfg, "gbm": False}))
    jobs.sort(key=lambda j: j[5].get("gbm") is False)  # 오래 걸리는 것부터
    recs = run_jobs(jobs, args.procs)

    fixed = fixed_recipe(s)
    out = {"config": cfg, "assumptions": {
        "dev_period_days": [0, DEV_END_DAY], "warmup_days": 7, "deploy_period_days": [DEV_END_DAY, float(s.day.max())],
        "default_delay_h": DEFAULT_DELAY_H, "prediction_time": {"vm": "연마 종료", "forecast": "연마 시작"},
        "n_deploy": int(period_mask(recs["vm_delay1"], "deploy").sum())},
        "contest_reference": {"test": 6.445, "val": 6.099, "note": "대회 방식(앞·뒤 학습 웨이퍼 정답 사용), 인과적이지 않음"},
        "scenarios": {k: summarize(v) for k, v in sorted(recs.items())}}
    out["flags_vm_delay1"] = flag_table(recs["vm_delay1"])
    r2r = {}
    for name in ("forecast_delay1", "forecast_periodic20", "forecast_smart20"):
        rec = recs[name].copy()
        rec["p_fixed"] = pd.Series(rec["reg"]).map(fixed).to_numpy()
        r2r[name] = {c: {k: r3(v) for k, v in r2r_errors(rec, "deploy", c).items()}
                     for c in ("p_fixed", "p_ewma", "p_kalman", "p_gbm", "pred")}
    out["r2r"] = r2r
    f = Path(args.out)
    prev = json.loads(f.read_text()) if f.exists() else {}
    out.update({k: prev[k] for k in MAIN_KEYS if k in prev})
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[save] {args.out}")
    for k, v in sorted(out["scenarios"].items()):
        print(f"{k:22s} deploy {v['deploy']['mse']:7.3f}  unmeasured {v.get('deploy_unmeasured', {}).get('mse', float('nan')):7.3f}"
              f"  rate {v['measured_rate_deploy']:.2f}  cov {v['coverage_deploy']['coverage']}  test {v['test_deploy']} val {v['val_deploy']}")


# ==============================================================================================
def stage_tfm_dev(args):
    """TFM 멤버(칼만 잔차 TabICLv2)를 넣을지 개발 기간(7~30일) 채점으로만 정한다 (vm, 지연 1시간, 전수 계측).
    TFM 쪽 기록(tfm_vm_delay1)은 --stage tfm_main 이 --reuse 로 다시 쓴다."""
    jobs = [("tfm_vm_delay1", "vm", DEFAULT_DELAY_H, "all", 1.0, FINAL_CFG_TFM),
            ("tfmdev_vm_delay1", "vm", DEFAULT_DELAY_H, "all", 1.0, FINAL_CFG)]
    recs = run_jobs(jobs, args.procs, args.reuse)

    def dev(rec):
        out = {"blend": {k: r3(v) for k, v in score(rec, "dev").items()}, "members": members(rec, "dev"),
               "latency_ms": r3(rec.attrs["latency_ms"]), "seconds": rec.attrs["seconds"]}
        if "tfm_fits" in rec.attrs:
            out.update(tfm_fit_s=r3(rec.attrs["tfm_fit_s"]), tfm_fits=rec.attrs["tfm_fits"])
        return out

    res = {"period": f"day 7-{DEV_END_DAY:g} (warm-up 7 days excluded)", "mode": "vm", "delay_h": DEFAULT_DELAY_H,
           "policy": "all", "tfm": FINAL_CFG_TFM["tfm"],
           "without_tfm": dev(recs["tfmdev_vm_delay1"]), "with_tfm": dev(recs["tfm_vm_delay1"])}
    f = Path(args.dev_out)
    out = json.loads(f.read_text()) if f.exists() else {}
    out["tfm_dev"] = res
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(res, indent=1))
    print(f"[save] {args.dev_out}")


def stage_tfm_main(args):
    """FINAL_CFG_TFM 으로 주요 시나리오를 돌려 fab_results.json 에 'scenarios_tfm'·'r2r_tfm' 으로 덧붙인다
    (키 이름은 'scenarios' 와 같게: vm_delay1 등). 기존 'scenarios'·'r2r' 는 건드리지 않는다."""
    s, _ = _stream()
    cfg = FINAL_CFG_TFM
    jobs = [("tfm_vm_delay0", "vm", 0.0, "all", 1.0, cfg),
            ("tfm_vm_delay4", "vm", 4.0, "all", 1.0, cfg),
            ("tfm_vm_periodic20", "vm", DEFAULT_DELAY_H, "periodic", 0.2, cfg),
            ("tfm_forecast_delay1", "forecast", DEFAULT_DELAY_H, "all", 1.0, cfg),
            ("tfm_vm_delay1", "vm", DEFAULT_DELAY_H, "all", 1.0, cfg)]  # 마지막: tfm_dev 기록을 --reuse 로 쓴다
    recs = {k[len("tfm_"):]: v for k, v in run_jobs(jobs, args.procs, args.reuse).items()}
    f = Path(args.out)
    out = json.loads(f.read_text())
    out["config_tfm"] = cfg
    out["scenarios_tfm"] = {k: summarize(v) for k, v in sorted(recs.items())}
    rec = recs["forecast_delay1"].copy()
    rec["p_fixed"] = pd.Series(rec["reg"]).map(fixed_recipe(s)).to_numpy()
    out["r2r_tfm"] = {"forecast_delay1": {c: {k: r3(v) for k, v in r2r_errors(rec, "deploy", c).items()}
                                          for c in ("p_fixed", "p_ewma", "p_kalman", "p_gbm", "p_tfm", "pred")}}
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[save] {args.out}")
    for k, v in out["scenarios_tfm"].items():
        b = out["scenarios"][k]
        print(f"{k:22s} deploy {b['deploy']['mse']:7.3f} -> {v['deploy']['mse']:7.3f}"
              f"  test {b['test_deploy']} -> {v['test_deploy']}  val {b['val_deploy']} -> {v['val_deploy']}"
              f"  members {v['members_deploy']}")


def main():
    p = argparse.ArgumentParser(description="FAB 적용 가정 실시간 VM 시뮬레이션")
    p.add_argument("--stage", choices=["dev", "sampling_dev", "main", "tfm_dev", "tfm_main"], default="main")
    p.add_argument("--out", default=str(HERE / "fab_results.json"))
    p.add_argument("--dev-out", default=str(HERE / "fab_dev.json"))
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--reuse", action="store_true", help="같은 설정·코드로 만든 기록이 있으면 다시 쓴다")
    args = p.parse_args()
    t0 = time.time()
    {"dev": stage_dev, "sampling_dev": stage_sampling_dev, "main": stage_main, "tfm_dev": stage_tfm_dev,
     "tfm_main": stage_tfm_main}[args.stage](args)
    print(f"[time] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
