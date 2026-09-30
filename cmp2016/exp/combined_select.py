"""combined_select: 최종 특징 층(combined_features) 위에서 앙상블 멤버를 nested CV 로 고르고 최종 수치를 낸다.

절차
1. 후보 멤버 전부의 OOF·평가 예측을 fold_seed=0 에서 만든다. 기준 4 멤버는 cache_combined_base_s0.npz,
   새 멤버(combined_members.MEMBERS)는 멤버 하나당 harness.run 한 번 → cache_combined_<멤버>_s0.npz
   (멤버별 별도 프로세스: 하나가 죽어도 나머지 캐시는 남는다; 4 CPU 에 맞춰 두 줄로 병행).
2. blend.greedy_forward: lgb 에서 시작해 nested 레짐별 클리핑 NNLS CV 를 가장 줄이는 멤버를 더하고, 이득 < 0.05 면 멈춤
   (최대 6 멤버). 고른 집합을 fold_seed=2 에서 확인: seed 2 nested 가 기준 4 멤버의 seed 2 nested 보다 나아야 하며,
   아니면 마지막에 더한 멤버부터 뺀다.
3. 집합이 정해진 뒤에만 테스트·검증을 본다: seed 0 OOF 로 맞춘 클리핑 범위·레짐별 NNLS 가중치를 평가 예측에 적용
   (전체·레짐별), in-sample CV 는 7.596 과의 비교용.
4. combined_select.json 에 선택표·가중치·수치 기록. --stage timing 은 고른 멤버 전부를 한 프로세스에서 처음부터
   학습해 벽시계 시간을 잰다 (run_cmp.py 통합 예산용).

결과 (tsshape lite + NB EDO 층, 580 특징; 기준 4 멤버 nested seed 0 6.770 / seed 2 6.900)
  탐욕 (seed 0 nested / seed 2 nested): lgb 7.013 / 6.988 → +seqdl 6.714 / 6.738 → +gp_state 6.608 / 6.624
  → 멈춤 (다음 최선 lgb_xt 6.569, 이득 0.039 < 0.05). lgb_small·extra_trees·ridge 도 0.05 미만이라 빠진다.
  최종 [lgb, seqdl, gp_state]: seed 0 nested 6.608 (in-sample 레짐별 클리핑 6.568, 전역 NNLS 6.693), 테스트 6.445, 검증 6.099
  (레짐별 테스트 1A 9.188 / 4A 5.558 / 4B 6.156, 검증 1A 6.781 / 4A 5.456 / 4B 6.525); seed 2 nested 6.624, 테스트 6.368, 검증 6.073.
  가중치(seed 0) 1A lgb 0.03 / seqdl 0.14 / gp_state 0.83, 4A 0.45 / 0.46 / 0.09, 4B 0.71 / 0.21 / 0.08.
  한 프로세스 처음부터 학습(--stage timing): 5.8 분 (특징 캐시 warm), 캐시 수치와 동일하게 재현.

실행
    python cmp2016/exp/combined_select.py --stage member --member seqdl --seed 0   # 멤버 하나 (캐시)
    python cmp2016/exp/combined_select.py --stage select                            # 1~4 (없는 캐시는 서브프로세스로 생성)
    python cmp2016/exp/combined_select.py --stage timing                            # 고른 집합 한 프로세스 재학습 시간

스레드: combined_members 를 먼저 import 해 OMP_WAIT_POLICY=PASSIVE / OMP_NUM_THREADS=4, torch 2, BLAS 2 를 맞춘다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import combined_members as cm  # noqa: E402  (lightgbm/torch/harness 보다 먼저)
import numpy as np  # noqa: E402

import blend  # noqa: E402
import combined_features as cf  # noqa: E402
from harness import get_table, run  # noqa: E402
from cmp_data import TARGET, regime  # noqa: E402

OUT_JSON = HERE / "combined_select.json"
BASE_NAMES = ["lgb", "lgb_small", "extra_trees", "ridge"]
MIN_GAIN, MAX_MEMBERS = 0.05, 6
CONFIRM_SEED = 2
# 병행 실행 계획 (4 CPU): seqdl(torch 2 스레드)은 홀로, 나머지(n_jobs 4)는 순서대로 한 줄
PARALLEL_GROUPS = [["seqdl"], ["lgb_global", "mlp", "cat_d6_global", "cat_d4", "gp_state", "lgb_xt", "cat_d6_compact"]]
SCRATCH = Path(os.environ.get("COMBINED_SELECT_LOGDIR", str(HERE)))


def cache_path(name, seed):
    return HERE / (f"cache_combined_base_s{seed}.npz" if name == "base" else f"cache_combined_{name}_s{seed}.npz")


# ----------------------------------------------------------------------------- 1. 멤버 캐시
def run_member(name, seed):
    """새 멤버 하나를 최종 특징 층에서 harness 로 학습해 npz 캐시로 저장한다 (verbose=False: 테스트·검증은 보지 않는다)."""
    spec = cm.MEMBERS[name]
    t0 = time.time()
    res = run(static_fn=cf.static_fn, feature_fn=cf.feature_fn, models={name: spec}, fold_seed=seed,
              label=f"combined_{name}_s{seed}", keep_pred=True, verbose=False)
    cf.save_base_cache(res, cache_path(name, seed))
    print(f"[member] {name} s{seed}: cv {res['cv'][name]:.3f}  n_features {res['n_features']}  "
          f"{time.time() - t0:.0f}s -> {cache_path(name, seed).name}", flush=True)
    return res


def ensure_members(names, seed, groups=PARALLEL_GROUPS):
    """캐시가 없는 멤버를 서브프로세스로 만든다 (groups 의 줄끼리 병행, 줄 안은 순차). 반환: 벽시계 초."""
    todo = [n for n in names if not cache_path(n, seed).exists()]
    if not todo:
        return 0.0
    t0 = time.time()
    lines = [[n for n in g if n in todo] for g in groups]
    lines = [g for g in lines if g] + [[n] for n in todo if not any(n in g for g in groups)]
    print(f"[ensure] seed {seed} 계산: {lines}", flush=True)
    procs = []
    for line in lines:
        cmd = " && ".join(f"{sys.executable} {__file__} --stage member --member {n} --seed {seed}" for n in line)
        log = SCRATCH / f"combined_select_{'_'.join(line)}_s{seed}.log"
        procs.append((line, subprocess.Popen(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
    for line, p, log in procs:
        rc = p.wait()
        print(f"[ensure] {line}: rc {rc} ({time.time() - t0:.0f}s, log {log})", flush=True)
    missing = [n for n in todo if not cache_path(n, seed).exists()]
    assert not missing, f"멤버 캐시 생성 실패: {missing}"
    return time.time() - t0


def load_members(names, seed):
    """기준 캐시 + 멤버 캐시를 합쳐 (oof {멤버: 배열}, pred {멤버: 배열}, meta) 로 돌려준다. 행 순서·폴드 일치를 확인."""
    base = cf.load_base_cache(cache_path("base", seed))
    oof, pred = dict(base["oof"]), dict(base["pred"])
    for n in names:
        if n in oof:
            continue
        z = cf.load_base_cache(cache_path(n, seed))
        assert (z["train_idx"] == base["train_idx"]).all() and (z["eval_idx"] == base["eval_idx"]).all(), n
        assert (z["fold_id"] == base["fold_id"]).all() and int(z["fold_seed"]) == int(base["fold_seed"]), n
        oof[n], pred[n] = z["oof"][n], z["pred"][n]
    meta = {k: base[k] for k in ["y_train", "y_eval", "regime_train", "regime_eval", "split_eval", "fold_id", "train_idx",
                                 "eval_idx"]}
    return oof, pred, meta


# ----------------------------------------------------------------------------- 2. 선택
def nested_of(names, oof, meta):
    return blend.mse(blend.nested_blend(oof, meta["y_train"], meta["regime_train"], meta["fold_id"], names=names),
                     meta["y_train"])


def select(seed0=0, seed1=CONFIRM_SEED):
    t_all = time.time()
    table = get_table()
    y, reg, split = table[TARGET].to_numpy(), regime(table), table["split"].to_numpy()
    out = {"layer": {"static": list(cf.STATIC_FAMILIES), "nb": True, "lgb": cf.FINAL_LGB}, "candidates": BASE_NAMES + cm.NAMES}

    # --- seed 0: 후보 전부
    sec0 = ensure_members(cm.NAMES, seed0)
    oof0, pred0, m0 = load_members(cm.NAMES, seed0)
    out["member_cv_s0"] = {n: round(blend.mse(oof0[n], m0["y_train"]), 3) for n in oof0}
    print("[members s0] " + ", ".join(f"{n} {v:.3f}" for n, v in out["member_cv_s0"].items()), flush=True)
    base_nested0 = nested_of(BASE_NAMES, oof0, m0)
    path = blend.greedy_forward(oof0, m0["y_train"], m0["regime_train"], m0["fold_id"], start=("lgb",), min_gain=MIN_GAIN)
    if len(path) > MAX_MEMBERS:
        print(f"[greedy] cap {MAX_MEMBERS} members", flush=True)
        path = path[:MAX_MEMBERS]
    chosen = [path[0][0]] + [n for n, _ in path[1:]]
    out["base_nested"] = {f"s{seed0}": round(base_nested0, 3)}
    out["greedy_s0"] = [{"member": n, "nested_s0": round(v, 3)} for n, v in path]

    # --- seed 2: 고른 집합 확인
    need = [n for n in chosen if n not in BASE_NAMES]
    sec2 = ensure_members(need, seed1) if need else 0.0
    oof2, pred2, m2 = load_members(need, seed1)
    base_nested2 = nested_of(BASE_NAMES, oof2, m2)
    out["base_nested"][f"s{seed1}"] = round(base_nested2, 3)
    for k, row in enumerate(out["greedy_s0"]):
        row[f"nested_s{seed1}"] = round(nested_of(chosen[: k + 1], oof2, m2), 3)
    print(f"[confirm] base-only nested s{seed0} {base_nested0:.3f} / s{seed1} {base_nested2:.3f}", flush=True)
    for row in out["greedy_s0"]:
        print(f"  {row['member']:>16s}  s{seed0} {row['nested_s0']:.3f}  s{seed1} {row[f'nested_s{seed1}']:.3f}", flush=True)
    dropped = []
    while len(chosen) > 1 and nested_of(chosen, oof2, m2) >= base_nested2:
        dropped.append(chosen.pop())
        print(f"[confirm] seed {seed1} nested 가 기준보다 나쁨 -> {dropped[-1]} 제외", flush=True)
    out["dropped_by_confirm"] = dropped
    out["members_kept"] = chosen

    # --- 3. 최종 수치 (여기서 처음 테스트·검증을 본다)
    final = {}
    for seed, (oof, pred, m) in {seed0: (oof0, pred0, m0), seed1: (oof2, pred2, m2)}.items():
        sub = {n: oof[n] for n in chosen}
        W, nested, ins = blend.clip_and_weights(sub, m["y_train"], m["regime_train"], m["fold_id"])
        _, _, ins_global = blend.clip_and_weights(sub, m["y_train"], m["regime_train"], m["fold_id"], per_regime=False,
                                                  clip=False)  # harness 의 cv.ensemble 과 같은 정의 (7.596 비교용)
        bounds = blend.clip_bounds(m["y_train"], m["regime_train"])
        p_ev = blend.apply({n: pred[n] for n in chosen}, m["regime_eval"], W, bounds)
        nb = blend.nested_blend(sub, m["y_train"], m["regime_train"], m["fold_id"])
        row = {"cv_nested": nested, "cv_nested_by_regime": blend.per_regime_mse(nb, m["y_train"], m["regime_train"]),
               "cv_in_sample_per_regime_clipped": ins, "cv_in_sample_global_nnls": ins_global,
               "weights": W, "clip_bounds": bounds, "member_cv": {n: blend.mse(oof[n], m["y_train"]) for n in chosen}}
        for s in ["test", "val"]:
            mk = m["split_eval"] == s
            row[s] = blend.per_regime_mse(p_ev[mk], m["y_eval"][mk], m["regime_eval"][mk])
        final[f"s{seed}"] = row
    out["final"] = final
    out["seconds"] = {"members_s0": round(sec0, 1), f"members_s{seed1}": round(sec2, 1), "select_total": round(time.time() - t_all, 1)}
    _round = lambda o: {k: _round(v) for k, v in o.items()} if isinstance(o, dict) else (round(o, 4) if isinstance(o, float) else o)  # noqa: E731
    out = _round(out)
    prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    if "timing" in prev:
        out["timing"] = prev["timing"]
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    f0 = final[f"s{seed0}"]
    print(f"[final] members {chosen}\n[final] s{seed0} nested {f0['cv_nested']:.3f} (in-sample per-regime clipped "
          f"{f0['cv_in_sample_per_regime_clipped']:.3f}, global NNLS {f0['cv_in_sample_global_nnls']:.3f})  "
          f"test {f0['test']['all']:.3f}  val {f0['val']['all']:.3f}", flush=True)
    for r, w in f0["weights"].items():
        print(f"[final] weights {r}: " + ", ".join(f"{n} {v:.2f}" for n, v in w.items()), flush=True)
    for s in ["test", "val"]:
        print(f"[final] {s} by regime: " + ", ".join(f"{k} {v:.3f}" for k, v in f0[s].items()), flush=True)
    f2 = final[f"s{seed1}"]
    print(f"[final] s{seed1} nested {f2['cv_nested']:.3f}  test {f2['test']['all']:.3f}  val {f2['val']['all']:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------- 4. 벽시계 (한 프로세스, 처음부터)
def timing(seed=0):
    """고른 멤버 전부를 한 프로세스에서 처음부터(특징 생성 포함) 학습해 벽시계 시간을 재고 nested CV 를 다시 확인한다."""
    sel = json.loads(OUT_JSON.read_text())
    chosen = sel["members_kept"]
    base = cf.models_with(cf.lgb_spec(**cf.TUNE[cf.FINAL_LGB]))
    models = {n: (base[n] if n in BASE_NAMES else cm.MEMBERS[n]) for n in chosen}  # 고른 멤버만 (기준 4 중 빠진 것은 학습하지 않는다)
    t0 = time.time()
    res = run(static_fn=cf.static_fn, feature_fn=cf.feature_fn, models=models, fold_seed=seed, label=f"timing_s{seed}",
              keep_pred=True, verbose=False)
    wall = time.time() - t0
    table = get_table()
    y, reg = table[TARGET].to_numpy(), regime(table)
    tr, ev = np.asarray(res["train_idx"]), np.asarray(res["eval_idx"])
    oof = {n: np.asarray(res["oof"][n]) for n in chosen}
    W, nested, ins = blend.clip_and_weights(oof, y[tr], reg[tr], seed)
    p_ev = blend.apply({n: np.asarray(res["pred"][n]) for n in chosen}, reg[ev], W, blend.clip_bounds(y[tr], reg[tr]))
    split = table["split"].to_numpy()[ev]
    row = {"seed": seed, "wall_seconds": round(wall, 1), "wall_minutes": round(wall / 60, 1), "members": list(models),
           "member_cv": {n: round(blend.mse(res["oof"][n], y[tr]), 3) for n in models},
           "cv_nested": round(nested, 3), "test": round(blend.mse(p_ev[split == "test"], y[ev][split == "test"]), 3),
           "val": round(blend.mse(p_ev[split == "val"], y[ev][split == "val"]), 3)}
    sel["timing"] = row
    OUT_JSON.write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print("[timing]", json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="combined ensemble selection")
    p.add_argument("--stage", choices=["member", "select", "timing"], default="select")
    p.add_argument("--member", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.stage == "member":
        run_member(a.member, a.seed)
    elif a.stage == "select":
        select()
    else:
        timing(a.seed)


if __name__ == "__main__":
    main()
