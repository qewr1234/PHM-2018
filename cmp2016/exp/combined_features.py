"""combined_features: 검증된 특징 아이디어(physics, tsshape, neighbors EDO)를 합친 특징 층과 그에 맞춘 lgb 멤버.

구성
- static_fn(table): 정답 미사용 정적 특징. physics PH_* (187, cache_physics.parquet) + tsshape lite TS_* (418,
  cache_tsshape.parquet + tsshape_features.txt). 캐시가 없을 때만 다시 계산한다. cols="all" 인 lgb 멤버만 본다.
- feature_fn(table, ref_mask): neighbors.make_feature_fn(("E","D","O"), "NB_") — 참조 폴드 정답으로 만드는 이웃 특징 50개
  (inner-fold 프로토콜, 모든 멤버가 본다).
- lgb_spec(**over): 넓어진 표에 맞춘 lgb 멤버 명세 (n_jobs=1: 레짐당 수백 행이라 스레드가 많을수록 오히려 느리다).
  prune=True 면 정답 없이 상수·중복(|corr|>0.98) 컬럼을 뺀 PH_/TS_ 목록만 쓰고, topk=K 면 fit 안에서 학습 행만으로
  gain 상위 K 컬럼을 고른 뒤 다시 학습한다 (폴드 안이라 정직).
- nested_cv(res, table, seed): harness.run(keep_pred=True) 결과에 stacking 의 클리핑 + 레짐별 NNLS 를 nested 5-fold 로 채점
  (헤드라인 지표; in-sample NNLS 는 기준 7.596 과의 비교용).

절제 (fold_seed=0, 기준 4 멤버; 기준 in-sample 7.596 / nested 7.543; 특징 수는 기본 112 포함)
  변형          특징   lgb    in-sample  nested   (테스트 / 검증은 마지막에 한 번만 확인)
  physics       299   7.241   7.144     7.131    6.514 / 6.419
  tsshape       530   7.241   7.135     7.016    6.619 / 6.384
  both          717   7.183   7.082     7.024    6.574 / 6.435
  nb (EDO)      162   7.536   7.427     7.297    6.739 / 6.636
  physics_nb    349   7.006   6.933     6.848    6.411 / 6.277
  tsshape_nb    580   7.074   6.945     6.803    6.502 / 6.261   ← 최종 층
  both_nb       767   6.947   6.844     6.769    6.461 / 6.209
  판단(nested, 0.05 문턱): nb 는 tsshape 위에서 -0.213, tsshape 는 physics+nb 위에서 -0.079 → 유지;
  physics 는 tsshape+nb 위에서 -0.034 (in-sample 로는 -0.101 로 보이지만 nested 로는 문턱 미달) → 제외.
  두 정적 가족은 같은 '단계 0 실제 연마 시간' 신호를 밀어 강하게 준가산적이다.
lgb 조정 (tsshape_nb 층, lgb 만 다시 학습, compact 멤버 OOF 재사용; nested seed 0 / seed 2)
  base 6.803 / 6.948, colsample 0.3 6.793, min_child 20 6.792, extra_trees 6.805, top100 6.781 / 6.892,
  top150 6.770 / 6.900, cs03+top150 6.784. 가지치기(prune)는 이 층에서 무변화(tsshape lite 목록이 이미 같은 규칙으로 가지치기됨).
  → top150 채택 (두 시드 모두 개선, 폭은 0.03~0.05 로 작다; lgb 학습 시간도 줄어든다).
최종 (tsshape lite + neighbors EDO, lgb=top150; 580 특징): seed 0 in-sample 6.897 / nested 6.770 (테스트 6.532, 검증 6.262),
  seed 2 in-sample 6.918 / nested 6.900 (테스트 6.581, 검증 6.269). 멤버 OOF·평가 예측은 cache_combined_base_s{0,2}.npz.

실행: python cmp2016/exp/combined_features.py --stage ablation --variant both_nb --seed 0
      python cmp2016/exp/combined_features.py --stage tune --variant prune_cs03 --seed 0   # lgb 만 다시 학습 (compact 멤버 재사용)
      python cmp2016/exp/combined_features.py --stage final --seed 2                        # 최종 층 전체 학습 + npz 캐시
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 병행 실행 시 OpenMP 스핀 대기 낭비를 줄인다
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, N_FOLDS, get_table, run  # noqa: E402
from cmp_data import KEY, TARGET, regime  # noqa: E402
from run_cmp import EXCLUDE, mse  # noqa: E402
from stacking import NNLSStack, clip_members, nested  # noqa: E402
import neighbors  # noqa: E402
import physics  # noqa: E402
import tsshape  # noqa: E402

TS_LIST = HERE / "tsshape_features.txt"
OUT_JSON = HERE / "combined_features.json"
N_JOBS = {"lgb": 1, "lgb_small": 1, "extra_trees": 2}
STATIC_FAMILIES = ("tsshape",)  # 최종 층: 절제 결과 physics 는 tsshape+nb 위에서 nested +0.034 < 0.05 라 제외
UNION_FAMILIES = ("physics", "tsshape")


# ----------------------------------------------------------------------------- 정적 특징 (정답 미사용)
def physics_static(table):
    """PH_* 187 컬럼 (캐시 없으면 physics.physics_static 이 원본 센서 행에서 다시 계산)."""
    return physics.physics_static(table)


def tsshape_static(table):
    """TS_* lite 418 컬럼: tsshape 캐시에서 tsshape_features.txt 목록만 골라 table 순서로 맞춘다."""
    df = tsshape.compute_all()  # 캐시 없으면 다시 계산 (약 1.5분)
    if TS_LIST.exists():
        names = TS_LIST.read_text().split()
    else:  # 목록이 없으면 tsshape 의 lite 정의(정답 미사용 가지치기 + 가족 선택)를 다시 만든다
        base = table[[c for c in table.columns if c not in EXCLUDE]]
        names = tsshape.select(tsshape.prune(tsshape.align(table, df), base), tsshape.VARIANTS["lite"]["families"])
        TS_LIST.write_text("\n".join(names) + "\n")
    out = tsshape.align(table, df[KEY + names]).astype(float)
    out.columns = [f"TS_{c}" for c in out.columns]
    return out


def make_static(families=STATIC_FAMILIES):
    """static_fn 팩토리: 요청한 가족만 붙인다. families 가 비면 None (정적 특징 없음)."""
    families = tuple(families)
    if not families:
        return None

    def fn(table):
        parts = []
        if "physics" in families:
            parts.append(physics_static(table))
        if "tsshape" in families:
            parts.append(tsshape_static(table))
        out = pd.concat(parts, axis=1)
        print(f"[static] {'+'.join(families)}: {out.shape[1]} columns", flush=True)
        return out
    return fn


def static_fn(table):
    """최종 정적 특징 (STATIC_FAMILIES): tsshape lite. 합집합은 make_static(UNION_FAMILIES)."""
    return make_static(STATIC_FAMILIES)(table)


# 이웃 특징 (참조 폴드 정답 사용, inner-fold 프로토콜): E(지수이동평균) D(상태 변화량) O(다른 스테이지)
feature_fn = neighbors.make_feature_fn(families=("E", "D", "O"), prefix="NB_")


def prune_static(table, static, max_corr=0.98):
    """정답 없이 가지치기: 정적 컬럼 중 상수(std<1e-9)·기존 표 컬럼이나 앞선 정적 컬럼과 |corr|>max_corr 인 것을 뺀 목록."""
    base = table[[c for c in table.columns if c not in EXCLUDE]]
    return tsshape.prune(static, base, min_notna=0.0, max_corr=max_corr)


_PRUNED = None


def pruned_columns(table=None):
    """physics+tsshape 정적 컬럼의 가지치기 결과 (한 번만 계산)."""
    global _PRUNED
    if _PRUNED is None:
        table = get_table() if table is None else table
        static = make_static()(table)
        keep = prune_static(table, static)
        print(f"[prune] static {static.shape[1]} -> {len(keep)} (label-free)", flush=True)
        _PRUNED = keep
    return _PRUNED


# ----------------------------------------------------------------------------- lgb 멤버
LGB_BASE = dict(n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=1)
TOPK_SCREEN = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=1)


class TopKLGB:
    """fit 안에서 학습 행만으로 gain 상위 topk 컬럼을 고른 뒤(짧은 lgb) 본 모델을 그 컬럼으로 학습한다. topk=None 이면 그냥 lgb."""

    def __init__(self, topk=None, **params):
        self.topk, self.params = topk, params

    def fit(self, X, y):
        self.cols = list(X.columns)
        if self.topk is not None and len(self.cols) > self.topk:
            g = pd.Series(lgb.LGBMRegressor(**TOPK_SCREEN).fit(X, y).booster_.feature_importance("gain"), index=self.cols)
            self.cols = list(g.sort_values(ascending=False).index[: self.topk])
        self.m = lgb.LGBMRegressor(**self.params).fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])


def _cols_pruned(cols):
    keep = set(pruned_columns())
    return [c for c in cols if not c.startswith(("PH_", "TS_")) or c in keep]


def lgb_spec(prune=False, topk=None, **over):
    """넓어진 표용 lgb 멤버 명세. prune: 가지치기한 정적 컬럼만, topk: fit 내부 gain 상위 K 선택, over: LightGBM 매개변수."""
    params = {**LGB_BASE, **over}
    return {"make": lambda: TopKLGB(topk=topk, **params), "cols": _cols_pruned if prune else "all",
            "fillna": None, "per_regime": True}


def _threaded(name, spec):
    nj = N_JOBS.get(name)
    if nj is None:
        return spec
    make = spec["make"]

    def mk():
        m = make()
        m.set_params(n_jobs=nj)
        return m
    return {**spec, "make": mk}


# 기준 멤버 (스레드 수만 고정; 결과는 같다)
MODELS = {k: _threaded(k, v) for k, v in BASE_MODELS.items()}


def models_with(spec):
    """기준 4 멤버에서 lgb 만 spec 으로 바꾼 모델 집합."""
    return {**MODELS, "lgb": spec}


# ----------------------------------------------------------------------------- nested 채점
def fold_ids(n_train, seed):
    fid = np.zeros(n_train, int)
    for k, (_, b) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train))):
        fid[b] = k
    return fid


def nested_cv(res, table=None, seed=None):
    """harness 결과(keep_pred=True) 위에서 멤버 클리핑 + 레짐별 NNLS 를 nested 5-fold 로 채점하고, 전체 OOF 로 맞춘
    가중치로 테스트·검증도 낸다 (보고용). 반환: cv_insample, cv_nested, weights(레짐별), test, val."""
    table = get_table() if table is None else table
    seed = res["fold_seed"] if seed is None else seed
    tr, ev = np.asarray(res["train_idx"]), np.asarray(res["eval_idx"])
    names = list(res["oof"])
    P, Pe = np.column_stack([res["oof"][k] for k in names]), np.column_stack([res["pred"][k] for k in names])
    y, reg = table[TARGET].to_numpy(), regime(table)
    ytr, R, Re = y[tr], reg[tr], reg[ev]
    fid = fold_ids(len(tr), seed)
    M = np.zeros((len(tr), 1))
    make = lambda: NNLSStack(per_regime=True)  # noqa: E731
    cv_nested = mse(nested(make, P, M, R, ytr, fid), ytr)
    st = make().fit(clip_members(P, R, ytr, R), M, R, ytr)
    p_ev = st.predict(clip_members(Pe, Re, ytr, R), np.zeros((len(ev), 1)), Re)
    split = table["split"].to_numpy()[ev]
    out = {"cv_insample": res["cv"]["ensemble"], "cv_nested": cv_nested,
           "cv_members": {k: res["cv"][k] for k in names},
           "weights": {r: dict(zip(names, map(float, w))) for r, w in st.w.items()}}
    for s in ["test", "val"]:
        m = split == s
        out[s] = {"mse": mse(p_ev[m], y[ev][m]),
                  **{f"mse_{r}": mse(p_ev[m & (Re == r)], y[ev][m & (Re == r)]) for r in np.unique(Re)}}
    return out


# ----------------------------------------------------------------------------- 캐시 (멤버 OOF·평가 예측)
def save_base_cache(res, path, table=None):
    """멤버별 OOF/평가 예측을 npz 로 저장한다 (Select 단계가 기본 멤버를 다시 학습하지 않도록)."""
    table = get_table() if table is None else table
    tr, ev = np.asarray(res["train_idx"]), np.asarray(res["eval_idx"])
    y, reg = table[TARGET].to_numpy(), regime(table)
    names = list(res["oof"])
    arrays = {f"oof_{k}": np.asarray(res["oof"][k]) for k in names}
    arrays.update({f"pred_{k}": np.asarray(res["pred"][k]) for k in names})
    np.savez(path, members=np.array(names), y_train=y[tr], y_eval=y[ev], train_idx=tr, eval_idx=ev, regime=reg,
             regime_train=reg[tr], regime_eval=reg[ev], split_eval=table["split"].to_numpy()[ev],
             fold_id=fold_ids(len(tr), res["fold_seed"]), fold_seed=res["fold_seed"], n_features=res["n_features"],
             label=res["label"], **arrays)


def load_base_cache(path):
    z = np.load(path, allow_pickle=True)
    names = list(z["members"])
    return {"members": names, "oof": {k: z[f"oof_{k}"] for k in names}, "pred": {k: z[f"pred_{k}"] for k in names},
            **{k: z[k] for k in z.files if not k.startswith(("oof_", "pred_"))}}


def cache_to_res(cache):
    """npz 캐시를 nested_cv 가 받는 res 형태로 되돌린다."""
    return {"oof": cache["oof"], "pred": cache["pred"], "train_idx": cache["train_idx"], "eval_idx": cache["eval_idx"],
            "fold_seed": int(cache["fold_seed"]), "n_features": int(cache["n_features"]), "label": str(cache["label"]),
            "cv": {"ensemble": float("nan"), **{k: float("nan") for k in cache["members"]}}}


# ----------------------------------------------------------------------------- 변형
ABLATION = {  # 1단계: 특징 가족 절제 (기준 lgb 설정)
    "physics": dict(static=("physics",), nb=False),
    "tsshape": dict(static=("tsshape",), nb=False),
    "both": dict(static=("physics", "tsshape"), nb=False),
    "both_nb": dict(static=("physics", "tsshape"), nb=True),
    "nb": dict(static=(), nb=True),
    "tsshape_nb": dict(static=("tsshape",), nb=True),   # physics 를 tsshape+nb 위에서 판단하기 위한 추가 칸
    "physics_nb": dict(static=("physics",), nb=True),
}
TUNE = {  # 2단계: 선택한 층 위에서 lgb 만 조정 (compact 멤버 OOF 는 절제 실행에서 재사용)
    # tsshape lite 목록은 이미 기존 표 기준 |corr|>0.98 가지치기를 거쳤으므로 prune=True 는 이 층에서 사실상 무변화
    "base": dict(),
    "cs03": dict(colsample_bytree=0.3),
    "mcs20": dict(min_child_samples=20),
    "xt": dict(extra_trees=True),
    "top150": dict(topk=150),
    "top100": dict(topk=100),
    "cs03_top150": dict(colsample_bytree=0.3, topk=150),
    "prune": dict(prune=True),
    "prune_cs03": dict(prune=True, colsample_bytree=0.3),
}
LAYER = dict(static=STATIC_FAMILIES, nb=True)  # 최종 특징 층: tsshape lite + neighbors EDO (절제 결과)
FINAL_LGB = "top150"  # 최종 lgb 설정: fit 안 gain 상위 150 컬럼 (seed 0 nested 6.803→6.770, seed 2 6.948→6.900)


def layer_name(layer=None):
    """LAYER 에 해당하는 ABLATION 변형 이름 (튜닝 단계가 재사용할 절제 캐시를 찾는 데 쓴다)."""
    layer = LAYER if layer is None else layer
    for k, v in ABLATION.items():
        if tuple(v["static"]) == tuple(layer["static"]) and v["nb"] == layer["nb"]:
            return k
    raise KeyError(layer)


def layer_fns(static=STATIC_FAMILIES, nb=True):
    return make_static(static), (feature_fn if nb else None)


def run_layer(name, seed, models=None, static=STATIC_FAMILIES, nb=True, cache=None):
    """특징 층 하나를 harness 로 채점하고 nested CV 를 더한다. cache 경로를 주면 npz 로 저장."""
    sfn, ffn = layer_fns(static, nb)
    res = run(static_fn=sfn, feature_fn=ffn, models=MODELS if models is None else models, fold_seed=seed,
              label=f"combined_{name}_s{seed}", keep_pred=True)
    st = nested_cv(res)
    if cache is not None:
        save_base_cache(res, cache)
    return res, st


def run_tune(name, seed, base_cache, static=STATIC_FAMILIES, nb=True):
    """lgb 멤버만 다시 학습해 base_cache 의 compact 멤버 OOF 와 합쳐 nested 채점한다."""
    sfn, ffn = layer_fns(static, nb)
    spec = lgb_spec(**TUNE[name])
    res = run(static_fn=sfn, feature_fn=ffn, models={"lgb": spec}, fold_seed=seed, label=f"tune_{name}_s{seed}",
              keep_pred=True)
    cache = load_base_cache(base_cache)
    merged = cache_to_res(cache)
    merged["oof"]["lgb"], merged["pred"]["lgb"] = np.asarray(res["oof"]["lgb"]), np.asarray(res["pred"]["lgb"])
    merged["cv"] = {k: mse(v, cache["y_train"]) for k, v in merged["oof"].items()}
    # in-sample 전역 NNLS (harness 와 같은 계산, 비교용)
    from scipy.optimize import nnls
    A = np.column_stack(list(merged["oof"].values()))
    w, _ = nnls(A, cache["y_train"])
    merged["cv"]["ensemble"] = mse(A @ (w / w.sum()), cache["y_train"])
    merged["n_features"] = res["n_features"]
    merged["label"] = res["label"]
    st = nested_cv(merged)
    st["lgb_seconds"] = res["seconds"]
    return merged, st


def summarize(name, st, n_features):
    return {"variant": name, "n_features": int(n_features), "cv_insample": round(st["cv_insample"], 3),
            "cv_nested": round(st["cv_nested"], 3), "lgb_cv": round(st["cv_members"]["lgb"], 3),
            "test": round(st["test"]["mse"], 3), "val": round(st["val"]["mse"], 3),
            "weights": {r: {k: round(v, 2) for k, v in w.items()} for r, w in st["weights"].items()}}


def main():
    p = argparse.ArgumentParser(description="combined feature layer")
    p.add_argument("--stage", choices=["ablation", "tune", "final"], default="final")
    p.add_argument("--variant", default="both_nb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(HERE))
    args = p.parse_args()
    outdir = Path(args.outdir)
    t0 = time.time()
    if args.stage == "ablation":
        v = ABLATION[args.variant]
        cache = outdir / f"cache_combined_abl_{args.variant}_s{args.seed}.npz"
        res, st = run_layer(args.variant, args.seed, static=v["static"], nb=v["nb"], cache=cache)
        row = summarize(args.variant, st, res["n_features"])
    elif args.stage == "tune":
        base = outdir / f"cache_combined_abl_{layer_name()}_s{args.seed}.npz"
        res, st = run_tune(args.variant, args.seed, base, static=LAYER["static"], nb=LAYER["nb"])
        save_base_cache(res, outdir / f"cache_combined_tune_{args.variant}_s{args.seed}.npz")
        row = summarize(args.variant, st, res["n_features"])
        row["lgb_seconds"] = st["lgb_seconds"]
    else:
        cache = HERE / f"cache_combined_base_s{args.seed}.npz"
        res, st = run_layer("final", args.seed, models=models_with(lgb_spec(**TUNE[FINAL_LGB])),
                            static=LAYER["static"], nb=LAYER["nb"], cache=cache)
        row = summarize("final", st, res["n_features"])
        row["lgb_setting"] = FINAL_LGB
    row["seed"], row["seconds"] = args.seed, round(time.time() - t0, 1)
    (outdir / f"combined_{args.stage}_{args.variant if args.stage != 'final' else 'final'}_s{args.seed}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=1))
    print("[result]", json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
