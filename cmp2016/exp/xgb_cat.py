"""xgb_cat: XGBoost · CatBoost 회귀기를 앙상블 멤버로 추가 (레짐별 / 전역 / compact 변형, CV 로 소폭 튜닝).

아이디어
- 기존 앙상블(LightGBM 2종, ExtraTrees, 릿지)에 다른 부스팅 구현을 멤버로 더한다. 반복 수는 고정하고
  조기 종료(held-out 평가)는 쓰지 않는다. 특징은 그대로(112 개)이고 모델만 추가한다.
- 탐색은 기본 4 멤버의 OOF/평가 예측을 캐시(cache_xgb_cat_base_s{seed}.npz)한 가짜 모델로 대체해 새 멤버만
  학습했다 (harness 의 CV 앙상블 수치와 정확히 같고, 최종 실행은 진짜 멤버로 다시 학습해 재현을 확인했다).

시도한 것 (5-fold CV, fold_seed=0, 기준선 앙상블 7.596; "단일" 은 그 멤버 하나의 CV, "+it" 은 기본 4 + 그 멤버 NNLS)
- XGBoost (hist, max_bin 64, 레짐별): d4 단일 7.843 (+it 7.584), d3 7.977 (7.579), d6 7.937 (7.596), d4 정규화 강화 7.905 (7.586),
  d4 compact 8.443 (7.596), d4 전역 8.146 (7.553). 6 개 전부 넣어도 7.544 (-0.052).
  → XGBoost 는 LightGBM 과 잔차 상관 0.97~0.98 로 거의 같은 모델이라 다양성이 없다 (compact·전역 변형만 0.93).
- CatBoost (대칭 트리, border_count 64, 반복 고정): d4 단일 7.864 (+it 7.516), d6 7.915 (7.541), d6 l2=10 7.899 (7.534),
  d6 compact 8.096 (7.544), d6 전역 8.802 (7.535). 5 개 전부 7.461 (-0.135).
  → 단일로는 LightGBM(7.718) 보다 못하지만 잔차 상관 0.88~0.95 로 더 다르고, 그 다양성이 앙상블을 낮춘다.
- 탐욕적 전진 선택 (XGB+Cat 11 후보): cat_d4 7.516 → +cat_d6_global 7.481 → +cat_d6_compact 7.463 → +xgb_d4_global 7.458 (미미) → 중단.
- 전역 CatBoost 반복 800→2000, d4 반복 800→1500 은 단일 CV 만 조금 나아지고 앙상블 이득은 잡음 수준이라 쓰지 않았다.
- 결론: 이득은 "더 좋은 단일 모델" 이 아니라 "다른 종류의 부스팅이 주는 다양성" 에서 온다. XGBoost 는 제외.

최종 (기본 4 + cat_d4 + cat_d6_global + cat_d6_compact, 가중치 lgb 0.37, lgb_small 0.00, extra_trees 0.05, ridge 0.04,
      cat_d4 0.24, cat_d6_global 0.13, cat_d6_compact 0.18)
- fold_seed=0: CV 앙상블 7.463 (기준선 7.596, -0.133), 공식 테스트 7.135 (6.969), 검증 6.874 (6.850)
- fold_seed=1: CV 앙상블 7.260 (같은 폴드 기준선 7.510, -0.250), 테스트 7.255, 검증 6.915
- CV 는 두 시드 모두 확실히 좋아지지만 홀드아웃(테스트·검증) 은 좋아지지 않았다 (424 행 MSE 의 잡음 범위 안이지만 주의).
- 실행 시간: 다른 실험과 CPU 를 나눈 상태에서 5.4 분 (CatBoost 멤버 3 개 약 2 분).

실행: python cmp2016/exp/xgb_cat.py                 # 최종 변형만 → cmp2016/exp/xgb_cat.json
      python cmp2016/exp/xgb_cat.py --all           # 탐색(스크리닝) 변형 전부 (기본 멤버 캐시 필요)
      python cmp2016/exp/xgb_cat.py --cache-base [--seed 1]   # 기본 멤버 OOF/평가 예측 캐시 (탐색용)
      python cmp2016/exp/xgb_cat.py --cached --seed 1         # 캐시된 기본 멤버 + 최종 새 멤버로 빠른 확인
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 다른 실험과 CPU 를 나눌 때 스핀 대기 낭비를 줄인다
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, get_table, run, summary  # noqa: E402
from cmp_data import TARGET  # noqa: E402

NJ = 4
OUT = HERE / "xgb_cat.json"


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ---------------------------------------------------------------------------------------------
# 기본 멤버 캐시: 특징이 같으므로 기본 4 멤버의 OOF/평가 예측은 고정 → 탐색 때는 캐시를 읽는 가짜 모델로 대체
# (harness 가 계산하는 CV 앙상블과 수치가 정확히 같다). 최종 실행은 진짜 멤버로 다시 학습한다.
# ---------------------------------------------------------------------------------------------
def cache_path(seed):
    return HERE / f"cache_xgb_cat_base_s{seed}.npz"


def cache_base(seed):
    res = run(models=BASE_MODELS, fold_seed=seed, label=f"base_s{seed}", keep_pred=True, verbose=True)
    n = len(get_table())
    arrs = {}
    for name in BASE_MODELS:
        full = np.full(n, np.nan)
        full[res["train_idx"]] = res["oof"][name]
        full[res["eval_idx"]] = res["pred"][name]
        arrs[name] = full
    np.savez(cache_path(seed), **arrs)
    print(f"[cache] {cache_path(seed)}  cv ensemble {res['cv']['ensemble']:.3f}")


class Cached:
    """캐시된 예측을 X.index 로 찾아 돌려주는 가짜 모델 (fit 은 아무것도 하지 않는다)."""

    def __init__(self, arr):
        self.arr = arr

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.arr[np.asarray(X.index)]


def cached_models(seed):
    d = np.load(cache_path(seed))
    return {name: {"make": (lambda a=d[name]: Cached(a)), "cols": (lambda cols: ["GROUP"]), "fillna": None,
                   "per_regime": False} for name in BASE_MODELS}


# ---------------------------------------------------------------------------------------------
# 새 멤버
# ---------------------------------------------------------------------------------------------
XGB_DEFAULT = dict(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.5,
                   reg_lambda=1.0, min_child_weight=5, tree_method="hist", max_bin=64, n_jobs=NJ, random_state=0)
# border_count 254(기본) → 64 로 줄이면 학습이 10 배 빨라지고 (행 800 개라 충분) 반복 수는 고정, 조기 종료 없음.
CAT_DEFAULT = dict(iterations=600, learning_rate=0.06, depth=6, l2_leaf_reg=3.0, rsm=0.5, border_count=64,
                   random_seed=0, thread_count=NJ, verbose=0, allow_writing_files=False, loss_function="RMSE")


def xgb_spec(cols="all", per_regime=True, **over):
    p = {**XGB_DEFAULT, **over}
    return {"make": lambda: xgb.XGBRegressor(**p), "cols": cols, "fillna": None, "per_regime": per_regime}


def cat_spec(cols="all", per_regime=True, **over):
    p = {**CAT_DEFAULT, **over}
    return {"make": lambda: CatBoostRegressor(**p), "cols": cols, "fillna": None, "per_regime": per_regime}


# 탐색 후보 (이름: 명세)
XGB_CANDS = {
    "xgb_d4": xgb_spec(),
    "xgb_d3": xgb_spec(max_depth=3, n_estimators=1200, learning_rate=0.04, min_child_weight=3),
    "xgb_d6": xgb_spec(max_depth=6, n_estimators=800, reg_lambda=5.0, min_child_weight=10),
    "xgb_d4_reg": xgb_spec(reg_lambda=5.0, min_child_weight=10, colsample_bytree=0.7),
    "xgb_d4_compact": xgb_spec(cols="compact", colsample_bytree=0.7),
    "xgb_d4_global": xgb_spec(per_regime=False, n_estimators=1200),
}
CAT_CANDS = {
    "cat_d6": cat_spec(),
    "cat_d4": cat_spec(depth=4, iterations=800, learning_rate=0.05),
    "cat_d6_l10": cat_spec(l2_leaf_reg=10.0, rsm=0.3),
    "cat_d6_compact": cat_spec(cols="compact", rsm=0.7),
    "cat_d6_global": cat_spec(per_regime=False, iterations=800),
}


# ---------------------------------------------------------------------------------------------
# 오프라인 부분집합 블렌드: harness 와 같은 NNLS (OOF 만 사용) 로 "기본 4 + 후보" 조합의 CV 를 본다.
# ---------------------------------------------------------------------------------------------
def blend(oof, y, names):
    A = np.column_stack([oof[n] for n in names])
    w, _ = nnls(A, y)
    w = w / w.sum()
    return mse(A @ w, y), dict(zip(names, map(float, w)))


def screen(cands, seed, label):
    models = {**cached_models(seed), **cands}
    res = run(models=models, fold_seed=seed, label=label, keep_pred=True)
    y = get_table()[TARGET].to_numpy()[res["train_idx"]]
    oof = {k: np.asarray(v) for k, v in res["oof"].items()}
    base = list(BASE_MODELS)
    b0, _ = blend(oof, y, base)
    print(f"\n[screen {label}] base4 cv {b0:.3f}")
    rows = []
    for c in cands:
        m1, w1 = blend(oof, y, base + [c])
        rows.append((c, res["cv"][c], m1, w1[c]))
    for c, single, m1, w in sorted(rows, key=lambda r: r[2]):
        print(f"  {c:16s} single {single:.3f}  base4+it {m1:.3f} (Δ{m1 - b0:+.3f})  weight {w:.2f}")
    m_all, w_all = blend(oof, y, base + list(cands))
    print(f"  base4+all        {m_all:.3f} (Δ{m_all - b0:+.3f})  " + ", ".join(f"{k} {v:.2f}" for k, v in w_all.items() if v > 0.005))
    # 탐욕적 전진 선택
    chosen, cur = [], b0
    pool = list(cands)
    while pool:
        best = min(pool, key=lambda c: blend(oof, y, base + chosen + [c])[0])
        m, _ = blend(oof, y, base + chosen + [best])
        if m > cur - 0.005:
            break
        chosen.append(best)
        cur = m
        pool.remove(best)
        print(f"  greedy + {best:16s} -> {cur:.3f}")
    np.savez(HERE / f"cache_xgb_cat_screen_{label}_s{seed}.npz", **oof, y=y)
    return res


# ---------------------------------------------------------------------------------------------
# 최종 변형 (탐색 후 확정)
# ---------------------------------------------------------------------------------------------
FINAL_NEW = {
    "cat_d4": CAT_CANDS["cat_d4"],                 # 레짐별, 전체 특징, 얕은 트리
    "cat_d6_global": CAT_CANDS["cat_d6_global"],   # 전역 (GROUP·STAGE_B·regime_mu 가 레짐 정보), 단일로는 약하지만 가장 다르다
    "cat_d6_compact": CAT_CANDS["cat_d6_compact"], # compact 특징 부분집합
}


def final_models():
    return {**BASE_MODELS, **FINAL_NEW}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="탐색(스크리닝) 변형 전부")
    p.add_argument("--cache-base", action="store_true")
    p.add_argument("--screen", choices=["xgb", "cat", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cached", action="store_true", help="최종 변형을 캐시된 기본 멤버로 빠르게 채점")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    if a.cache_base:
        cache_base(a.seed)
        return
    if a.screen or a.all:
        which = a.screen or "both"
        if which in ("xgb", "both"):
            screen(XGB_CANDS, a.seed, "xgb")
        if which in ("cat", "both"):
            screen(CAT_CANDS, a.seed, "cat")
        return

    t0 = time.time()
    models = {**cached_models(a.seed), **FINAL_NEW} if a.cached else final_models()
    res = run(models=models, fold_seed=a.seed, label="xgb_cat", out=None if a.cached else a.out)
    print(summary(res))
    print(f"[time] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
