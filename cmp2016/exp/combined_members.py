"""combined_members: 탐색에서 검증된 새 앙상블 멤버 8개의 명세를 한곳에 모은다 (import 부작용 없음).

멤버 (모두 harness.run 의 models 명세 형식 {"make", "cols", "fillna", "per_regime"})
- seqdl          seqdl.FINAL_NEW  : 원시 시퀀스 CNN + compact 표 특징 융합망 (전역, 레짐 중심화; 시퀀스는 seqdl.load_seq 캐시)
- mlp            mlp.FINAL_NEW    : 분위수 변환 MLP + 레짐 임베딩 (전역, compact)
- gp_state       residual_gp      : 레짐별 GP (시간 + 소모품 상태)
- lgb_global     gbm_tuning.FINAL : 전역 huber LightGBM (all)
- lgb_xt         gbm_tuning.FINAL : Optuna 2차 trial 4 LightGBM (레짐별, all)
- cat_d4, cat_d6_global, cat_d6_compact   xgb_cat.FINAL_NEW : CatBoost 3종

특징 표 호환: harness 가 어떤 특징표(기본 112 + PH_/TS_/NB_ 확장, ~700 컬럼)를 주더라도 동작한다.
- compact 를 쓰던 멤버(seqdl·mlp 의 표 가지, cat_d6_compact)는 callable 선택자 compact_cols 로 같은 접두사만 고른다
  (run_cmp.COMPACT_PREFIX; neighbors.py 의 NB_* 확장은 포함, PH_/TS_ 정적 특징은 제외).
- all 을 쓰던 멤버(lgb_global, lgb_xt, cat_d4, cat_d6_global)는 컬럼 전부를 본다 (LightGBM/CatBoost 는 700 컬럼도 문제없다).
  줄이고 싶으면 with_cols(spec, selector) 로 바꾼다.
- gp_state 는 residual_gp.pick(STATE) 그대로 (3개 사용량 컬럼 + X.index 로 표에서 찾는 시각).

스레드: OMP_WAIT_POLICY=PASSIVE, OMP_NUM_THREADS=4 (import 전에 설정), torch 2, BLAS 2 (threadpoolctl), LightGBM/CatBoost n_jobs 4.
make() 마다 enforce_threads() 를 다시 불러 다른 모듈(mlp.py 는 import 시 torch 1 스레드)이 바꾼 값을 되돌린다.
이 모듈을 lightgbm/torch/harness 보다 먼저 import 해야 환경변수가 OpenMP 런타임에 적용된다.

사용 예:
    from combined_members import MEMBERS, TINY, with_cols
    res = run(models={**BASE_MODELS, **MEMBERS}, static_fn=..., fold_seed=2)
    python cmp2016/exp/combined_members.py     # 스모크 테스트 (TINY 설정, 레짐 4A 폴드 0, ~700 컬럼 특징표)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_NUM_THREADS", "4")  # mlp.py 의 setdefault(1) 보다 먼저 (LightGBM 기본 스레드 수)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from run_cmp import COMPACT_PREFIX  # noqa: E402
import gbm_tuning  # noqa: E402
import mlp  # noqa: E402
import residual_gp  # noqa: E402
import seqdl  # noqa: E402
import xgb_cat  # noqa: E402

TORCH_THREADS, BLAS_THREADS, N_JOBS = 2, 2, 4
STATIC_PREFIX = ("PH_", "TS_")  # 정답 미사용 정적 특징: compact 멤버는 보지 않는다
BASE_NB = ("NB_time_", "NB_state_", "NB_trend_", "NB_slope_")  # cmp_data.neighbor_features 의 기본 NB_*
# 탐색에서 CV 이득이 테스트·검증으로 옮겨가지 않은 멤버 (nested CV 로 판단하되 먼저 빼 볼 후보)
DROP_FIRST = ["lgb_global", "lgb_xt", "cat_d4", "cat_d6_global", "cat_d6_compact", "gp_state"]


def enforce_threads():
    """torch 2, BLAS 2 스레드로 고정 (LightGBM/CatBoost 는 명세의 n_jobs/thread_count=4)."""
    if torch.get_num_threads() != TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    threadpool_limits(limits=BLAS_THREADS, user_api="blas")


enforce_threads()


# ----------------------------------------------------------------------------- 컬럼 선택자
def compact_cols(columns):
    """run_cmp.COMPACT_PREFIX 컬럼 (기본 compact + neighbors.py 의 NB_* 확장), PH_/TS_ 제외."""
    return [c for c in columns if c.startswith(COMPACT_PREFIX) and not c.startswith(STATIC_PREFIX)]


def compact_base_cols(columns):
    """기본 112 컬럼 기준의 compact 만 (neighbors.py 의 NB_E_/NB_D_/NB_O_ 확장도 제외)."""
    return [c for c in compact_cols(columns) if not c.startswith("NB_") or c.startswith(BASE_NB)]


def all_cols(columns):
    return list(columns)


def no_static_cols(columns):
    """PH_/TS_ 를 뺀 전부 (기본 + NB_ 확장)."""
    return [c for c in columns if not c.startswith(STATIC_PREFIX)]


def with_cols(spec, cols):
    """명세의 컬럼 선택자만 바꾼 사본."""
    return {**spec, "cols": cols}


# ----------------------------------------------------------------------------- 명세
def _spec(make, cols, per_regime, fillna=None):
    def mk():
        enforce_threads()
        return make()
    return {"make": mk, "cols": cols, "fillna": fillna, "per_regime": per_regime}


def _seqdl(tiny, tab_cols):
    base = seqdl.FINAL_NEW["seqdl"]
    p = base["make"]().p  # fusion, n_points 64, active_only False, center regime
    over = dict(max_epochs=8, min_epochs=1, patience=100, n_seeds=1, n_inner=1) if tiny else {}
    return _spec(lambda: seqdl.SeqDL(**{**p, **over}), tab_cols, base["per_regime"])


def _mlp(tiny, tab_cols):
    base = mlp.FINAL_NEW["mlp"]
    p = base["make"]().p  # quantile, embed, compact
    over = dict(max_epochs=8, min_epochs=1, n_seeds=1, n_inner=1) if tiny else {}
    return _spec(lambda: mlp.TorchMLP(**{**p, **over}), tab_cols, base["per_regime"])


def _gp(tiny):
    base = residual_gp.NEW_MODELS["gp_state"]
    g = base["make"]()
    kw = dict(rich=g.rich, n_restarts=g.n_restarts, time_bounds=g.time_bounds, maxiter=3 if tiny else g.maxiter,
              time_kernel=g.time_kernel, state_cols=g.state_cols, extra_cols=g.extra_cols, product=g.product)
    return _spec(lambda: residual_gp.GPMember(**kw), base["cols"], base["per_regime"])


def _lgb(name, tiny):
    base = gbm_tuning.FINAL[name]
    m = base["make"]()
    params = {**m.params, "n_jobs": N_JOBS, **({"n_estimators": 20} if tiny else {})}
    return _spec(lambda: gbm_tuning.BaggedLGB(seeds=m.seeds, monotone=m.monotone, **params), base["cols"],
                 base["per_regime"], base["fillna"])


def _cat(name, tiny, tab_cols):
    base = xgb_cat.FINAL_NEW[name]
    params = {**base["make"]().get_params(), "thread_count": N_JOBS, **({"iterations": 20} if tiny else {})}
    cols = tab_cols if base["cols"] == "compact" else base["cols"]
    return _spec(lambda: xgb_cat.CatBoostRegressor(**params), cols, base["per_regime"], base["fillna"])


def members(tiny=False, tab_cols=compact_cols):
    """멤버 명세 dict. tiny: 스모크 테스트용 (트리 20 그루, 에폭 8, GP 반복 3). tab_cols: compact 멤버의 컬럼 선택자."""
    return {
        "seqdl": _seqdl(tiny, tab_cols),
        "mlp": _mlp(tiny, tab_cols),
        "gp_state": _gp(tiny),
        "lgb_global": _lgb("lgb_global", tiny),
        "lgb_xt": _lgb("lgb_xt", tiny),
        "cat_d4": _cat("cat_d4", tiny, tab_cols),
        "cat_d6_global": _cat("cat_d6_global", tiny, tab_cols),
        "cat_d6_compact": _cat("cat_d6_compact", tiny, tab_cols),
    }


MEMBERS = members()
TINY = members(tiny=True)
NAMES = list(MEMBERS)


# ----------------------------------------------------------------------------- 스모크 테스트
def smoke_frame(table, ref_mask, cache_key="s0_k5_f0"):
    """기본 특징 + PH_(물리 캐시 187) + TS_(tsshape_features.txt, 캐시) + 가짜 NB_E_ 2개 → ~720 컬럼 특징표 (정답 미사용)."""
    from harness import base_features
    from cmp_data import KEY
    X = base_features(table, ref_mask, cache_key)
    f = HERE / "cache_physics.parquet"
    if f.exists():
        X = X.join(pd.read_parquet(f))
    f, names = HERE / "cache_tsshape.parquet", HERE / "tsshape_features.txt"
    if f.exists() and names.exists():
        ts = pd.read_parquet(f)
        want = [c for c in names.read_text().split() if c in ts.columns]
        ts = table[KEY].merge(ts[KEY + want], on=KEY, how="left").drop(columns=KEY).astype(float)
        ts.index, ts.columns = table.index, [f"TS_{c}" for c in want]
        X = X.join(ts)
    rng = np.random.RandomState(0)
    X["NB_E_dummy"] = rng.normal(size=len(X))
    X["NB_D_dummy"] = np.where(rng.rand(len(X)) < 0.3, np.nan, rng.normal(size=len(X)))
    return X


def smoke(regime_name="4A", fold_seed=0, specs=None):
    """TINY 멤버 전부를 한 레짐(폴드 0 학습 행 → 폴드 0 held-out + 테스트·검증 행)에서 fit/predict 한다."""
    from sklearn.model_selection import KFold
    from harness import fit_predict, get_table
    from cmp_data import TARGET, regime
    specs = TINY if specs is None else specs
    table = get_table()
    y, reg = table[TARGET].to_numpy(), regime(table)
    tr = np.flatnonzero(((table["split"] == "train") & ~table["OUTLIER"]).to_numpy())
    ev = np.flatnonzero(table["split"].isin(["test", "val"]).to_numpy())
    a, b = list(KFold(5, shuffle=True, random_state=fold_seed).split(tr))[0]
    ref = np.zeros(len(table), bool)
    ref[tr[a]] = True
    X = smoke_frame(table, ref, f"s{fold_seed}_k5_f0")
    train_idx = tr[a][reg[tr[a]] == regime_name]
    pred_idx = np.concatenate([tr[b][reg[tr[b]] == regime_name], ev[reg[ev] == regime_name]])
    n_hold = int((reg[tr[b]] == regime_name).sum())
    var = float(y[train_idx].var())
    print(f"[smoke] regime {regime_name}: X {X.shape}, train {len(train_idx)}, predict {len(pred_idx)} "
          f"(held-out {n_hold} + eval {len(pred_idx) - n_hold}), train var {var:.1f}")
    ok = True
    for name, spec in specs.items():
        t0 = time.time()
        p = fit_predict(spec, X, y, reg, train_idx, pred_idx)
        n_cols = len(spec["cols"](list(X.columns))) if callable(spec["cols"]) else spec["cols"]
        m_hold, m_ev = float(np.mean((p[:n_hold] - y[pred_idx[:n_hold]]) ** 2)), float(np.mean((p[n_hold:] - y[pred_idx[n_hold:]]) ** 2))
        good = np.isfinite(p).all() and len(p) == len(pred_idx) and m_hold < 3 * var
        ok &= bool(good)
        print(f"  {name:15s} cols {str(n_cols):>4s}  held-out mse {m_hold:8.3f}  eval mse {m_ev:8.3f}  "
              f"{'ok' if good else 'FAIL'}  ({time.time() - t0:.1f}s, torch {torch.get_num_threads()} threads)", flush=True)
    print("[smoke]", "all members fit/predict ok" if ok else "FAILURE")
    return ok


if __name__ == "__main__":
    smoke()
