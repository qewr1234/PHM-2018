"""GBM 멤버 쥐어짜기 실험 (gbm_tuning): LightGBM 초매개변수·목적함수·단조 제약·시드 배깅·전역 모델.

아이디어: 기존 앙상블(레짐별 lgb, lgb_small, extra_trees, ridge)의 그래디언트 부스팅 멤버를 튜닝하거나 변형해
멤버로 추가/교체한다. 탐색은 캐시된 seed 0 폴드 특징으로 후보 멤버의 OOF 를 만들어 기준 4멤버 OOF 와 NNLS 로
섞은 앙상블 CV 로만 판단했다 (harness.run 과 같은 계산, 테스트·검증은 보지 않음).

시도한 것 (seed 0 폴드, 기준 앙상블 CV 7.596; "추가" = 기준 4멤버 + 후보)
- (b) 전역 모델: per_regime=False (GROUP·STAGE_B 컬럼으로 레짐 구분), 기본 lgb 설정. 멤버 7.808, 추가 7.509 (Δ-0.087).
      huber(alpha 3) 전역 7.767 / 추가 7.499 (Δ-0.097)이 가장 좋았다 (alpha 2: 9.74 발산, alpha 5: 7.516, fair 7.531,
      num_leaves 31: 7.547). 레짐별 모델을 교체하면 오히려 나빠지고(7.598), 나란히 두어야 이득이 난다.
- (a) Optuna(TPE) 1차 10회, 목적 = 기준 4멤버 + 후보 앙상블 CV: 어느 후보도 멤버 단독으로 기준 lgb(7.718)를 이기지 못했다
      (최고 7.789). 최적은 extra_trees + 강한 정규화(trial 7, 멤버 7.828)로 추가 7.540 (Δ-0.056): 다양성 효과.
      2차 8회, 목적 = 기준 4멤버 + lgb_global + 후보: 최적 trial 4(얕은 extra_trees, 컬럼 23%, 멤버 7.898) → 7.438.
- (c) 목적함수: 레짐별 huber(alpha 3) 멤버 7.705 (lgb 보다 조금 낫지만) 추가 7.576 (Δ-0.020), fair 7.568 (Δ-0.028).
      단조 제약(monotone_constraints_method=advanced): 드레서 사용량 감소 제약 7.590 (Δ-0.006), 이웃 평균 증가 제약 7.592,
      둘 다 이득 없음.
- (d) 시드 배깅: lgb 두 번째 시드를 멤버로 추가 7.587 (Δ-0.009), 전역 huber 두 시드 평균은 7.438 → 7.430 (Δ-0.007). 잡음 수준.
- (e) goss 7.576 (Δ-0.020), dart(600그루, lr 0.05)는 예측이 발산해(멤버 MSE 4880) 가중치 0. extra_trees 만 켠 lgb 7.540.
- 세 번째 후보를 더 얹어도 ≤0.004 밖에 안 줄어 두 멤버만 추가했다.

최종: 기준 4멤버 + lgb_global(전역, huber alpha 3) + lgb_xt(Optuna 2차 trial 4).
- seed 0: CV 앙상블 7.446 (Δ-0.150, 기준 7.596), 테스트 7.008 (기준 6.969), 검증 6.876 (기준 6.850).
  가중치 lgb 0.24, lgb_small 0.08, extra_trees 0.04, ridge 0.02, lgb_global 0.36, lgb_xt 0.26.
- seed 1 확인: CV 앙상블 7.295 (Δ-0.301, 기준 seed 범위 7.51~7.67), 테스트 6.992, 검증 6.894 (병행 실행 시 593s).
- CV 이득은 문턱(0.10)을 넘지만 공개 테스트·검증에서는 확인되지 않는다 (둘 다 ±0.04 안에서 같거나 조금 나쁨).
  새 멤버 둘 다 홀드아웃에서 단독 성능이 기존 lgb 보다 낮다 (테스트 lgb_global 7.484, 검증 lgb_xt 7.659).

실행: python cmp2016/exp/gbm_tuning.py             # 최종 변형만 (혼자 약 3분, 병행 시 5~6분), gbm_tuning.json 저장
      python cmp2016/exp/gbm_tuning.py --seed 1    # fold_seed 1 확인
      python cmp2016/exp/gbm_tuning.py --all       # 탐색 변형들을 harness 로 채점
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # 다른 실험과 CPU 를 나눌 때 스핀 대기 낭비를 줄인다

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import BASE_MODELS, run, summary  # noqa: E402

NJ = 4
LGB_DEFAULT = dict(n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1)
# Optuna 1차(목적 = 기준 4멤버 + 후보 앙상블 CV, 10회) 최적 trial 7: extra_trees + 강한 정규화
OPT7 = dict(num_leaves=26, min_child_samples=21, learning_rate=0.0125, n_estimators=1300, colsample_bytree=0.31,
            subsample=0.95, reg_alpha=2.87, reg_lambda=1.27, min_split_gain=0.31, extra_trees=True, path_smooth=1.48)
# Optuna 2차(목적 = 기준 4멤버 + lgb_global + 후보 앙상블 CV, 8회) 최적 trial 4: 얕은 extra_trees, 적은 컬럼, 강한 정규화
OPT2_4 = dict(num_leaves=6, min_child_samples=19, learning_rate=0.0347, n_estimators=1400, colsample_bytree=0.235,
              subsample=0.768, reg_alpha=3.32, reg_lambda=5.15, min_split_gain=0.945, max_bin=127, extra_trees=True,
              path_smooth=3.98)


def lgb_params(**over):
    p = {**LGB_DEFAULT, "n_jobs": NJ}
    p.update(over)
    return p


class BaggedLGB:
    """같은 설정을 여러 시드로 학습해 평균 (시드 배깅). monotone 이 있으면 X.columns 로 단조 제약을 만든다."""

    def __init__(self, seeds=(0,), monotone=None, **params):
        self.seeds, self.monotone, self.params = seeds, monotone or {}, params

    def _constraints(self, columns):
        mc = []
        for c in columns:
            v = 0
            for prefix, sign in self.monotone.items():
                if c.startswith(prefix) and "dist" not in c and "slope" not in c:
                    v = sign
            mc.append(v)
        return mc

    def fit(self, X, y):
        extra = {}
        if self.monotone:
            extra = {"monotone_constraints": self._constraints(X.columns), "monotone_constraints_method": "advanced"}
        self.ms = [lgb.LGBMRegressor(**lgb_params(**self.params, random_state=s, **extra)).fit(X, y) for s in self.seeds]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)


def spec(cols="all", per_regime=True, fillna=None, seeds=(0,), monotone=None, **params):
    return {"make": lambda: BaggedLGB(seeds=seeds, monotone=monotone, **params), "cols": cols, "fillna": fillna,
            "per_regime": per_regime}


# 최종: 기준 4멤버 + 전역(레짐 공통, GROUP/STAGE_B 컬럼으로 구분) huber LightGBM + Optuna 후보(extra_trees)
FINAL = {
    **BASE_MODELS,
    "lgb_global": spec(per_regime=False, objective="huber", alpha=3.0),
    "lgb_xt": spec(**OPT2_4),
}

VARIANTS = {
    "final": lambda: FINAL,
    "global_only": lambda: {**BASE_MODELS, "lgb_global": spec(per_regime=False, objective="huber", alpha=3.0)},
    "global_l2": lambda: {**BASE_MODELS, "lgb_global": spec(per_regime=False)},
    "xt_only": lambda: {**BASE_MODELS, "lgb_xt": spec(**OPT2_4)},
    "xt7": lambda: {**BASE_MODELS, "lgb_global": spec(per_regime=False, objective="huber", alpha=3.0), "lgb_xt": spec(**OPT7)},
    "huber": lambda: {**BASE_MODELS, "lgb_huber": spec(objective="huber", alpha=3.0)},
    "mono": lambda: {**BASE_MODELS, "lgb_mono": spec(monotone={"USAGE_OF_DRESSER_start": -1, "NB_time_": 1, "NB_trend": 1, "NB_state_": 1})},
    "bag3": lambda: {**BASE_MODELS, "lgb_bag3": spec(seeds=(0, 1, 2))},
    "goss": lambda: {**BASE_MODELS, "lgb_goss": spec(boosting_type="goss", subsample=1.0, subsample_freq=0)},
    "dart": lambda: {**BASE_MODELS, "lgb_dart": spec(boosting_type="dart", n_estimators=600, learning_rate=0.05)},
}


def main():
    p = argparse.ArgumentParser(description="GBM 멤버 튜닝 실험")
    p.add_argument("--all", action="store_true", help="탐색 변형들을 전부 harness 로 채점")
    p.add_argument("--variant", default="final")
    p.add_argument("--seed", type=int, default=0, help="fold_seed (확인용 1)")
    args = p.parse_args()
    names = list(VARIANTS) if args.all else [args.variant]
    for name in names:
        out = str(HERE / "gbm_tuning.json") if (name == "final" and args.seed == 0) else None
        res = run(models=VARIANTS[name](), fold_seed=args.seed, label=f"gbm_tuning_{name}_s{args.seed}", out=out)
        print(summary(res), flush=True)


if __name__ == "__main__":
    main()
