# PHM 2018 Data Challenge – 간단한 베이스라인

[PHM Society 2018 Data Challenge](https://phmsociety.org/)(Ion Mill Etching 장비) 문제를 위한 간단한 파이썬 베이스라인입니다.
장비 센서 데이터로 아래 3가지 고장까지 남은 시간(TTF, Time To Failure)을 예측합니다.

| 고장 모드 | TTF 컬럼 |
|---|---|
| FlowCool Pressure Dropped Below Limit | `TTF_FlowCool Pressure Dropped Below Limit` |
| Flowcool Pressure Too High Check Flowcool Pump | `TTF_Flowcool Pressure Too High Check Flowcool Pump` |
| Flowcool leak | `TTF_Flowcool leak` |

## 방법

1. **윈도우 특징 추출** (`phm_data.py`): 원시 센서 데이터(17개 수치형 센서)를 30분 윈도우로 묶어
   윈도우마다 `mean / std / min / max / last` 통계를 계산합니다. 여기에 직전 24시간 평균 대비 상대 변화율(`_trend`)을 더해
   서서히 진행되는 열화를 잡아냅니다. 각 윈도우는 그 윈도우 끝까지의 데이터만 사용합니다.
2. **라벨**: 각 윈도우 끝 시점의 TTF(초). 마지막 고장 이후처럼 TTF를 알 수 없는 구간은 학습에서 제외합니다.
3. **모델** (`train.py`): 고장 모드마다 `HistGradientBoostingRegressor` 하나씩 학습합니다.
4. **검증**: 장비(tool) 단위 `GroupKFold` 교차검증으로 처음 보는 장비에 대한 MAE/RMSE를 계산하고,
   학습 데이터 중앙값만 예측하는 단순 기준과 비교합니다.

## 파일 구성

```
phm_data.py        데이터 로딩, 윈도우 특징/라벨 생성
train.py           교차검증 + 모델 학습/저장
predict.py         테스트 데이터 TTF 예측
make_synthetic.py  실제 데이터와 같은 형식의 합성 데이터 생성 (데모/테스트용)
tests/             pytest 테스트
```

## 빠른 시작 (합성 데이터)

실제 데이터가 없어도 합성 데이터로 전체 흐름을 바로 실행해 볼 수 있습니다.

```bash
pip install -r requirements.txt

python make_synthetic.py --out data/synthetic
python train.py --train-dir data/synthetic/train --max-ttf-hours 72
python predict.py --test-dir data/synthetic/test --model models/baseline.joblib
```

합성 데이터(장비 4대 × 30일)에서의 교차검증 결과 예시 (`--max-ttf-hours 72`):

| 고장 모드 | 모델 MAE | 중앙값 예측 MAE |
|---|---|---|
| FlowCool Pressure Dropped Below Limit | 13.4h | 19.7h |
| Flowcool Pressure Too High Check Flowcool Pump | 14.0h | 20.4h |
| Flowcool leak | 10.5h | 17.4h |

합성 데이터는 고장 전 1~3일 동안 FlowCool 압력/유량이 점점 벗어나도록 만든 것이라, 이 수치는 파이프라인이 동작하는지 확인하는 용도일 뿐
실제 데이터 성능을 뜻하지는 않습니다.

## 실제 데이터로 실행

PHM 2018 데이터를 받아 아래 구조로 두면 됩니다. `time` 컬럼과 TTF는 초 단위로 가정합니다.

```
data/phm2018/
├── train/
│   ├── 01_M01_DC_train.csv            센서 데이터
│   ├── ...
│   ├── train_ttf/01_M01_DC_train.csv  TTF (센서 파일과 같은 이름)
│   └── train_faults/...               고장 기록 (이 베이스라인에서는 사용하지 않음)
└── test/
    └── 01_M01_DC_test.csv
```

```bash
python train.py --train-dir data/phm2018/train --model-out models/phm2018.joblib
python predict.py --test-dir data/phm2018/test --model models/phm2018.joblib --out-dir predictions
```

TTF 파일이 다른 폴더에 있으면 `--ttf-dir`로 지정하세요. 주요 옵션:

- `--window-sec`: 윈도우 길이(초, 기본 1800)
- `--max-ttf-hours`: TTF 라벨 상한. 고장 징후가 없는 먼 미래는 예측이 어려우므로, 상한을 두면 고장 임박 구간에 학습이 집중됩니다.
- `--n-splits`: 교차검증 fold 수 (장비 수보다 크면 장비 수로 줄어듭니다)

예측 결과는 장비별 `predictions/<tool>_pred.csv`에 윈도우마다 저장되며(`time`은 윈도우의 마지막 샘플 시각), 마지막 행이 데이터 끝 시점의 예측입니다.

## 테스트

```bash
pip install pytest
python -m pytest
```

## 개선 아이디어

- `recipe`, `recipe_step`별로 센서 값을 나눠 통계 내기 (공정 단계마다 정상 범위가 다름)
- 여러 길이의 추세 윈도우, 이동 표준편차, 기울기 특징 추가
- 고장 기록(`train_faults`)을 이용한 "마지막 고장 이후 경과 시간" 특징
- TTF 로그 변환, 생존 분석(survival) 모델, LSTM/1D-CNN 같은 시계열 모델
