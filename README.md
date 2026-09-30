# 반도체 장비 데이터 프로젝트

| 프로젝트 | 장비 | 과제 | 핵심 결과 |
|---|---|---|---|
| [이온 밀 FlowCool 고장 감지](#이온-밀-flowcool-고장-감지-phm-2018-data-challenge) (PHM 2018) | 이온 밀 식각 장비 | 헬륨 냉각 계통 고장 3종을 1시간 전에 감지 | 오경보 1%에서 무작위 대비 21~30배 감지 |
| [CMP 연마량 예측](cmp2016/README.md) (PHM 2016) | 웨이퍼 연마 장비(CMP) | 웨이퍼별 평균 연마량 예측 + 소모품 마모 해석 | 테스트 MSE 6.29 (대회 상위 5팀 최고 7.07; 대회 후 연구 6.33~6.36 과는 테스트 잡음 안의 차이). 표 파운데이션 모델(TabPFN v2·TabICLv2) 추가로 nested CV 6.61 → 6.31 (Built with PriorLabs-TabPFN). [공장 조건 시뮬레이션](cmp2016/FAB.md): 실시간 VM은 계측 지연 1시간에서 12.9(TabICLv2 멤버로 12.3), 연마 24시간 뒤 확정하는 2단계 VM은 배포 기간 테스트 6.99 / 검증 7.30(같은 기간 대회 방식 6.76 / 6.87) |

# 이온 밀 FlowCool 고장 감지 (PHM 2018 Data Challenge)

반도체 웨이퍼 식각 장비(Ion Mill Etch, Seagate 제공) 20대의 4초 간격 센서 로그로, 웨이퍼 냉각용 헬륨 계통(FlowCool)에서 생기는 세 가지 고장이 1시간 안에 일어날지를 예측한 개인 프로젝트입니다.

- 데이터: [PHM Society 2018 Data Challenge](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/) (압축 5.4GB, 풀면 34GB, 학습 8,219만 행)
- 결과 대시보드: `dashboard/index.html` (한 파일짜리 HTML, 브라우저로 열면 됩니다)

## 결과 요약

검증은 장비마다 앞 75% 기간으로 학습하고 뒤 25% 기간으로 했습니다. 대회 테스트 데이터가 같은 장비의 바로 다음 기간이라 같은 방식으로 나눴습니다.

| 고장 | AUC | 감지율 (오경보 1%) | 무작위 대비 |
|---|---|---|---|
| FlowCool Pressure Dropped Below Limit (압력 저하) | 0.850 | 21.4% | 21배 |
| Flowcool Pressure Too High Check Flowcool Pump (압력 과다) | 0.822 | 29.9% | 30배 |
| Flowcool leak (누설) | 0.881 | 30.1% | 30배 |

감지율은 정상 구간의 1%에서만 경보가 울리도록 기준을 잡았을 때 고장 1시간 이내 구간을 잡아낸 비율입니다. 무작위로 경보를 울리면 1%입니다.

### 대회 채점식에 대해

대회 최종 점수는 원래 점수(S1)와 2차 점수(S2)의 평균이었습니다(2018 PHM 학회 Data Challenge 논문 #589, #590, #591의 채점표). 2차 점수는 숫자로 답한 TTF의 오차를 제곱으로 벌하고, 정답이 있는데 NaN으로 답하면 20점만 벌합니다. 그래서 오차를 14초 안으로 보장하지 못하면 NaN이 항상 유리합니다. 기대 벌점을 최소화하는 결정 규칙은 모든 구간에서 NaN을 골랐고, 검증 점수는 전부 NaN 제출과 같습니다(최종 76.284). 대회 순위(1위 0.762)는 공개되지 않은 별도 검증 세트와 정규화로 매겨져 직접 비교할 수 없습니다.

## 방법

1. **변환** (`convert_rows.py`): 원본 tar.gz를 압축 해제 없이 스트리밍으로 읽어 장비별 parquet으로 저장합니다.
2. **행 단위 특징** (`row_features.py`): 과거 데이터만 쓰는 206개 특징을 만듭니다.
   - 1분~하루 롤링 통계 (FlowCool 압력·유량, 챔버 압력, 빔·억제 전류)
   - 레시피 단계별 정상값 대비 z (학습 기간 데이터로만 기준을 만듦)
   - 정상 구간에서 학습한 압력→유량 곡선의 잔차 (Hitachi 팀 논문 #590의 아이디어)
   - 압력·유량 이탈, 장비 정지 이후 경과 시간과 최근 발생 횟수 (같은 고장의 절반이 하루 안에 재발)
3. **모델** (`train_rows.py`): 남은 시간을 13개 구간으로 나눠 LightGBM 다중 분류로 확률을 예측합니다. 고장 3시간 이내 행은 2행마다 하나, 나머지는 2%만 뽑고 가중치로 보정합니다. 예측 확률과 채점식으로 기대 벌점이 가장 작은 답(숫자 또는 NaN)을 고릅니다.
4. **대시보드** (`build_dashboard.py`): 검증 결과를 요약해 `dashboard/index.html`을 만듭니다.

## 실행

```bash
pip install -r requirements.txt

# 1) 원본 데이터 받기 (대회 페이지의 Google Drive 링크, 5.4GB)
pip install gdown
gdown 15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb -O data/raw/phm_data_challenge_2018.tar.gz

# 2) 행 단위 parquet으로 변환 (약 10분, 3.2GB)
python convert_rows.py --archive data/raw/phm_data_challenge_2018.tar.gz --out data/rows

# 3) 학습 표본 만들기 (약 20분) -> 학습과 검증 (약 30분)
python train_rows.py sample --rows-dir data/rows --out data/samples/rows_v3.parquet
python train_rows.py fit --sample data/samples/rows_v3.parquet --save-proba data/val_proba_v5.npz

# 4) 대시보드 생성
python build_dashboard.py --sample data/samples/rows_v3.parquet --proba data/val_proba_v5.npz
```

메모리 15GB, 4코어 환경에서 확인했습니다.

## 파일 구성

```
convert_rows.py        원본 tar.gz -> 장비별 행 단위 parquet
row_features.py        행 단위 특징 (롤링 통계, 단계별 z, 압력->유량 잔차, 이벤트 경과 시간)
train_rows.py          표본 생성, LightGBM 학습, 채점식 기반 결정 규칙, 검증 점수
build_dashboard.py     대시보드 수치 요약과 HTML 생성
dashboard/             대시보드 템플릿, 요약 수치(data.json), 결과 페이지(index.html)
phm_data.py            공통 상수, 대회 채점식(S1, S2, 최종)

# 처음에 만든 30분 윈도우 베이스라인 (비교용)
prepare_features.py    원본 -> 30분 윈도우 특징
train.py, predict.py   윈도우 특징으로 TTF 회귀
make_synthetic.py      실제 데이터 없이 파이프라인을 시험하는 합성 데이터
tests/                 pytest 테스트
```

## 테스트

```bash
pip install pytest
python -m pytest
```
