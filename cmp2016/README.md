# CMP 연마량 예측 (PHM 2016 Data Challenge)

웨이퍼 연마 장비(CMP, Chemical Mechanical Planarization)의 센서 로그와 소모품 사용량으로 웨이퍼마다 평균 연마량(Material Removal Rate, MRR)을 예측하고, 드레서·패드 마모가 연마 성능에 주는 영향을 물리 모델로 해석한 프로젝트입니다.

- 데이터: [PHM Society 2016 Data Challenge](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/) (학습 웨이퍼·스테이지 1,981건, 테스트 424건, 검증 424건, 정답 공개)
- 결과 대시보드: `cmp2016/dashboard/index.html`

## 결과

대회가 공개한 테스트·검증 정답으로 채점했습니다(MSE, 낮을수록 좋음). 모델 선택과 앙상블 가중치는 학습 데이터 5-fold 교차검증으로만 정했습니다.

| 모델 | 테스트 MSE | 검증 MSE |
|---|---|---|
| 챔버·스테이지 평균 | 51.57 | 59.55 |
| 물리 모델 (소모품 상태 → Preston 계수) | 44.10 | 46.82 |
| 시간상 가까운 웨이퍼 3장 평균 | 8.90 | 10.18 |
| LightGBM 단일 | 7.28 | 6.82 |
| **앙상블 (최종)** | **6.97** | **6.85** |

### 대회 참가팀·논문과 비교 (테스트 세트)

공식 순위는 검증 세트 MSE(90%)와 물리 설명(10%)을 합친 점수로 매겼지만 상위 팀의 검증 MSE는 공개되지 않아, 공개된 테스트 MSE로 비교합니다.

| 출처 | 테스트 MSE |
|---|---|
| 융합 신경망, 대회 후 연구 ([Appl. Sci. 2022](https://www.mdpi.com/2076-3417/12/22/11478)) | 6.36 |
| **이 프로젝트** | **6.97** |
| 대회 상위 5팀 중 최고 ([Di 외, IJPHM 2017](https://papers.phmsociety.org/index.php/ijphm/article/view/2641)) | 7.07 |
| 대회 상위 5팀 2~5위 (같은 논문) | 7.4, 7.4, 7.4, 7.5 |
| 칭화대 팀, 대회 2위 자체 보고 ([ICEEA 2018](https://www.atlantis-press.com/article/25894228.pdf)) | 7.4 |
| 직전 웨이퍼 연마량 그대로 (Di 외 기준선) | 8.23 |

## 설비 관점에서 읽은 것

- **드레서가 연마 성능의 핵심 레버입니다.** 챔버 4에서 드레서가 닳을수록 연마량이 약 11~14% 줄었습니다. 드레서가 패드를 거칠게 되살리지 못해 패드가 매끈해지면(글레이징) Preston 계수가 떨어지는 것과 맞습니다.
- **공정 설정보다 소모품 상태가 차이를 만듭니다.** 레시피가 압력과 회전 속도를 거의 고정해 P·V는 웨이퍼마다 비슷했습니다.
- **소모품 카운터만으로는 부족합니다.** 물리 모델은 평균 예측 대비 테스트 MSE를 51.6 → 44.1로 줄이는 데 그쳤고, 가까운 시각에 처리된 웨이퍼의 측정 결과를 반영하는 것이 훨씬 효과적이었습니다(run-to-run 제어와 같은 발상).
- **같은 챔버 그룹의 A·B 스테이지는 함께 움직입니다.** 다른 스테이지의 최근 결과를 넣자 교차검증 MSE가 8.11 → 7.78로 줄었습니다.

## 방법

1. **특징** (`cmp_data.py`): 웨이퍼·스테이지마다 챔버 3곳의 단계별 압력·회전·슬러리 요약, 소모품 6종 사용량, 압력 × 회전 속도의 시간 적분(Preston 항)
2. **이웃·순서 특징**: 시간·소모품 상태가 가까운 학습 웨이퍼의 연마량, 국소 추세, 처리 순서상 앞뒤 웨이퍼(같은 챔버 그룹의 다른 스테이지 포함)의 연마량 편차. 학습 행은 자기 자신을 빼고, 교차검증에서는 학습 폴드 안에서만 찾습니다.
3. **모델** (`run_cmp.py`): 연마 조건(챔버 그룹 + 스테이지)마다 LightGBM 2종, ExtraTrees, 릿지를 학습하고 교차검증 예측으로 정한 비음수 가중치로 섞습니다. 해석용으로 소모품 상태에 따라 Preston 계수가 변하는 릿지 모델을 따로 맞춥니다.
4. 학습 데이터의 연마량 수천 단위 값 4개는 측정 오류로 보고 뺐습니다(테스트·검증 최댓값 164).

## 실행

원래 대회 링크는 사라져서 웹 아카이브에서 받습니다.

```bash
mkdir -p data/cmp2016/raw && cd data/cmp2016/raw
A=https://web.archive.org/web
F=https://www.phmsociety.org/sites/phmsociety.org/files
curl -L -C - -o train_test.zip "$A/20210224223550id_/$F/2016%20PHM%20DATA%20CHALLENGE%20CMP%20DATA%20SET.zip"
curl -L -C - -o validation.zip "$A/20210224234703id_/$F/2016%20PHM%20DATA%20CHALLENGE%20CMP%20VALIDATION%20DATA%20SET.zip"
curl -L -C - -o answers.zip "$A/20210224234745id_/$F/PHM16TestValidationAnswers.zip"
for z in train_test validation answers; do unzip -oq $z.zip -d $z; done
cd ../../..

python cmp2016/run_cmp.py              # 약 2분, cmp2016/results.json
python cmp2016/build_cmp_dashboard.py  # cmp2016/dashboard/index.html
```

다운로드가 중간에 끊기면 같은 명령을 다시 실행하면 이어받습니다.
