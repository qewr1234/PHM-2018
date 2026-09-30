# FAB 적용을 가정한 실시간 가상 계측 (PHM 2016 CMP)

대회용 모델(`run_cmp.py`, 테스트 MSE 6.45)은 시험 웨이퍼 **앞과 뒤**에 처리된 학습 웨이퍼의 정답을 함께 참고합니다. 공장에서는 불가능한 조건입니다. 공장에서는 이미 처리된 웨이퍼 중 **계측 결과가 도착한 것만** 볼 수 있고, 계측은 연마가 끝나고 한참 뒤에 도착하며, 모든 웨이퍼를 재지도 않습니다.

이 문서는 그 조건을 그대로 흉내 낸 스트리밍 시뮬레이터(`fab_vm.py`, `fab_sim.py`, `fab_run.py`)와 결과를 정리합니다.

## 한눈에 보기

| 조건 (배포 기간 30~65일, 1,582건) | MSE |
|---|---|
| 대회 방식 (앞·뒤 정답 사용, 인과적이지 않음) · 같은 기간 테스트/검증 웨이퍼 | 6.76 / 6.87 |
| **실시간 VM, 계측이 연마 직후 도착 (지연 0)** | **9.26** |
| 실시간 VM, 계측 지연 6분 / 15분 | 10.54 / 11.27 |
| **실시간 VM, 계측 지연 1시간 (기준 시나리오)** | **12.90** |
| 실시간 VM, 계측 지연 4시간 / 12시간 | 17.17 / 21.30 |
| 같은 조건(지연 1시간)에서 LightGBM 없이 EWMA·칼만만 | 15.78 |

- **공장에서 쓸 수 있는 성능은 대회 점수의 약 2배 오차입니다.** 가장 큰 원인은 모델이 아니라 **계측 지연**입니다. 같은 조건 웨이퍼가 3~4분 간격으로 연속 처리되는데, 가장 강한 단서인 "바로 앞 웨이퍼의 실제 연마량"을 지연 때문에 쓸 수 없게 되기 때문입니다.
- 그래도 이 웨이퍼의 센서 기록을 쓰는 LightGBM이 EWMA·칼만보다 오차를 18% 줄였습니다(15.78 → 12.90).
- 90% 예측 구간의 실제 적중률은 89.3%, 구간 반폭은 평균 ±5.8입니다(연마량 약 80 기준 약 7%).
- 예측 1건 3.8 ms, 하루 1번 재학습 2.9초(레짐 3개). 대회 파이프라인 전체 학습(약 7분)보다 훨씬 가볍습니다.

## 시뮬레이션 규칙

| 항목 | 설정 | 근거 |
|---|---|---|
| 순서 | 학습·테스트·검증 구분 없이 2,829건을 연마 종료 시각 순으로 흘려보냄 | Han 외 2025, River의 delayed progressive validation |
| 예측 시점 | vm: 연마 직후(이 웨이퍼 센서 기록 사용) / forecast: 연마 시작 전(소모품 카운터와 과거 계측만) | Cheng 외 2007(연마 직후 1단계 VM) |
| 계측 도착 | 연마 종료 + 지연(0~12시간). 도착 전 계측값은 어디에도 쓰지 않음 | Yi 외 2005(실제 CMP 라인: 1~2로트 지연) |
| 표본 계측 | 전수 / 무작위 / 조건별 N장마다 1장 / 불확실성 기반, 계측 비율 50·20·10% | Han 외 2025, Nduhura-Munga 외 2013 |
| 계측값 검증 | 같은 조건 과거 계측 중앙값의 3배를 넘으면 측정 오류로 버림(학습 데이터의 4,000대 값 4개) | Kadlec 외 2009(소프트 센서 데이터 품질) |
| 기간 분할 | 0~30일 개발(7일 워밍업 제외 후 채점으로만 설정 선택), 30~65일 배포(설정 고정, 보고만) | 시간 순 분할 |
| 누수 검사 | 컷오프 이후 정답을 바꿔도 컷오프 이전 예측이 한 자리도 바뀌지 않는지 테스트 (`tests/test_fab.py`) | |

배포 기간에도 모델은 도착하는 계측으로 계속 갱신됩니다(실제 운영과 같음). 바뀌지 않는 것은 하이퍼파라미터와 정책입니다.

## 구성

```
계측 도착 ─▶ VMService.on_metrology()  검증 → 이력 → EWMA·칼만 갱신 → 결합 가중치·예측 구간 갱신
매일 1회  ─▶ VMService.maybe_retrain()  도착한 계측으로 LightGBM 재학습
연마 종료 ─▶ VMService.on_wafer()       예측값, 90% 구간, 멤버별 예측, 신뢰 플래그
```

| 구성 요소 | 내용 |
|---|---|
| EWMA (`EWMAPredictor`) | run-to-run 제어의 표준 추정기. 수준 = λ·새 계측 + (1-λ)·이전 수준, λ = 0.5 |
| 칼만 동적 선형 모델 (`KalmanDLM`) | 연마량 = 수준 + b·(드레서 수명 위치, 가공 시간). 수준은 시간에 비례해 흔들리는 랜덤 워크라 오래 안 쟀을수록 불확실성이 커짐. 드레서 교체(사용량 카운터 초기화)를 감지하면 수준의 분산을 키움 |
| 주기 재학습 LightGBM (`OnlineGBM`) | 레짐별, 매일 재학습, gain 상위 120 특징. 입력 = 이 웨이퍼 센서 요약·TS_ 모양 특징 + **예측 당시 도착해 있던** 계측의 이력 특징(최근 편차, 지수가중 평균, 추세, 같은 드레서 수명 평균, 마지막 계측 웨이퍼 대비 상태 차이) + EWMA·칼만 예측 |
| 온라인 결합 (`OnlineBlend`) | 최근 150개 계측 웨이퍼에서 **그 당시 저장한** 멤버 예측으로 레짐별 NNLS 가중치, 과거 계측 범위로 자름 |
| 예측 구간 (`AdaptiveConformal`) | 적응형 컨포멀 추론(ACI): 계측이 도착해 구간 밖이었는지 확인할 때마다 α ← α + γ(0.1 − 벗어남), γ = 0.005 |
| 신뢰 플래그 | 센서 활성 구간 없음 / 마지막 계측 후 8시간 초과 / 멤버 불일치(표준편차 > 3) / 이력 부족 |

TS_ 모양 특징 목록(정답을 쓰지 않는 가지치기)도 **개발 기간 행만으로** 다시 정했습니다. 배포 기간 데이터는 특징 선택에도 쓰지 않았습니다.

### 개발 기간(7~30일)에서 고른 설정 (`fab_dev.json`)

vm 모드, 지연 1시간, 개발 기간 MSE:

| 비교 | MSE |
|---|---|
| EWMA λ 0.1 / 0.2 / 0.3 / **0.5** / 0.7 | 최저 13.80 (λ=0.5) |
| 칼만 (수준 잡음·계수 잡음·측정 잡음 72조합) | 최저 13.37 |
| **EWMA + 칼만 + LightGBM (최종)** | **11.02** |
| LightGBM에서 TS_ 모양 특징 빼기 | 11.97 |
| 최근 14일 창으로만 학습 | 12.16 |
| 레짐 공통 모델 | 12.16 |
| 칼만 예측과의 잔차를 학습 | 12.61 |
| 결합 창 75 / **150** / 300 (NNLS) | 11.47 / **11.02** / 11.37 |
| Di 외 2017의 1/오차³ 가중 결합 (창 150) | 11.17 |
| 온라인 가우시안 프로세스 멤버 추가 (Han 외 2025를 단순화, 이 실험만 LightGBM 없이) | 13.15 (추가 전 13.25), 13분 소요 → 채택 안 함 |

드레서 교체 시 칼만 분산 키우기는 개발 기간에 교체가 1~2번뿐이라 차이가 없었고(13.368 = 13.368), Han 외 2025가 보고한 "드레서 교체 직후 EWMA 추적 실패"를 근거로 켜 두었습니다.

## 결과 (배포 기간 30~65일)

### 1. 계측 지연

| 지연 | vm MSE (1A / 4A / 4B) | forecast MSE | 90% 구간 적중률 · 반폭 |
|---|---|---|---|
| 0 | 9.26 (8.85 / 7.81 / 10.76) | — | 89.2% · ±5.1 |
| 6분 | 10.54 | — | 88.8% · ±5.3 |
| 15분 | 11.27 | 13.61 | 89.0% · ±5.4 |
| **1시간** | **12.90** (10.53 / 11.29 / 15.28) | **15.41** | 89.3% · ±5.8 |
| 4시간 | 17.17 | 22.64 | 88.9% · ±6.8 |
| 12시간 | 21.30 | — | 88.9% · ±7.7 |

- 지연 6분만으로도 오차가 14% 커집니다. **통합 계측(연마 장비에 붙은 in-line 두께 측정)**이 VM 성능에 직접 연결된다는 뜻입니다.
- 구간 적중률은 지연과 상관없이 목표 90% 근처를 유지했습니다(ACI가 폭을 자동으로 넓힘).
- forecast(연마 전 예측)는 이 웨이퍼 센서 기록 없이 예측하므로 vm보다 20~30% 나쁩니다.

### 2. 계측 표본 정책 (지연 1시간, vm)

전체 웨이퍼(계측한 것 포함) VM 예측의 MSE:

| 계측 비율 | 무작위 | **조건별 N장마다 1장** | 불확실성 기반 |
|---|---|---|---|
| 100% | 12.90 | 12.90 | 12.90 |
| 50% | 15.40 | **14.34** | 15.08 |
| 20% | 17.52 | **15.70** | 19.80 |
| 10% | 19.29 | 19.40 | 24.94 |

개발 기간 비교(`fab_dev.json`의 `sampling_dev`)에서도 주기 계측이 모든 비율에서 가장 좋았습니다(20%: 주기 14.20, 무작위 15.36, 불확실성 16.89, 멤버 불일치만 23.18, 절반 주기 + 절반 불확실성 18.81).

**예상과 반대였던 점.** 불확실성이 큰 웨이퍼를 골라 재는 방식(Han 외 2025의 분산 기반 동적 표본을 변형)이 가장 나빴습니다. 불확실성이 큰 웨이퍼는 대개 **오래 쉰 뒤 처음 들어온 웨이퍼**였습니다(계측한 웨이퍼의 직전 계측 후 경과 시간 중앙값 13.1시간, 안 잰 웨이퍼 6.4시간). 쉬고 난 첫 웨이퍼는 로트 대표성이 낮아, 그 값으로 뒤따르는 웨이퍼를 보정하면 오히려 틀립니다. 그래서 이 데이터와 이 모델에서는 **"조건별로 N장마다 1장"을 권합니다.** Han 외 2025는 가우시안 프로세스의 예측 분산(입력 공간 거리 포함)을 썼고, 여기서는 칼만 분산(주로 경과 시간)을 썼다는 차이가 있습니다.

### 3. run-to-run: 연마 시간 결정

forecast 예측 f로 연마 시간을 "목표 제거량 / f"로 정하면, Preston 식에 따라 실제 제거량의 상대 오차는 y/f − 1입니다. 배포 기간 1,582장:

| 연마 시간을 정한 방식 | 전수 계측(지연 1시간): RMS 오차 · ±5% 안 비율 | 20% 주기 계측 |
|---|---|---|
| 고정 레시피 (개발 기간 조건별 평균) | 14.6% · 24.5% | 14.6% · 24.5% |
| EWMA (전통적 R2R) | 4.82% · 74.1% | 5.32% · 68.8% |
| 칼만 | 4.82% · 73.5% | 5.36% · 67.8% |
| LightGBM | 5.36% · 67.8% | 6.49% · 57.6% (n = 1,530) |
| **결합 (최종)** | **4.72% · 74.7%** | **5.30% · 68.6%** |

- 계측을 되먹이는 것만으로 제거량 오차가 14.6% → 4.8%로 1/3이 됩니다. 소모품 마모로 연마량이 계속 움직이기 때문입니다.
- 연마 **전** 예측에서는 ML이 EWMA를 거의 못 이깁니다(4.82% → 4.72%). 이 웨이퍼 센서 정보가 없으면 남는 단서는 과거 계측의 추세뿐이고, EWMA가 이미 그것을 잘 씁니다. ML의 가치는 연마 **후** 품질 추정(vm)에서 더 큽니다.

### 4. 신뢰 플래그 (지연 1시간, vm)

| 플래그 | 해당 비율 | 해당 웨이퍼 MSE | 나머지 MSE |
|---|---|---|---|
| 멤버 불일치 | 5.1% | 14.59 | 12.81 |
| 마지막 계측 후 8시간 초과 | 12.1% | 11.19 | 13.14 |
| 센서 활성 구간 없음 | 10.1% | 10.83 | 13.14 |

멤버 불일치만 실제로 큰 오차를 가려냈습니다. 나머지 두 플래그는 대부분 챔버 그룹 1(연마량이 안정적인 조건) 웨이퍼라 오히려 오차가 작았습니다. 운영에서는 **ACI 구간 폭과 멤버 불일치**를 신뢰도 지표로 쓰고, 나머지 두 플래그는 데이터 품질 알림으로만 쓰는 편이 맞습니다.

## FAB 관점에서 정리

1. **대회 점수는 공장 성능이 아닙니다.** 같은 기간 대회 방식 6.8 vs 실시간 12.9(지연 1시간). 공개 저장소들도 시간순 분할에서 오차가 크게 늘었습니다(jsw010204-web: MAE 2.80 → 5.28, kamalpraven: 2.58 → 4.38).
2. **계측 지연을 줄이는 것이 모델을 바꾸는 것보다 효과가 큽니다.** 지연 1시간 → 0이면 12.9 → 9.3. 어떤 모델 개선도 이만큼 줄이지 못했습니다.
3. **계측을 줄일 때는 "조건별 N장마다 1장"이 안전합니다.** 20% 계측(5장 중 1장)으로 15.7, 전수 계측 대비 오차 22% 증가로 계측 비용을 80% 줄일 수 있습니다.
4. **R2R 연마 시간 결정에는 EWMA가 이미 강합니다.** 고정 레시피 대비 효과의 대부분은 "계측을 되먹이는 것" 자체에서 나옵니다.
5. **불확실성은 적응형 컨포멀로 보장할 수 있습니다.** 지연·표본 조건이 바뀌어도 90% 목표 적중률이 유지됐습니다.

## 한계

- 계측 지연은 고정값으로 가정했습니다. 실제로는 계측 장비 대기열에 따라 달라집니다(Ai 외 2011은 확률적 지연을 다룸).
- run-to-run은 "연마량이 연마 시간과 무관하다"는 Preston 가정으로 계산했습니다. 실제 두께 목표·전 공정 두께 데이터는 없습니다.
- 챔버·레시피가 3가지뿐인 65일 데이터입니다. 신규 레시피, 장비 간 이동, 장기 드리프트는 확인하지 못했습니다.
- 대회 모델의 융합 신경망은 실시간 버전에 넣지 않았습니다(매일 재학습 비용). 주 1회 재학습으로 넣는 것은 다음 과제입니다.
- 이전에 보고한 "과거만 본 조건 8.56"(이전 버전 문서)은 특징만 과거로 제한하고 모델은 미래 학습 웨이퍼까지 포함해 학습한 값이라 완전히 인과적이지 않았습니다. 이 문서의 수치가 그것을 대체합니다.

## 실행

```bash
python cmp2016/fab_run.py --stage dev           # 개발 기간 설정 비교, 약 14분 → cmp2016/fab_dev.json
python cmp2016/fab_run.py --stage sampling_dev  # 계측 표본 정책 비교, 약 3분 → fab_dev.json 에 추가
python cmp2016/fab_run.py --stage main          # 배포 기간 시나리오 22개, 약 7분 → cmp2016/fab_results.json
python -m pytest tests/test_fab.py -q           # 누수·지연·검증·표본 비율 테스트
```

웨이퍼별 기록은 `data/cmp2016/fab/<시나리오>.parquet`에 저장됩니다(git 제외).

## 참고한 자료

조사는 별도로 원문·초록·DOI 기록을 직접 열어 확인했습니다. 원문을 읽지 못하고 서지 정보만 확인한 것은 "서지만"으로 표시했습니다. 전체 목록, 확인 수준, 각 자료에서 가져온 수치는 [`fab_references.md`](fab_references.md)(영문)에 있습니다. **코드는 어느 저장소에서도 가져오지 않았고 모두 직접 작성했습니다.** 아래는 설계와 해석에 실제로 쓴 것입니다.

### 설계에 직접 반영한 논문

| 자료 | 이 프로젝트에서 쓴 부분 |
|---|---|
| Han, Miller, Moyne, Vogl, Penkova, Jia. "A Comparative Study of Semiconductor Virtual Metrology Methods and Novel Algorithmic Framework for Dynamic Sampling." IEEE Trans. Semicond. Manuf. 38(2), 2025. [doi:10.1109/TSM.2025.3531920](https://doi.org/10.1109/TSM.2025.3531920) · [NIST 원문](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=958452) | 같은 PHM 2016 데이터를 시간순으로 흘리는 평가 방식, 칼만·EWMA 비교, 드레서 교체 후 EWMA 실패 → 칼만 분산 키우기, 분산 기반 동적 표본(재현해 보니 이 데이터·모델에서는 주기 계측이 나았음), 온라인 GP 시도 |
| Di, Jia, Lee. "Enhanced Virtual Metrology on Chemical Mechanical Planarization Process using an Integrated Model and Data-Driven Approach." IJPHM 8(2), 2017. [doi:10.36001/ijphm.2017.v8i2.2641](https://doi.org/10.36001/ijphm.2017.v8i2.2641) | 레시피(챔버 그룹 × 스테이지)별 모델, 직전 계측값 기준선, 시간 지연·이웃 특징을 "도착한 계측만"으로 바꿔 사용, 1/오차³ 가중 결합(개발 기간 비교에서 NNLS가 나아 NNLS 채택) |
| Yi, Sang, Zhao. "A run-to-run film thickness control of chemical-mechanical planarization processes." ACC 2005. [doi:10.1109/ACC.2005.1470643](https://doi.org/10.1109/ACC.2005.1470643) | 실제 CMP 라인 조건(로트당 1장 계측, 1~2로트 지연, 최적 λ ≈ 0.33) → 지연·표본 시나리오 범위와 EWMA λ 탐색 범위 |
| Gibbs, Candès. "Adaptive Conformal Inference Under Distribution Shift." NeurIPS 2021. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170) | 예측 구간 갱신식 α ← α + γ(α − 벗어남), γ = 0.005 |
| Cheng, Chen, Su, Zeng. "Evaluating Reliance Level of a Virtual Metrology System." IEEE TSM 21(1), 2008. [doi:10.1109/TSM.2007.914373](https://doi.org/10.1109/TSM.2007.914373) (서지만; 정의는 특허 [US7593912B2](https://patents.google.com/patent/US7593912B2/en)에서 확인) | 두 모델 예측의 차이로 신뢰도를 매기는 RI 아이디어 → 멤버 불일치 플래그 |
| Cheng, Huang, Kao. "Dual-Phase Virtual Metrology Scheme." IEEE TSM 20(4), 2007. [doi:10.1109/TSM.2007.907633](https://doi.org/10.1109/TSM.2007.907633) (서지만) | 연마 직후 즉시 내는 1단계 VM을 채점 기준으로 삼음 |
| Wan, McLoone. "Gaussian Process Regression for Virtual Metrology-Enabled Run-to-Run Control in Semiconductor Manufacturing." IEEE TSM 31(1), 2018. [doi:10.1109/TSM.2017.2768241](https://doi.org/10.1109/TSM.2017.2768241) | VM 불확실성을 R2R에 반영하는 틀, 연마 시간 = (목표 − 보정) / 예측 연마량 형태의 R2R 시뮬레이션 |
| Ingolfsson, Sachs. "Stability and Sensitivity of an EWMA Controller." J. Quality Technology 25(4), 1993. [doi:10.1080/00224065.1993.11979473](https://doi.org/10.1080/00224065.1993.11979473) (서지만) · Boning 외. "Run by run control of chemical-mechanical polishing." IEEE CPMT-C 19(4), 1996. [doi:10.1109/3476.558560](https://doi.org/10.1109/3476.558560) (서지만) | EWMA R2R 추정기, CMP 연마량 드리프트를 R2R로 보정한다는 기본 틀 |
| Wu, Lin, Wong, Jang, Tseng. "Performance Analysis of EWMA Controllers Subject to Metrology Delay." IEEE TSM 21(3), 2008. [doi:10.1109/TSM.2008.2001218](https://doi.org/10.1109/TSM.2008.2001218) (서지만) | 계측 지연을 핵심 시나리오 변수로 둔 근거 |
| Kadlec, Gabrys, Strandt. "Data-driven Soft Sensors in the process industry." Comput. Chem. Eng. 33(4), 2009. [doi:10.1016/j.compchemeng.2008.12.012](https://doi.org/10.1016/j.compchemeng.2008.12.012) · Kadlec, Grbić, Gabrys. "Review of adaptation mechanisms for data-driven soft sensors." 35(1), 2011. [doi:10.1016/j.compchemeng.2010.07.034](https://doi.org/10.1016/j.compchemeng.2010.07.034) (서지만) | 계측값 검증, 주기 재학습 vs 이동 창(14일 창이 더 나빴음) 비교 |
| Feng 외. "An Online Virtual Metrology Model With Sample Selection for the Tracking of Dynamic Manufacturing Processes With Slow Drift." IEEE TSM 32(4), 2019. [doi:10.1109/TSM.2019.2942768](https://doi.org/10.1109/TSM.2019.2942768) | 도착한 계측을 무조건 받지 않고 거르는 발상 → 여기서는 측정 오류 제거만 구현 |

### 비교·배경으로 쓴 논문

- Liu 외. "Predicting the Wafer Material Removal Rate for Semiconductor Chemical Mechanical Polishing Using a Fusion Network." Applied Sciences 12(22):11478, 2022. [doi:10.3390/app122211478](https://doi.org/10.3390/app122211478): 대회 방식 최고 6.36 (인과적이지 않은 비교 기준)
- Nduhura-Munga 외. "A Literature Review on Sampling Techniques in Semiconductor Manufacturing." IEEE TSM 26(2), 2013. [doi:10.1109/TSM.2013.2256943](https://doi.org/10.1109/TSM.2013.2256943): 정적·적응·동적 표본 분류
- Kurz, De Luca, Pilz. "A Sampling Decision System for Virtual Metrology in Semiconductor Manufacturing." IEEE T-ASE 12(1), 2015. [doi:10.1109/TASE.2014.2360214](https://doi.org/10.1109/TASE.2014.2360214) (서지만)
- Ai, Wong, Jang, Zheng. "Stability Analysis of EWMA Run-to-Run Controller Subjects to Stochastic Metrology Delay." IFAC 2011. [doi:10.3182/20110828-6-IT-1002.01041](https://doi.org/10.3182/20110828-6-IT-1002.01041) (서지만): 확률적 지연 (한계 항목)
- Preston. "The Theory and Design of Plate Glass Polishing Machines." J. Soc. Glass Technol. 11, 1927 (서지만) · Luo, Dornfeld. IEEE TSM 14(2), 2001. [doi:10.1109/66.920723](https://doi.org/10.1109/66.920723) (서지만): 제거량 ∝ 압력 × 속도 × 시간 (R2R 계산)

### 참고한 GitHub 저장소 (코드는 가져오지 않음)

| 저장소 | 참고한 점 |
|---|---|
| [jsw010204-web/cmp-metrology-allocation](https://github.com/jsw010204-web/cmp-metrology-allocation) | 같은 데이터로 무작위·정비 주기·시간순 분할을 비교(MAE 2.80 → 4.30 → 5.28)하고 계측 할당을 시뮬레이션. 공개 저장소 중 이 작업과 가장 가까움 |
| [kamalpraven/cmp-virtual-metrology](https://github.com/kamalpraven/cmp-virtual-metrology) | 시간순 "스트레스 분할"에서 오차 증가(MAE 2.58 → 4.38), 드레서 테이블 사용량의 시간 분포 변화 |
| [JamesLeeCY/semiconductor-quality-ml](https://github.com/JamesLeeCY/semiconductor-quality-ml) | 시작 시각 순 검증 분할 |
| [akangel0307/PHM-Data-Challenge](https://github.com/akangel0307/PHM-Data-Challenge) | 연마량 수준별 모드 분리 벤치마크 (오프라인) |
| [dasolma/phmd](https://github.com/dasolma/phmd) | PHM 데이터셋 접근 라이브러리 (원본 링크가 사라진 데이터의 대체 경로) |
| [online-ml/river](https://github.com/online-ml/river) | `progressive_val_score`의 `delay` 인자 = 이 시뮬레이터의 평가 방식(delayed progressive validation) |
| [aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series) | ACI 참조 구현 (여기서는 직접 구현) |

### Kaggle

- [markshizhe/cmp-data-set](https://www.kaggle.com/datasets/markshizhe/cmp-data-set): 테스트 웨이퍼 424건의 연마량 파일 하나. **공식 정답이 아닙니다.** 공식 정답(PHM16TestValidationAnswers)과 424행이 모두 다르고(서로 MSE 7.52), 어떤 모델의 예측값으로 보입니다. 이 파일로 채점한 공개 결과는 공식 정답 기준 점수와 비교할 수 없습니다. 예를 들어 JamesLeeCY 저장소의 XGBoost는 이 파일 기준 6.24, 공식 정답 기준 11.46입니다(저장소에 올라온 웨이퍼별 예측으로 다시 계산). 이 프로젝트는 공식 정답 파일(웹 아카이브)만 썼습니다.
- PHM 2016 CMP를 다룬 Kaggle 노트북은 찾지 못했습니다.

적응형 컨포멀 추론을 반도체 VM에 적용한 동료 심사 논문은 조사 범위에서 찾지 못했습니다.
