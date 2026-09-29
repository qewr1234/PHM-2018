# References for a fab-realistic (causal, online) VM system on the PHM 2016 CMP data

Compiled 2026-09-29. I checked every entry myself in this session. Verification levels:

- **yes (full text)**: I downloaded or opened the paper, repo README or doc and read the relevant parts.
- **yes (abstract)**: I fetched the abstract from the landing page, the Semantic Scholar API, the arXiv API, PMC or a repository record.
- **yes (DOI record)**: I fetched the Crossref record, so the title, authors, venue and DOI are confirmed. Any statement about the paper's *content* is then marked **content: snippet only**.
- **snippet only**: I saw only a search-engine snippet.

Many IEEE and Elsevier landing pages returned 403 or empty pages to the fetch tool. For those papers I verified the metadata through Crossref and say so.

---

## 0. The dataset and challenge

| Item | Details |
|---|---|
| PHM Society 2016 Data Challenge (CMP) page | https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/ (HTTP 200). **verified: yes (page resolves)** |
| Official Call for Participation PDF | https://phmsociety.org/wp-content/uploads/2016/05/PHM16DataChallengeCFP.pdf. **verified: yes (full text)** |

**Key facts (from the CFP).** Teams predicted `AVG_REMOVAL_RATE` for each (WAFER_ID, STAGE). Scoring during the contest was MSE. The final score was 90 % MSE on a validation set posted a few weeks before the close, plus 10 % for the physics-based modelling write-up: dresser condition effect 3 %, pad condition effect 3 %, other parameters 4 %. The original Dropbox data link in the CFP is dead.

**How it applies.** The contest split interleaves test and validation wafers in time with the training wafers. Any "neighbour" or "time-lag" feature built on it can see future measurements. Report contest-split MSE only as a comparison point, and report a separate causal protocol as the fab-realistic number.

---

## 1. Papers on the PHM 2016 CMP data

### 1.1 Di, Jia & Lee (2017): winning method
- **Title:** Enhanced Virtual Metrology on Chemical Mechanical Planarization Process using an Integrated Model and Data-Driven Approach
- **Authors:** Yuan Di, Xiaodong Jia, Jay Lee (Univ. of Cincinnati, IMS)
- **Venue / year:** International Journal of Prognostics and Health Management 8(2), 2017
- **URL / DOI:** https://papers.phmsociety.org/index.php/ijphm/article/view/2641 · https://doi.org/10.36001/ijphm.2017.v8i2.2641
- **Verified:** yes (full text, PDF read)
- **Key idea:**
  - Features: Preston-equation physics (ARR = K·P·V), a dressing-rate term simplified to K_D·U_D (dresser usage), 11 removal-rate time lags, and the MRRs of the 10 nearest training wafers in consumable-usage space ("usage nearest neighbours").
  - Models: five models (persistent r_t = r_{t−1}, KNN, linear regression, tree bagging, SVR), fitted per recipe. There are three conditions set by chamber and stage.
  - Integration: models are combined with weights w ∝ 1/e³, where e = mean + 3·std of the Monte Carlo CV error over 20 repeats.
  - Results: CV MSE overall is 6.18 for the integrated model and 8.78 for the persistent model. Liu et al. 2022 cite this method at 7.07 test MSE.
- **Causal?** No. Neighbour and lag features come from the training set, which is interleaved in time with the test wafers, so both past and future measurements are used. Evaluation is Monte Carlo CV plus the contest test set.
- **How it applies:**
  - Keep the per-recipe split (chamber group × stage).
  - Keep the persistent model as the mandatory causal baseline.
  - Keep usage-KNN and time-lag features, but compute them only from wafers whose metrology has already *arrived* (after the delay) at prediction time.

### 1.2 Jia, Di, Feng, Yang, Dai & Lee (2018): adaptive VM with GMDH
- **Title:** Adaptive virtual metrology for semiconductor chemical mechanical planarization process using GMDH-type polynomial neural networks
- **Venue / year:** Journal of Process Control 62:44–54, 2018
- **DOI:** https://doi.org/10.1016/j.jprocont.2017.12.004
- **Verified:** yes (DOI record). Content: snippet only (ScienceDirect returned 403).
- **Key idea:**
  - Per the search snippets: a GMDH polynomial network with AICC as the external criterion does automatic feature and model-complexity selection, and adds two new feature types.
  - It is validated on PHM 2016. Liu et al. 2022 report its test MSE as 6.62.
- **Causal?** Not verified. The comparison tables in later papers imply the contest split.
- **How it applies:**
  - "Adaptive" here means automatic structure selection, not online updating.
  - Useful as an example of low-capacity models with automatic complexity control. That matters when the model has to be refit on few, delayed labels.

### 1.3 Feng, Jia, Zhu, Moyne, Iskandar & Lee (2019): online Bayesian ARX with sample selection
- **Title:** An Online Virtual Metrology Model With Sample Selection for the Tracking of Dynamic Manufacturing Processes With Slow Drift
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 32(4):574–582, 2019
- **DOI:** https://doi.org/10.1109/TSM.2019.2942768
- **Verified:** yes (abstract, via the Semantic Scholar API)
- **Key idea:**
  - An online Bayesian ARX model with time-varying parameters, updated by Bayes' rule.
  - A "Sample Importance" test admits a new measured sample into the model only when it matters, judged by freshness, prediction error and prediction uncertainty. The same test prunes the offline database.
  - Validated on PHM 2016. It reportedly beats JIT and deep-learning models.
- **Causal?** Yes. It is explicitly an online, tracking model. The exact split was not read.
- **How it applies:** This is a direct template for a streaming update rule. Update the model only with arrived measurements that pass an importance test, which keeps compute and the risk of overfitting drift noise low.

### 1.4 Han, Miller, Moyne, Vogl, Penkova & Jia (2025): online GP with dynamic sampling on PHM 2016 CMP (most relevant)
- **Title:** A Comparative Study of Semiconductor Virtual Metrology Methods and Novel Algorithmic Framework for Dynamic Sampling
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 38(2):232–239, 2025
- **DOI / URL:** https://doi.org/10.1109/TSM.2025.3531920 · NIST record: https://www.nist.gov/publications/comparative-study-semiconductor-virtual-metrology-methods-and-novel-algorithmic · full-text PDF: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=958452
- **Verified:** yes (full text)
- **Key idea:**
  - Compares MLR, EWMA, ARX, KF, Bayesian ARX and a dual linear KF against a proposed **online Gaussian process (OGP)** on the PHM 2016 CMP data (two data groups).
  - The paper stresses three needs:
    1. A "dynamic term", the last measurement y_{t−1}. Adjacent MRRs are highly correlated (their Fig. 2).
    2. Tracking drift and shift. Dresser and dresser-table usage drift, then reset at replacement.
    3. Fixed-rate versus dynamic sampling.
  - EWMA fails to track after the dresser replacement at run 144 unless its weight is re-tuned, and it gives no uncertainty.
  - Dynamic sampling: measure a wafer only when its predictive variance exceeds T, the t-th percentile of predictive variance. T is initialised by leave-one-out on the training set. t = 0 is the same as fixed-rate sampling. **t = 0.8 was the best cost/accuracy trade-off.**
  - Fixed-rate sampling needed 88 and 92 measurements on the two data groups. With dynamic sampling, OGP matched or beat that with fewer or equal samples.
  - BARX suffers "vanishing variance": it becomes over-confident as samples accumulate. GP kernels avoid this.
  - Kernel hyperparameters can be refit per batch rather than per sample.
- **Causal?** Yes. The data are sorted by cycle start time. Case I trains on only the **first 100 runs**. Case II uses 25 or 50 initial wafers and fixed-rate sampling ("first of every five wafers from each lot"; the paper notes that normally the first and last wafers of a lot are measured). Case III uses variance-triggered dynamic sampling.
- **How it applies:** This is the closest published precedent for the requested design. Adopt:
  - time-sorted evaluation;
  - a small initial window of about 25–100 wafers;
  - sampling scenarios "1 of 5 per lot" versus variance-percentile triggering with t ≈ 0.8;
  - reporting the number of measurements used next to MSE and MAPE.

### 1.5 Cai, Feng, Yang, Li, Li & Lee (2020): KNN plus multi-task GP with uncertainty
- **Title:** A virtual metrology method with prediction uncertainty based on Gaussian process for chemical mechanical planarization
- **Venue / year:** Computers in Industry 119:103228, 2020
- **DOI:** https://doi.org/10.1016/j.compind.2020.103228
- **Verified:** yes (abstract)
- **Key idea:** KNN finds reference MRR samples in the history. A GPR fuses them, and a multi-task GP gives the final MRR and its uncertainty from historical and reference MRR. Evaluated on PHM 2016.
- **Causal?** Not verified. "Historical dataset" and "past MRR" suggest partly past-based, but the split was not read.
- **How it applies:** Use a GP (or GP residual model) over time plus consumable usage to supply the predictive variance that drives sampling and EWMA weighting (see §5 and §6).

### 1.6 Cai, Feng, Zhu, Yang, Li & Lee (2021): JIT reference plus particle filter
- **Title:** Adaptive virtual metrology method based on Just-in-time reference and particle filter for semiconductor manufacturing
- **Venue / year:** Measurement 168:108338, 2021
- **DOI:** https://doi.org/10.1016/j.measurement.2020.108338
- **Verified:** yes (abstract)
- **Key idea:**
  - Just-in-time search retrieves similar historical samples. SVR fuses them with past MRR.
  - A particle filter estimates and updates the fused result, so the model "tracks the CMP process change and continuously learns". The authors call it an online dynamic method.
  - Evaluated on the public (PHM 2016) dataset against dynamic and static baselines.
- **Causal?** Described as online and dynamic. The split details were not verified.
- **How it applies:** A state-space correction layer (a particle or Kalman filter on the bias) on top of a static regressor. This is a principled alternative to an EWMA offset when measurements arrive irregularly.

### 1.7 Liu, Tseng, Hsaio, Wu & Lu (2022): fusion network, best published contest-split MSE
- **Title:** Predicting the Wafer Material Removal Rate for Semiconductor Chemical Mechanical Polishing Using a Fusion Network
- **Venue / year:** Applied Sciences 12(22):11478, 2022
- **DOI / URL:** https://doi.org/10.3390/app122211478 · https://www.mdpi.com/2076-3417/12/22/11478 (MDPI returns 403 to scripts. I read the Wayback copy: https://web.archive.org/web/20230423052328/https://www.mdpi.com/2076-3417/12/22/11478/htm)
- **Verified:** yes (full text, archived copy)
- **Key idea:** A shallow branch takes engineered features and a deep branch learns embeddings; the two are fused. Reported test MSE:

  | Method | Test MSE |
  |---|---|
  | Fusion network (this paper) | 6.36 |
  | Jia et al. | 6.62 |
  | XGBoost | 6.66 |
  | Random forest | 6.69 |
  | Zhang et al. ResCNN | 6.72 |
  | Di et al. | 7.07 |

- **Causal?** No. The training data were split randomly 9:1 into train and validation, and results are on the contest test set.
- **How it applies:** It gives the non-causal state of the art for the README comparison table. State explicitly that it is not comparable to a delayed, sampled, past-only protocol.

### 1.8 Other PHM 2016 CMP papers (brief)

| Title | Authors · venue · year | DOI / URL | Verified | Protocol and relevance |
|---|---|---|---|---|
| Prediction of material removal rate in chemical mechanical polishing via residual convolutional neural network | J. Zhang, Y. Jiang, H. Luo, S. Yin · Control Engineering Practice 107:104673, 2021 | https://doi.org/10.1016/j.conengprac.2020.104673 | yes (abstract) | RF variable selection, then ResCNN. Contest split. A deep-learning baseline. |
| Virtual Metrology for Semiconductor CMP Process Using Wide & Deep Learning | F. Zhang, W. Jiang, H. Wang · ICCPR 2021, pp. 345–349 | https://doi.org/10.1145/3497623.3497679 | yes (abstract) | Statistical plus nearest-neighbour MRR features, filter and wrapper selection, Wide & Deep. Contest split. |
| Semi-Supervised Deep Kernel Active Learning for MRR Prediction in CMP | C. Lv, J. Huang, M. Zhang, H. Wang, T. Zhang · Sensors 23(9):4392, 2023 | https://doi.org/10.3390/s23094392 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10181745/ | yes (full text, PMC) | Phase partition, deep-kernel GP, semi-supervised learning plus active learning under varying labelled fractions (labels drawn at random). Contest train/val/test split. The closest work to "only a sample of wafers measured", but not time-causal. |
| Data-driven Approach to MRR Prediction in CMP | Z. Li, D. Wu · Annual Conf. PHM Society 10(1), 2018 | https://doi.org/10.36001/phmconf.2018.v10i1.489 | yes (abstract; protocol read in PDF) | Stacking of RF, GBT and ERT. Uses the contest train/validation/test sets. |
| Prediction of MRR for CMP Using Decision Tree-Based Ensemble Learning | Z. Li, D. Wu, T. Yu · J. Manuf. Sci. Eng. 141(3), 2019 | https://doi.org/10.1115/1.4042051 | yes (abstract) | Journal version of the stacking approach. |
| A deep learning-based approach to material removal rate prediction in polishing | P. Wang, R. X. Gao, R. Yan · CIRP Annals 66(1):429–432, 2017 | https://doi.org/10.1016/j.cirp.2017.04.013 | yes (DOI record); content: snippet only | Deep-belief-network MRR model. Di et al. 2017 compare against it. |
| Recurrent feature-incorporated CNN for VM of the CMP process | K. B. Lee, C. O. Kim · J. Intelligent Manufacturing (online 2018) | https://doi.org/10.1007/s10845-018-1437-4 | yes (DOI record); content: snippet only | RNN plus CNN VM robust to nonlinear drift. Use of PHM 2016 data not verified. |
| Virtual metrology on CMP process based on Just-In-Time Learning | M. A. Jebri, G. Graton, E. M. El Adel, M. Ouladsine, J. Pinaton · ICSC 2016, pp. 169–174 | https://doi.org/10.1109/ICoSC.2016.7507082 | yes (DOI record); content: snippet only | JIT local models for CMP VM (an STMicroelectronics collaboration). |
| Material removal rate prediction in CMP with conditional probabilistic autoencoder and stacking ensemble learning | Y. Wei, D. Wu · J. Intelligent Manufacturing, 2022 | https://doi.org/10.1007/s10845-022-02040-w | yes (DOI record) | Later stacking work. Content not read. |
| Virtual metrology for chemical mechanical planarization of semiconductor wafers | B. Deivendran, V. Masampally, N. K. V. Nadimpalli, V. Runkana · J. Intelligent Manufacturing, 2024 | https://doi.org/10.1007/s10845-024-02335-0 | yes (DOI record) | Content not read. |
| Stacking ensemble learning based MRR prediction model for CMP process of semiconductor wafer | Z. Song et al. · AI EDAM 38, 2024 | https://doi.org/10.1017/S0890060424000167 | yes (DOI record) | Content not read. |
| Adaptive Online Time-Series Prediction for Virtual Metrology in Semiconductor Manufacturing | S. Zabrocki, P. S. Jo, C. Park, D. Yim, S. Yun, B.-J. Lee · ASMC 2023 | https://doi.org/10.1109/ASMC57536.2023.10121099 | yes (abstract) | Not PHM data. Time-aware normalisation plus adaptive online learner, deployed in APC on HVM lines. Supports the "normalise for drift, then learn online" design. |
| Virtual metrology in semiconductor manufacturing: Current status and future prospects | V. Maitra, Y. Su, J. Shi · Expert Systems with Applications 249:123559, 2024 | https://doi.org/10.1016/j.eswa.2024.123559 | yes (DOI record) | Recent VM review, useful as a general citation. |

---

## 2. GitHub repositories on the PHM 2016 CMP data

I found these through GitHub repository and code search (`PHM 2016 CMP`, `phm2016`, `CMP removal rate`, `virtual metrology`, and code search for `AVG_REMOVAL_RATE`, which gave 339 hits). Metadata came from the GitHub search API; READMEs were read from raw.githubusercontent.com. Star counts are as of 2026-09-29.

| owner/repo | URL | Stars | Language | Verified | Approach | Causal or online? |
|---|---|---|---|---|---|---|
| akangel0307/PHM-Data-Challenge | https://github.com/akangel0307/PHM-Data-Challenge | 21 | Python (scripts; API reports none) | yes (README) | Splits the data into two or three "modes" by MRR level and stage, then benchmarks LR, KNN, GPR, PLS, RF, AdaBoost, SVR, GBDT, ExtraTrees, XGBoost and ElasticNet (Chinese README). | No. Offline, mode-wise regression. |
| dasolma/phmd | https://github.com/dasolma/phmd | 23 | Python | yes (PHM16 metadata JSON read) | Dataset-access library for PHM datasets. `phmd/metadata/PHM16.json` describes the CMP task (target `AVG_REMOVAL_RATE`, identifier `WAFER_ID`). | Not applicable. Useful as a data-download route now that the original links are dead. |
| ningmengzhihe/deep_learning_prediction | https://github.com/ningmengzhihe/deep_learning_prediction | 6 | Jupyter Notebook | yes (README) | Many notebooks: classic ML, semi-supervised Coreg, Conv1D/Conv2D, RNN/LSTM/GRU, Transformer, time-segment LSTM plus SAE, Wide & Deep (mode I, chamber 4). The README cites "文兰硕士论文" (Wenlan's master's thesis), which suggests a link to the Tsinghua Wide & Deep authors. That link is my inference. | No. Contest splits. |
| IonCojucari/Time-Series-Analysis---Chemical-Mechanical-Polishing | https://github.com/IonCojucari/Time-Series-Analysis---Chemical-Mechanical-Polishing | 5 | C | yes (README) | Course project: descriptive statistics, kσ outlier removal, per-wafer aggregation, optional linear regression. | No. |
| JamesLeeCY/semiconductor-quality-ml (folder `02_phm_cmp`) | https://github.com/JamesLeeCY/semiconductor-quality-ml | 0 | Python | yes (README) | 208 length-invariant time and frequency features plus XGBoost, giving 6.24 MSE on the official test set. Validation uses `group_temporal_split` by WAFER_ID ordered by start time. Test labels recovered from the Kaggle file in §3. Hand-crafted features beat BiGRU and 1D-CNN. | Partly. A temporal validation split, but the headline number is on the interleaved official test set. No online updating. |
| jsw010204-web/cmp-metrology-allocation | https://github.com/jsw010204-web/cmp-metrology-allocation | 0 | Python | yes (README) | Same LightGBM under three splits: random split MAE 2.800, maintenance-cycle grouped split 4.299, chronological split 5.279. Also prediction-interval coverage, offline simulation of metrology allocation, and sequential-update analysis (Korean README). | Yes, partly. Chronological evaluation plus a metrology-allocation simulation. The closest public repo to the requested design. |
| kamalpraven/cmp-virtual-metrology | https://github.com/kamalpraven/cmp-virtual-metrology | 0 | Python | yes (README) | Process-aware XGBoost: group-held-out MAE 2.58, official test MAE 1.68. A timestamp "stress split" gives MAE 4.38, or 3.96 without the dresser-table features. Dresser-table usage shows a large distribution shift in time order (SMD 2.74). | Partly. The timestamp stress test only. |
| Others found (metadata only, READMEs not read) | Sryily1105/2016-PHM-Data-Challenge · kennychennn/Semiconductor-cmp-prediction · gringreen1111/SKTelecom-MachineLearning-CMP-analysis (LightGBM plus cross-chamber generalisation) · anandu-7/Virtual-Metrology-for-CMP · DGU-FabLens/fablens (VM plus SPC for CMP) · djrudwo05121212-glitch/CMP-MRR-Analysis · OsamaAbdulkhalique98/waferaverageremovalrateprediction · MaheshikaWalpola/cmpo-modpipe (CMP ontology and knowledge graph from PHM 2016) | all under https://github.com/<owner>/<repo> | 0–1 | Python / Jupyter | yes (exists; GitHub API) | Various offline regressions. | Not checked. |

**Takeaways.**
- No public repository implements a fully causal pipeline with metrology delay and sampling on this dataset.
- The three repos that do test time order all report a large degradation from random or interleaved splits to chronological splits: 2.80 → 5.28 MAE (jsw010204-web) and 2.58 → 4.38 MAE (kamalpraven).
- This supports making the causal protocol the headline of the README.

---

## 3. Kaggle

| Item | URL | Verified | Notes |
|---|---|---|---|
| "CMP data set" by ZheCJ (markshizhe) | https://www.kaggle.com/datasets/markshizhe/cmp-data-set | yes (Kaggle public API: dataset view and file list) | One file, `CMP-test-removalrate_V3.csv` (12,670 bytes). Dataset last updated 2016-10-12, ODbL licence, 234 downloads. JamesLeeCY's README says its 424 (WAFER_ID, STAGE) keys match the official test keys with real MRR values, so it is the source of the recovered test labels. I did not check that claim against the file contents. |
| UCI SECOM (Kaggle mirror) | https://www.kaggle.com/datasets/paresh2047/uci-semcom (16,599 downloads, per the API) · original: https://archive.ics.uci.edu/dataset/179/secom | yes (API and HTTP 200) | Pass/fail yield classification with 590 sensors. It has no regression target, so it is only relevant as a contrast. Not needed for a CMP removal-rate VM. |
| Kaggle notebooks on PHM 2016 CMP or VM | – | not found | The Kaggle kernels API needs authentication, and web search found no CMP-removal-rate notebooks. One search hit is a SECOM notebook ("FMST Semiconductor manufacturing project", https://www.kaggle.com/code/saurabhbagchi/fmst-semiconductor-manufacturing-project; snippet only). I recommend not claiming any Kaggle notebook precedent. |

---

## 4. Adaptive / online VM and soft sensors

### 4.1 Khan, Moyne & Tilbury (2008): VM plus feedback control with recursive PLS
- **Title:** Virtual metrology and feedback control for semiconductor manufacturing processes using recursive partial least squares
- **Venue / year:** Journal of Process Control 18(10):961–974, 2008
- **DOI:** https://doi.org/10.1016/j.jprocont.2008.04.014
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:** A recursive PLS VM model feeds wafer-to-wafer (W2W) control of a MIMO process. The formulation explicitly allows **metrology delays, consistent drifts and sudden changes**. The VM uses pre-process metrology plus FDC-collected tool data.
- **How it applies:** RPLS with a forgetting factor is a cheap, well-cited online baseline. Update when a delayed measurement arrives, and model the delay explicitly in the evaluation.

### 4.2 Khan, Moyne & Tilbury (2007): factory-wide control with VM
- **Title:** An Approach for Factory-Wide Control Utilizing Virtual Metrology
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 20(4):364–375, 2007
- **DOI:** https://doi.org/10.1109/TSM.2007.907609
- **Verified:** yes (DOI record)
- **How it applies:** The standard citation for using VM between sparse physical measurements in R2R/W2W loops.

### 4.3 Qin (1998): recursive PLS
- **Title:** Recursive PLS algorithms for adaptive data modeling
- **Venue / year:** Computers & Chemical Engineering 22(4–5):503–514, 1998
- **DOI:** https://doi.org/10.1016/S0098-1354(97)00262-7
- **Verified:** yes (DOI record)
- **How it applies:** Algorithmic reference for RPLS and moving-window PLS with a forgetting factor.

### 4.4 Kadlec, Gabrys & Strandt (2009): data-driven soft sensors
- **Title:** Data-driven Soft Sensors in the process industry
- **Venue / year:** Computers & Chemical Engineering 33(4):795–814, 2009
- **DOI / URL:** https://doi.org/10.1016/j.compchemeng.2008.12.012 · repository: https://eprints.bournemouth.ac.uk/8498/
- **Verified:** yes (abstract, repository page)
- **Key idea:** Surveys what process data look like for soft sensors: delays, sampling periods, missing data, outliers, drift and collinearity. Also covers modelling (PCA, ANN) and the open maintenance challenges.
- **How it applies:** Justifies the data-contract checks: measurement delay and sampling as first-class properties of the data, and drift monitoring.

### 4.5 Kadlec, Grbić & Gabrys (2011): adaptation mechanisms for soft sensors
- **Title:** Review of adaptation mechanisms for data-driven soft sensors
- **Venue / year:** Computers & Chemical Engineering 35(1):1–24, 2011
- **DOI:** https://doi.org/10.1016/j.compchemeng.2010.07.034
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:** Classifies soft-sensor adaptation into families such as moving window or recursive updating, ensemble re-weighting and just-in-time/local learning, using online data plus performance feedback.
- **How it applies:** A taxonomy for the README. The design would combine (a) a recursive or EWMA bias update, (b) window retraining, and (c) online ensemble weighting.

### 4.6 Cheng & Chiu (2004): just-in-time learning
- **Title:** A new data-based methodology for nonlinear process modeling
- **Venue / year:** Chemical Engineering Science 59(13):2801–2810, 2004
- **DOI:** https://doi.org/10.1016/j.ces.2004.04.020
- **Verified:** yes (DOI record)
- **Key idea:** The classic just-in-time learning (JITL) method: build a local model from the nearest historical samples for each query.
- **How it applies:** Usage-space nearest neighbours restricted to already-measured past wafers are a causal JIT model.

### 4.7 Jebri, El Adel, Graton, Ouladsine & Pinaton (2016): JITL VM for missing measurements from sampling
- **Title:** Virtual metrology on semiconductor manufacturing based on Just-in-time learning
- **Venue / year:** IFAC-PapersOnLine 49(12):89–94, 2016
- **DOI:** https://doi.org/10.1016/j.ifacol.2016.07.555
- **Verified:** yes (abstract)
- **Key idea:** VM fills in variables that are "missing" because of measurement sampling strategies, using a modified JITL. The work is from an STMicroelectronics project.
- **How it applies:** Frames VM as imputation of unmeasured wafers under sampling, which is exactly the setup here.

### 4.8 Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014): concept-drift survey
- **Title:** A survey on concept drift adaptation
- **Venue / year:** ACM Computing Surveys 46(4):1–37, 2014
- **DOI:** https://doi.org/10.1145/2523813
- **Verified:** yes (DOI record)
- **How it applies:** A general citation for drift detection and adaptation, and for evaluating with prequential or progressive validation.

Kalman-filter and state-space VM: see Han et al. 2025 (§1.4, which benchmarks KF, DLKF and BARX) and Cai et al. 2021 (§1.6, particle filter).

---

## 5. VM reliability: Automatic Virtual Metrology (AVM), RI and GSI

### 5.1 Cheng, Chen, Su & Zeng (2008): reliance index and global similarity index
- **Title:** Evaluating Reliance Level of a Virtual Metrology System
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 21(1):92–103, 2008
- **DOI:** https://doi.org/10.1109/TSM.2007.914373
- **Verified:** yes (DOI record). I took the formulas from the matching patent (§5.2), which I read.
- **Key idea:** Introduces the **reliance index (RI)** and the **global similarity index (GSI)** for judging each VM prediction.

### 5.2 Patent US7593912B2: exact RI and GSI definitions
- **Title:** Method for evaluating reliance level of a virtual metrology system in product manufacturing
- **Inventors:** Fan-Tien Cheng, Yeh-Tung Chen, Yu-Chuan Su (National Cheng Kung University)
- **Dates:** priority 2006-05-10, granted 2009-09-22
- **URL:** https://patents.google.com/patent/US7593912B2/en
- **Verified:** yes (full text via Google Patents)
- **Definitions (from the patent):**
  - Two models predict the same wafer: a *conjecture* model (NN in the example) gives ŷ_N, and a *reference* model (multiple regression in the example) gives ŷ_r. Both are z-standardised with the training metrology mean and standard deviation, giving Z_ŷN and Z_ŷr.
  - **RI** = 2 ∫ from (Z_ŷN + Z_ŷr)/2 to ∞ of a normal density with μ = min(Z_ŷN, Z_ŷr) and σ = 1. This is the overlap area of two unit normals centred on the two predictions, so **RI = 2·[1 − Φ(|Z_ŷN − Z_ŷr| / 2)] ∈ [0, 1]**, and a higher value means more reliable. The patent's typeset density reads "1/(2π)σ"; the standard 1/(√(2π)σ) is intended.
  - **RI_T** is the RI value when the two predictions differ by the maximal tolerable error E_L, where error is defined as |y − ŷ_N| / ȳ × 100 %. The patent uses Z_Center = Z_ŷN + ȳ·(E_L/2)/σ_y, which gives RI_T = 2·[1 − Φ(ȳ·E_L / (2σ_y))]. Example: E_L = 3 % gives RI_T = 0.567.
  - **GSI** = D²_λ = Z_λᵀ R⁻¹ Z_λ. This is the squared Mahalanobis distance of the standardised process-feature vector from the model centre (0), where R is the correlation matrix of the training process data. The patent does not divide by p.
  - **GSI_T** = 2 to 3 × max over training sets of GSI_a.
  - **ISI** (individual similarity index) is the z-score of each process parameter; values above 3 flag an unusual parameter.
  - A prediction is trusted when RI > RI_T and GSI < GSI_T.

### 5.3 Cheng, Huang & Kao (2007): dual-phase VM
- **Title:** Dual-Phase Virtual Metrology Scheme
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 20(4):566–571, 2007
- **DOI:** https://doi.org/10.1109/TSM.2007.907633
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:**
  - Phase I VM is output immediately once a wafer's process data are complete, for promptness.
  - Phase II VM is recomputed after the model is updated with newly arrived actual metrology, for accuracy.
  - The AVM server outputs VM-I, VM-II, RI and GSI.
- **How it applies:** Maps directly onto metrology delay. Emit a Phase-I prediction at process end, then log a Phase-II re-prediction for the same wafer after the next model update. Evaluate on Phase-I, since that is what drives real-time decisions.

### 5.4 Cheng, Huang & Kao (2012): the AVM system
- **Title:** Developing an Automatic Virtual Metrology System
- **Venue / year:** IEEE Transactions on Automation Science and Engineering 9(1):181–188, 2012
- **DOI:** https://doi.org/10.1109/TASE.2011.2169405
- **Verified:** yes (DOI record)
- **How it applies:** The standard AVM citation: automatic model refresh, data-quality checks, and RI/GSI as part of the deployment.

### 5.5 Kao, Cheng, Wu, Kong & Huang (2013): R2R control with RI
- **Title:** Run-to-Run Control Utilizing Virtual Metrology With Reliance Index
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 26(1):69–81, 2013
- **DOI:** https://doi.org/10.1109/TSM.2012.2228243
- **Verified:** yes (DOI record). Content read second-hand through Wan & McLoone 2018 (§6.5).
- **Key idea:** The EWMA R2R gain is scaled by RI when VM rather than actual metrology is fed back. It uses a CMP simulation with a **600-wafer maintenance cycle**, with **RI_T = 0.7 and GSI_T = 9**, as reused by Wan & McLoone.
- **How it applies:** Gives ready-made threshold values and a CMP simulation template for testing VM-in-the-loop on this data.

---

## 6. VM plus run-to-run (R2R) control and EWMA

### 6.1 Ingolfsson & Sachs (1993): EWMA controller stability
- **Title:** Stability and Sensitivity of an EWMA Controller
- **Venue / year:** Journal of Quality Technology 25(4):271–287, 1993
- **DOI / URL:** https://doi.org/10.1080/00224065.1993.11979473 · SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2288120
- **Verified:** yes (DOI record). Abstract: snippet only (SSRN returned 403).
- **Key idea:** Gives "a simple condition relating the EWMA weight and the estimated process gain" for stability of a drifting or wandering process, plus the output MSD as a function of λ, the drift rate and the noise.
- **How it applies:** The condition is commonly quoted as **0 < λ·ξ < 2**, where ξ is the true gain divided by the model gain. I did not re-read that form in the paper. Use it to bound λ, and note that delay shrinks the stable region (§6.6).

### 6.2 Sachs, Hu & Ingolfsson (1995): run-by-run control
- **Title:** Run by run process control: combining SPC and feedback control
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 8(1):26–43, 1995
- **DOI:** https://doi.org/10.1109/66.350755
- **Verified:** yes (DOI record)
- **How it applies:** The canonical R2R reference. Combine the EWMA "gradual mode" with SPC-triggered "rapid mode" resets, for example on dresser or pad replacement.

### 6.3 Boning et al. (1996): R2R control of CMP
- **Title:** Run by run control of chemical-mechanical polishing
- **Authors:** D. S. Boning, W. P. Moyne, T. H. Smith, J. Moyne, R. Telfeyan, A. Hurwitz, S. Shellman, J. Taylor
- **Venue / year:** IEEE Transactions on Components, Packaging, and Manufacturing Technology, Part C 19(4):307–314, 1996
- **DOI:** https://doi.org/10.1109/3476.558560
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:** Response-surface models of average removal rate and non-uniformity (from 9-point thickness measurements), with EWMA model adaptation and multivariate recipe generation. It "compensated for substantial drift in the tool's removal rate."
- **How it applies:** Shows that removal-rate drift from consumable wear is the classic target of EWMA R2R in CMP. The VM's removal-rate prediction would feed the polish-time recipe.

### 6.4 Moyne, del Castillo & Hurwitz (eds.): R2R control book
- **Title:** Run-to-Run Control in Semiconductor Manufacturing
- **Publisher:** CRC Press (originally 2000–2001; Crossref lists the DOI record as 2018)
- **DOI:** https://doi.org/10.1201/9781420040661
- **Verified:** yes (DOI record)
- **Includes:** "A Comparative Analysis of Run-to-Run Control Algorithms…" (Ning, Moyne, Smith, Boning, del Castillo, Yeh, Hurwitz), https://doi.org/10.1201/9781420040661.ch6, and a chapter on CMP process automation and control.
- **How it applies:** The book citation for EWMA and double-EWMA and their comparison.

### 6.5 Wan & McLoone (2018): GPR VM-enabled R2R control with uncertainty-scaled EWMA
- **Title:** Gaussian Process Regression for Virtual Metrology-Enabled Run-to-Run Control in Semiconductor Manufacturing
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 31(1):12–21, 2018
- **DOI / URL:** https://doi.org/10.1109/TSM.2017.2768241 · accepted manuscript: https://pureadmin.qub.ac.uk/ws/files/137825983/GPR_R2R_final_post_review_accepted_paper.pdf
- **Verified:** yes (full text, accepted manuscript)
- **Key idea:**
  - The EWMA disturbance update is η_{k+1} = α(y_k − b·u_k) + (1 − α)η_k.
  - With actual metrology, α = α0. With VM, **α1 = GRI·α0**, where GRI is a Gaussian reliance index from the GP coefficient of variation cv = std/mean.
  - Two GRI forms: GRI = [1 − β|cv|/cv_max]₊ or GRI = exp(−β|cv|/cv_max), with cv_max the maximum cv over the training data.
  - Threshold: **GRI_T is the 5th percentile of training GRIs.** Below it, α1 = 0 (the VM value is ignored).
  - Case study: a CMP simulation with polish-time control, u_{k+1} = (PreY_{k+1} − y_target + η_{k+1}) / r̄_{k+1}, over a 600-wafer maintenance cycle.
  - Results: the GRI-weighted scheme beat both unweighted VM feedback and the RI-weighted scheme. The optimum was **α0 = 0.3** in the Weibull scenario.
- **How it applies:**
  - Use the VM predictive standard deviation (GP, quantile or conformal width) to scale the EWMA or bias-correction gain.
  - Use a 5th-percentile gate.
  - Search α0 around 0.2–0.4.

### 6.6 Kang, Kim, Lee, Doh & Cho (2011): VM for R2R control
- **Title:** Virtual metrology for run-to-run control in semiconductor manufacturing
- **Venue / year:** Expert Systems with Applications 38(3):2508–2522, 2011
- **DOI:** https://doi.org/10.1016/j.eswa.2010.08.040
- **Verified:** yes (DOI record). Abstract: snippet only.
- **Key idea:** VM models from several data-mining methods, embedded in an EWMA R2R controller for photolithography. VM predicts all wafers from sensor data plus the actual metrology of sampled wafers.
- **How it applies:** The standard citation that "VM fills unmeasured wafers for the EWMA loop."

### 6.7 Effect of metrology delay on EWMA R2R

| Title | Authors · venue · year | DOI | Verified | Use |
|---|---|---|---|---|
| Performance Analysis of EWMA Controllers Subject to Metrology Delay | M.-F. Wu, C.-H. Lin, D. S.-H. Wong, S.-S. Jang, S.-T. Tseng · IEEE TSM 21(3):413–425, 2008 | https://doi.org/10.1109/TSM.2008.2001218 | yes (DOI record); content: snippet only | Transient and asymptotic effect of delay on EWMA with bias plus ARMA disturbance. The feasible weight region shrinks with delay. |
| On the Stability of MIMO EWMA Run-to-Run Controllers With Metrology Delay | R. P. Good, S. J. Qin · IEEE TSM 19(1):78–86, 2006 | https://doi.org/10.1109/TSM.2005.863211 | yes (DOI record) | MIMO stability with delay. |
| Stability analysis of double EWMA run-to-run control with metrology delay | R. Good, S. J. Qin · ACC 2002, pp. 2156–2161 | https://doi.org/10.1109/ACC.2002.1023956 | yes (DOI record) | Double-EWMA (drift) case. |
| Stability Analysis of EWMA Run-to-Run Controller Subjects to Stochastic Metrology Delay | B. Ai, D. S.-H. Wong, S.-S. Jang, Y. Zheng · IFAC Proc. 44(1):12354–12359, 2011 | https://doi.org/10.3182/20110828-6-IT-1002.01041 | yes (DOI record) | Random (stochastic) delay, which fits a queue-driven metrology delay. |
| Performance analysis of single EWMA controller subject to metrology delay under dynamic models | Q. Gong, G. Yang, C. Pan, Y. Chen · IISE Transactions 50(2):88–98, 2018 | https://doi.org/10.1080/24725854.2017.1386338 | yes (DOI record); content: snippet only | Delay shrinks the feasible λ region through the Hurwitz criterion. |

### 6.8 Yi, Sang & Zhao (2005): production-fab CMP R2R with numbers
- **Title:** A run-to-run film thickness control of chemical-mechanical planarization processes
- **Authors:** Jingang Yi (Texas A&M), Wei-Shu Sang, Eugene Zhao (Lam Research)
- **Venue / year:** Proc. American Control Conference 2005, pp. 4231–4236
- **DOI / URL:** https://doi.org/10.1109/ACC.2005.1470643 · PDF: https://skoge.folk.ntnu.no/prost/proceedings/acc05/PDFs/Papers/0757_FrB08_3.pdf
- **Verified:** yes (full text)
- **Key idea (with numbers from a production fab):**
  - Off-line metrology measures pre- and post-CMP thickness on **one wafer per lot of 25**.
  - **Metrology delay is 1–2 lots** (N_d = 1 or 2).
  - Removal-rate drift comes from consumable (pad and conditioner disk) wear.
  - Across λ, Cpk rises quickly from λ = 0 to 0.2, then declines slowly to λ = 1. **λ\* = 0.33** was the average Cpk-optimal value, with moving-average horizon N = 20.
  - Head-to-head compensation ratio r_H\* = 0.66.
  - Cpk improved from 1.22 to 1.58 with R2R, and from 1.32 to 2.11 with head compensation.
- **How it applies:** The best verified source for realistic simulation settings: **sampling about 1/25 (4 %), delay 25–50 wafers, λ ≈ 0.3.**

---

## 7. Metrology sampling decisions driven by VM uncertainty

### 7.1 Nduhura-Munga et al. (2013): sampling techniques review
- **Title:** A Literature Review on Sampling Techniques in Semiconductor Manufacturing
- **Authors:** J. Nduhura-Munga, G. Rodriguez-Verjan, S. Dauzère-Pérès, C. Yugma, P. Vialletelle, J. Pinaton
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 26(2):188–195, 2013
- **DOI / URL:** https://doi.org/10.1109/TSM.2013.2256943 · open copy: https://nru.uncst.go.ug/server/api/core/bitstreams/841ac6ea-8bd5-4aaa-b729-434dbae10460/content
- **Verified:** yes (full text)
- **Key idea:**
  - Sampling strategies fall into **static** (fixed rules), **adaptive** (rules adjusted by other controls) and **dynamic** (lot-by-lot decisions using factory state and metrology capacity) groups. Dynamic sampling suits modern high-mix fabs.
  - Example of industrial per-rule sampling rates (AMD's Dynamic Sampling System): metal etchers 30 %, plasma etch 10 %, a product 25 %.
  - The delay between process and inspection is flagged as an open research avenue.
- **How it applies:** Name the three sampling policies simulated as static (1-in-N), adaptive (1-in-N, raised after maintenance) and dynamic (uncertainty-triggered), and cite this review.

### 7.2 Kurz, De Luca & Pilz (2015): sampling decision system for VM
- **Title:** A Sampling Decision System for Virtual Metrology in Semiconductor Manufacturing
- **Venue / year:** IEEE Transactions on Automation Science and Engineering 12(1):75–83, 2015
- **DOI:** https://doi.org/10.1109/TASE.2014.2360214
- **Verified:** yes (DOI record). Abstract: snippet only.
- **Key idea:** Decide when a real measurement is needed using the expected utility of measurement information, a two-stage sampling decision model and wafer quality-risk values. Also covers updating VM reliability as real measurements arrive.
- **Conference version:** Kurz, Pilz, Schirru, Pampuri, De Luca, "A sampling decision system for semiconductor manufacturing – relying on virtual metrology and actual measurements", Winter Simulation Conference 2014, https://doi.org/10.1109/WSC.2014.7020109. **Verified:** yes (DOI record).
- **How it applies:** Provides the risk-based rule: measure when P(|y − target| > spec | VM prediction and uncertainty) exceeds a threshold, under a budget.

### 7.3 Susto (2017): dynamic sampling from VM confidence
- **Title:** A dynamic sampling strategy based on confidence level of virtual metrology predictions
- **Venue / year:** ASMC 2017, pp. 78–83
- **DOI:** https://doi.org/10.1109/ASMC.2017.7969203 (from Crossref. A search snippet said …7969205, but Crossref's record for this title is 7969203.)
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:** Dynamic sampling that skips measurements when VM confidence is high.
- **How it applies:** Supports variance- or interval-width-triggered sampling.

### 7.4 Other sampling evidence
- Han et al. 2025 (§1.4) is the PHM 2016 CMP instance of uncertainty-driven sampling (variance percentile t ≈ 0.8, versus fixed "1 of 5 per lot").
- Jebri et al. 2016 (§4.7) treats VM as imputation of wafers left unmeasured by sampling.

---

## 8. Online uncertainty: adaptive conformal inference and expert aggregation

### 8.1 Gibbs & Candès (2021): adaptive conformal inference (ACI)
- **Title:** Adaptive Conformal Inference Under Distribution Shift
- **Venue / year:** Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
- **URL:** https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html (title checked) · arXiv: https://arxiv.org/abs/2106.00170 · code: https://github.com/isgibbs/AdaptiveConformal (R, 8 stars)
- **Verified:** yes (full text, arXiv v3)
- **Key idea:**
  - Update rule: **α_{t+1} = α_t + γ(α − err_t)**, where err_t = 1 if y_t fell outside the set at level α_t.
  - Long-run miss frequency approaches α for *any* data sequence. The finite-time bound is roughly (max{α1, 1 − α1} + γ)/(γT).
  - The authors used **γ = 0.005**. Larger γ adapts faster but makes α_t more volatile, and γ that is too small behaves like static conformal.
- **How it applies:**
  - Wrap any point VM with split-conformal residual quantiles, updated with ACI as labels *arrive* (delayed).
  - Only measured wafers generate err_t, so coverage is guaranteed on the measured stream. If sampling is itself uncertainty-triggered, the measured stream is not representative. Report coverage on a random audit subsample, or on all wafers offline. This caution is my inference, not a claim from the paper.

### 8.2 Zaffran et al. (2022): AgACI
- **Title:** Adaptive Conformal Predictions for Time Series
- **Authors:** M. Zaffran, A. Dieuleveut, O. Féron, Y. Goude, J. Josse
- **URL:** arXiv:2202.07282, https://arxiv.org/abs/2202.07282 · code: https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries (71 stars)
- **Verified:** yes (abstract, arXiv API). My recollection is that it appeared at ICML 2022, but I did not re-verify the venue.
- **Key idea:** Analyses how γ affects ACI efficiency. **AgACI** runs ACI with several γ values and aggregates them with online expert aggregation, so no γ needs to be chosen.
- **How it applies:** Removes the need to tune γ. Run γ ∈ {0.001, 0.005, 0.01, 0.05} and aggregate.

### 8.3 Angelopoulos, Candès & Tibshirani (2023): conformal PID control
- **Title:** Conformal PID Control for Time Series Prediction
- **Venue / year:** NeurIPS 36 (2023), pp. 23047–23074
- **DOI / URL:** https://doi.org/10.52202/075280-1000 · arXiv: https://arxiv.org/abs/2307.16895 · code: https://github.com/aangelopoulos/conformal-time-series (145 stars)
- **Verified:** yes (abstract via arXiv API; DOI record)
- **Key idea:** Quantile tracking (P), error integration (I) and a scorecaster (D) for online conformal. Handles trends and seasonality.
- **How it applies:** When residuals trend within a dresser life, a "scorecaster" of residual size, for example on dresser usage, can make intervals sharper than plain ACI.

### 8.4 Manufacturing applications of conformal prediction

| Title | Authors · year | URL | Verified | Relevance |
|---|---|---|---|---|
| Robust and Reliable AI for Predictive Quality in Semiconductor Materials Manufacturing with MLOps and Uncertainty Quantification | M. Gao, J. M. Perathoner, A. L. Bonin, S. Eulig, G. Klesse · arXiv 2026 | https://arxiv.org/abs/2605.07752 | yes (abstract) | Five years of fab-materials data. **Fixed retraining every 5 production batches, without hyperparameter retuning, was best across drift conditions.** Conformal intervals are compared against control limits. |
| Online scalable Gaussian processes with conformal prediction for guaranteed coverage | J. Xu, Q. Lu, G. B. Giannakis · arXiv 2024 | https://arxiv.org/abs/2410.05444 | yes (abstract) | Online GP plus ACI-style threshold feedback. Fits the GP-VM plus conformal combination. |
| Distribution-Free Process Monitoring with Conformal Prediction | C. Burger · arXiv 2025 | https://arxiv.org/abs/2512.23602 | yes (abstract) | Conformal control charts and "uncertainty spikes", for an SPC view of VM residuals. |
| cmp-metrology-allocation (repo) | jsw010204-web | §2 | yes (README) | Interval coverage plus metrology allocation on PHM 2016 CMP. |

I found no peer-reviewed paper applying *adaptive* conformal inference specifically to semiconductor VM. Say "to our knowledge" in the README.

### 8.5 Cesa-Bianchi & Lugosi (2006): online expert aggregation
- **Title:** Prediction, Learning, and Games
- **Publisher / year:** Cambridge University Press, 2006
- **DOI / URL:** https://doi.org/10.1017/CBO9780511546921 · https://www.cambridge.org/core/books/prediction-learning-and-games/A05C9F6ABC752FAB8954C885D0065C8F
- **Verified:** yes (publisher landing page)
- **Key idea:**
  - The exponentially weighted average forecaster uses weights w_{i,t} ∝ exp(−η·L_{i,t−1}), where L is each expert's cumulative loss.
  - For convex losses bounded in [0, 1], η = √(8 ln N / n) gives regret ≤ √((n/2) ln N). This is the standard Theorem 2.2 form, recalled from the book and not re-read in this session.
- **How it applies:** Combine the causal members (persistent, EWMA-corrected tree model, GP, KNN-usage) with weights updated only when delayed labels arrive. Scale losses by a robust range to stay in [0, 1]. The R package **opera** (https://github.com/Dralliag/opera, 56 stars, verified) implements this family, including ML-Poly and BOA.

---

## 9. CMP physics for R2R

### 9.1 Preston (1927): the original Preston equation
- **Title:** The Theory and Design of Plate Glass Polishing Machines
- **Author / venue / year:** F. W. Preston · Journal of the Society of Glass Technology 11:214–256, 1927
- **URL:** no DOI. Bibliographic records: https://cir.nii.ac.jp/crid/1573950399985063296 (CiNii) · https://ndlsearch.ndl.go.jp/en/books/R100000136-I1571135650164455168 (NDL)
- **Verified:** snippet only. The records appeared in search results; I did not open the scan.
- **Related:** F. W. Preston, "The Polishing of Surfaces", Nature 119:13, 1927, https://doi.org/10.1038/119013a0. **Verified:** yes (DOI record).
- **Key idea:** Removal rate = K_p·P·V (pressure × relative velocity).
- **How it applies:** The physics prior, and the linear process model y = PreY − r·t that R2R polish-time control is built on (see Wan & McLoone §6.5).

### 9.2 Luo & Dornfeld (2001): material removal mechanism
- **Title:** Material removal mechanism in chemical mechanical polishing: theory and modeling
- **Venue / year:** IEEE Transactions on Semiconductor Manufacturing 14(2):112–133, 2001
- **DOI:** https://doi.org/10.1109/66.920723
- **Verified:** yes (DOI record). Content: snippet only.
- **Key idea:** A mechanistic abrasion model (plastic contact, abrasive size distribution, pad roughness) that leads to a revised Preston relation. Di et al. 2017 cite it for Preston.
- **How it applies:** The citation for "Preston is empirical; K_p depends on pad and abrasive state."

### 9.3 Tso & Ho (2007): pad-conditioning dressing rate
- **Title:** Factors influencing the dressing rate of chemical mechanical polishing pad conditioning
- **Venue / year:** International Journal of Advanced Manufacturing Technology 33(7–8):720–724, 2007 (online 2006)
- **DOI:** https://doi.org/10.1007/s00170-006-0501-y
- **Verified:** yes (DOI record). I read the formula as reproduced in Di et al. 2017.
- **Key idea:** Dressing rate = K_D·(V_D/(R·A·λ·d0))·(P/H_p)^1.5. Di et al. simplify this to K_D·U_D, a function of dresser usage.
- **How it applies:** Justifies dresser-usage features and a dresser-life piecewise model, and treating dresser replacement (the usage reset) as a regime reset for EWMA and ACI.

### 9.4 Krishnan, Nalaskowski & Cook (2010): slurry chemistry and mechanisms review
- **Title:** Chemical Mechanical Planarization: Slurry Chemistry, Materials, and Mechanisms
- **Venue / year:** Chemical Reviews 110(1):178–204, 2010 (online 2009)
- **DOI:** https://doi.org/10.1021/cr900170z
- **Verified:** yes (DOI record)
- **How it applies:** A standard broad CMP review for the README background.

### 9.5 Li, Baisie, Zhang & Zhang: diamond-disc pad conditioning
- **Title:** Diamond disc pad conditioning in chemical mechanical polishing
- **Venue / year:** chapter in *Advances in Chemical Mechanical Planarization (CMP)*, Elsevier, 2016 (a 2nd edition chapter is dated 2022)
- **DOI:** https://doi.org/10.1016/B978-0-08-100165-3.00013-9 (2022 edition: https://doi.org/10.1016/B978-0-12-821791-7.00014-9)
- **Verified:** yes (DOI record)
- **How it applies:** A review-level citation for conditioner and dresser effects on pad asperities and removal rate.

---

## 10. Online-ML and uncertainty libraries on GitHub

| owner/repo | URL | Stars | Verified | Relevance |
|---|---|---|---|---|
| online-ml/river | https://github.com/online-ml/river | 6,115 | yes (README) | Online linear models, trees and forests, KNN, drift detectors, time-series models. **`evaluate.progressive_val_score` has a `delay` parameter ("delayed progressive validation") where "the target is only revealed to the model after a certain amount of time"** (https://riverml.xyz/latest/api/evaluate/progressive-val-score/, verified). This is exactly the evaluation protocol needed. Paper: Montiel et al., "River: machine learning for streaming data in Python", JMLR 22(110), 2021, https://www.jmlr.org/papers/v22/20-1380.html (page title verified). |
| scikit-multiflow/scikit-multiflow | https://github.com/scikit-multiflow/scikit-multiflow | 798 | yes (API) | "The other ancestor of River." Legacy; cite River instead. |
| scikit-learn-contrib/MAPIE | https://github.com/scikit-learn-contrib/MAPIE | 1,596 | yes (API) | scikit-learn-compatible conformal prediction intervals. Time-series conformal (EnbPI/ACI-style) utilities, per the project; specific classes not verified in this session. |
| aangelopoulos/conformal-time-series | https://github.com/aangelopoulos/conformal-time-series | 145 | yes (API) | Reference implementations of ACI, quantile tracking and conformal PID. |
| mzaffran/AdaptiveConformalPredictionsTimeSeries | https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries | 71 | yes (API) | AgACI code. |
| Dralliag/opera | https://github.com/Dralliag/opera | 56 | yes (API) | Online expert aggregation in R (EWA, ML-Poly, BOA). |

---

## Design recommendations from the literature

1. **Evaluate causally and report the gap.**
   - Sort wafers by process start time. Train an initial model on a short prefix: Han et al. used the **first 100 runs**, and 25–50 initial wafers in the sampling studies.
   - Stream the rest with *delayed* progressive validation (River's `delay`).
   - Report the contest-split MSE only as a non-causal reference: Di 7.07, Jia 6.62, Liu 6.36.
   - Public repos show a large optimism gap from random or interleaved splits to chronological ones: MAE 2.80 → 5.28 (jsw010204-web) and 2.58 → 4.38 (kamalpraven). Expect the same.

2. **Use realistic metrology sampling and delay scenarios.**
   - A production CMP fab measured **one wafer per 25-wafer lot with a 1–2-lot delay** (Yi et al. 2005).
   - Han et al. used **"first of every five wafers per lot"** as the fixed-rate case and note that the first and last wafer per lot is common.
   - Suggested grid: sampling ∈ {100 %, 20 %, 4 %}, delay ∈ {0, 25, 50} wafers, plus a dynamic policy.

3. **Always include the last measured value as a "dynamic term".** Adjacent MRRs are highly correlated (Han Fig. 2), and the persistent model r_t = r_{t−1} is a strong baseline (Di: CV MSE 8.78). Under delay, "last" means the last *arrived* measurement in the same recipe or regime.

4. **EWMA bias correction: λ ≈ 0.2–0.4, with resets at maintenance.**
   - λ\* = 0.33 in a production CMP fab (Yi 2005). α0 = 0.3 was optimal in Wan & McLoone's CMP simulation.
   - Stability: λ·ξ in (0, 2) (Ingolfsson & Sachs). Metrology delay shrinks the stable and feasible region (Wu et al. 2008; Good & Qin 2002/2006), so stay at the low end when the delay is 25–50 wafers.
   - EWMA fails right after a dresser replacement (Han, run 144). Detect usage-counter resets and either reset the offset or temporarily raise λ, following Sachs et al.'s rapid mode.

5. **Scale updates by VM confidence.**
   - α1 = GRI·α0 with GRI = [1 − β·cv/cv_max]₊ or exp(−β·cv/cv_max), cv_max = the maximum cv on training data.
   - Ignore VM feedback when GRI is below its training **5th percentile** (Wan & McLoone 2018).
   - AVM equivalent: RI = 2[1 − Φ(|Z_N − Z_r|/2)] against RI_T from the tolerable error E_L (E_L = 3 % gives RI_T = 0.567), and GSI = zᵀR⁻¹z against GSI_T = 2–3 × the maximum training GSI (Cheng et al. patent). Kao et al. and Wan used **RI_T = 0.7 and GSI_T = 9**.
   - Log a "not reliable" flag per wafer.

6. **Dynamic sampling on predictive uncertainty.** Measure when predictive variance (or conformal interval width) exceeds its running **80th percentile**; t = 0.8 was the best trade-off in Han et al. on this dataset. Also enforce a minimum audit rate, for example one per lot. Report the number of measurements next to MSE, as Han does: fixed-rate needed 88 and 92 samples.

7. **Online intervals with ACI.** Use α_{t+1} = α_t + γ(α − err_t) with **γ ≈ 0.005** as the default (Gibbs & Candès), or aggregate several γ values (AgACI). Update only when delayed labels arrive. Because uncertainty-triggered sampling biases the labelled stream, estimate coverage on the random audit samples. This last point is my inference.

8. **Separate models or regimes by recipe (chamber group × stage)**, as Di et al. did with three conditions. Treat dresser, pad and dresser-table resets as regime boundaries. Han shows these usage variables drift and shift, and kamalpraven found dresser-table usage to be the strongest time-order distribution shift (SMD 2.74).

9. **Make model updating cheap and selective.**
   - Recursive or moving-window updates (RPLS, Qin 1998; Kadlec 2011 taxonomy).
   - Importance-screened sample admission: freshness, error and uncertainty (Feng et al. 2019).
   - GP hyperparameter refits per batch rather than per sample (Han 2025).
   - A fixed retraining cadence, for example every 5 batches, without hyperparameter retuning, was the most robust choice in a 5-year semiconductor-materials study (Gao et al. 2026).

10. **Emit dual-phase outputs.** Produce a Phase-I VM at process end, which drives real-time decisions and is what gets scored, and a Phase-II re-prediction after the next model update (Cheng et al. 2007).

---

### Notes on what could not be verified
- ScienceDirect, IEEE Xplore, SSRN and MDPI landing pages block automated fetching. For those papers I relied on Crossref records plus open copies where they existed (NIST, the QUB repository, PMC, the Wayback Machine, HAL mirrors, the ACC proceedings mirror).
- Any content claim marked "snippet only" should be checked against the PDF before quoting numbers from it.
- The EWMA stability form 0 < λξ < 2 and the EWA regret bound are standard results, but I did not re-read them in the original texts this session.
- I found no Kaggle notebook on PHM 2016 CMP.
