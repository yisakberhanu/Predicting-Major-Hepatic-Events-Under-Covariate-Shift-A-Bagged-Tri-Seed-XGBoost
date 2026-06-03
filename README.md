
# Predicting Major Hepatic Events Under Covariate Shift: A Bagged Tri-Seed XGBoost Cox Survival Ensemble Using Joint-Distribution Matching for Robust Patient Risk Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Machine_Learning-XGBoost--Cox-green.svg)](https://xgboost.readthedocs.io/)
[![Survival Analysis](https://img.shields.io/badge/Clinical_AI-Survival_Analysis-red.svg)](https://scikit-survival.readthedocs.io/)

---

## 📌 Executive Summary Matrix

* **The Clinical Problem:** Tracking and predicting clinical liver deterioration and major hepatic events (`evenements_hepatiques_majeurs`) in longitudinal patient records corrupted by severe covariate shift and administrative timeline censoring.
* **The Machine Learning Method:** A 60-model highly regularized framework executing a repeated Multilabel Stratified Cross-Validation loop on an XGBoost Cox Proportional Hazards engine.
* **The Implementation Mechanism (How):** Executing **Bagged Joint-Distribution Matching** across longitudinal patient metrics (Follow-up Duration + Healthcare Utilization Intensity) to dynamically align training data with test targets without introducing minority class data starvation.
* **The Clinical Solution:** Generates ultra-stable, rank-averaged patient risk forecasting matrices designed specifically to survive Private Leaderboard data variance shake-ups and prioritize targeted therapeutic interventions.

---

## 🎯 Project Deep-Dive & Clinical Context

In longitudinal electronic health records (EHR), data acquisition mechanics create inherent structural artifacts. Healthy patients frequently drop out of clinical monitoring tracking metrics early simply due to administrative reasons like moving locations (**Administrative Censoring**), causing an artificial spike at `Duration = 0`. Conversely, acutely ill patients exhibit dense, erratic appointment frequencies before dropping out due to critical medical emergencies.

If a gradient-boosted survival model is trained on raw, unadjusted rows, it mistakenly optimizes for hospital operational frequencies rather than actual biological pathophysiology (**Covariate Shift**). 

This repository presents an end-to-end framework that corrects these demographic shifts directly within the validation and data sampling layers. By framing the timeline structure as a joint probability cohort, the pipeline ensures the machine learning architecture evaluates biomarker progression with zero historical collection bias.

---

## 🚀 Architectural Breakthroughs

### 1. Bagged Joint-Distribution Cohort Balancing
To fix the profound distribution mismatch between the historical training data and the deployment target environment, this framework calculates the exact proportion of every combination of timeline duration and encounter counts. 

Instead of deploying rigid deterministic culling—which starves the algorithm of precious positive event signals—the pipeline implements **Bagged Undersampling**. It dynamically isolates a separate, distribution-matched control cohort for each initialization seed, ensuring full data utilization across the macro-ensemble.

### 2. Tri-Seed 60-Model Cross-Validation Stability
Sparse medical datasets with a low event density are highly susceptible to splitting anomalies. This framework completely mitigates initialization variance and prevents leaderboard overfitting by deploying a 60-model deep ensemble:
* **Seeds Instantiated:** `42`, `1234`, and `4`
* Each seed dynamically spawns a 20-fold Multilabel Stratified Split. The final output rank-averages all 60 independent probability streams, rendering the predictions immune to single-split data noise.

### 3. Multi-Tier Biomarker Trajectory Modeling
Rather than assessing snapshot values, the feature engineering layer maps the continuous tracking timeline of core clinical biomarkers (`AST`, `ALT`, `Bilirubin`, `GGT`, `Platelets`, `FIB-4 Stiffness`). It constructs trajectory delta metrics and evaluates personal patient trends against both global cohort means and localized completely healthy baselines to spot clear pathological anomalies.


## 📁 Repository Structure

```text
├── data/
│   └── README.md               # Place train.csv and test.csv data layers here
├── src/
│   └── pipeline.py             # Enterprise Class-based Python Pipeline implementation
├── requirements.txt            # Explicit dependency version tracking mapping
└── README.md                   # Core portfolio documentation

```

---

## 🛠️ Hyperparameter & Anti-Overfitting Controls

With small, high-stakes medical targets, severe regularization is mandatory to force the tree nodes to discover universal clinical rules rather than memorizing noise patterns:

| Hyperparameter | Configuration | Operational Purpose |
| --- | --- | --- |
| `objective` | `survival:cox` | Directly optimizes for Partial Likelihood under the Cox Proportional Hazards model |
| `max_depth` | `8` | Allows deep interaction mining between shifting multi-visit lab metrics |
| `learning_rate` | `0.015` | Slow contraction rate to ensure stable convergence across extreme OOF validation folds |
| `subsample` | `0.8` | Row subsampling to introduce structural diversity per boosting round |
| `colsample_bytree` | `0.5` | Forces trees to learn split criteria on changing subsets of biological indicators |
| `reg_alpha` | `2.0` | Heavy L1 penalty constraint to silence weak, noisy correlation coefficients |
| `reg_lambda` | `5.0` | Heavy L2 penalty constraint to compress large weight spikes across leaf nodes |

---

## 💻 Installation & Local Setup

Clone the repository and install the verified core scientific computing and biostatistical packages via pip:

```bash
git clone [https://github.com/yourusername/hepatic-risk-xgboost-cox-ensemble.git](https://github.com/yourusername/hepatic-risk-xgboost-cox-ensemble.git)
cd hepatic-risk-xgboost-cox-ensemble
pip install -r requirements.txt

```

### `requirements.txt`

```text
pandas>=1.5.0
numpy>=1.22.0
xgboost>=1.6.0
scikit-survival>=0.19.0
scikit-learn>=1.1.0
iterative-stratification>=0.1.7

```

---

## 🏃‍♂️ Execution & Pipeline Deployment

The core implementation is completely modularized inside an enterprise object-oriented class structure. To process data, balance cohorts across the multi-seed array, train the 60 out-of-fold models, and output rank-normalized risk profiles, execute the pipeline:

```bash
python src/pipeline.py

```

### Script Execution Log Flow:

```text
2026-06-03 12:19:03,142 - INFO - Loading and sanitizing clinical datasets...
2026-06-03 12:19:04,510 - INFO - Initiating 60-Model Tri-Seed Ensemble Training...
2026-06-03 12:19:04,512 - INFO - Phase 1/3: Executing Data Bagging & 20 Folds for Seed 42...
2026-06-03 12:19:04,890 - INFO -   -> Data bag constructed via Seed 42. Sample Size: 736 patients.
2026-06-03 12:19:22,110 - INFO - Phase 2/3: Executing Data Bagging & 20 Folds for Seed 1234...
2026-06-03 12:19:22,480 - INFO -   -> Data bag constructed via Seed 1234. Sample Size: 736 patients.
2026-06-03 12:19:40,330 - INFO - Phase 3/3: Executing Data Bagging & 20 Folds for Seed 4...
2026-06-03 12:19:40,710 - INFO -   -> Data bag constructed via Seed 4. Sample Size: 736 patients.
2026-06-03 12:19:58,950 - INFO - Ensemble complete. Rank averaging applied across all 60 models and 3 data bags.
2026-06-03 12:19:59,215 - INFO - Submission successfully exported to best_LB_hep_bagged_triseed.csv

```

```

```
