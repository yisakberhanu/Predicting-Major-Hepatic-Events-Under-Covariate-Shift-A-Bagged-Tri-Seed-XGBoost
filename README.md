# Hierarchical Clinical Risk Stratification via Bagged Tri-Seed Cox Ensembles

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Machine_Learning-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Survival Analysis](https://img.shields.io/badge/Biostatistics-Scikit--Survival-orange.svg)](https://scikit-survival.readthedocs.io/)

## 📌 Project Overview

This repository features a production-grade, object-oriented machine learning architecture designed for advanced survival analysis on longitudinal clinical profiles. The pipeline targets the prediction of major hepatic events (`evenements_hepatiques_majeurs`) by resolving critical real-world complications found in medical data: high class imbalance, administrative censoring noise, and profound covariate shift between cohort populations.

Instead of relying on brittle post-processing fixes or over-parameterized models that collapse during validation transitions, this architecture builds mathematical stability directly into the data tracking and validation layers.

---

## 🚀 Key Innovations & Engineering Strategy

### 1. Bagged Cohort Harmonization (Mitigating Covariate Shift)
Standard cross-validation frequently falls into the trap of optimizing for temporal distributions that do not map to the true target test environment (e.g., a massive over-representation of immediate dropouts at `Duration = 0`). 

To isolate purely biological risks, this pipeline implements **Joint Distribution Matching** across two intertwined axes: Longitudinal Follow-up Duration and Healthcare Utilization Intensity (visit frequency). Rather than performing simple deterministic filtering—which starves the algorithm of precious positive event samples—the pipeline applies **Bagged Undersampling**. It dynamically resamples a distinct, unbiased control subset matching target demographics for each initialization state, eliminating both sampling variance and distribution drift.

### 2. Tri-Seed 60-Model Ensemble (Surviving the Shake-up)
Medical datasets with sparse positive target populations are highly vulnerable to localized data splitting artifacts. This pipeline eradicates model initialization variance by establishing an ironclad ensemble of **60 distinct models** (3 unique random seeds × 20-fold cross-validation). 

* **Seeds Deployed:** `42`, `1234`, `4`
* Every unique seed loop initializes a separate data harmonization bag and forces the underlying gradient boosted decision trees to map independent feature splitting thresholds, averaging out background algorithmic noise.

### 3. Biomarker Trajectory & Population Deviations
Medical data points evaluated out of context fail to capture clinical deterioration dynamics. The feature engineering subsystem automatically tracks absolute longitudinal paths and contextualizes personal bio-profiles by calculating granular ratios against two distinct population metrics:
* **Global Population Baselines:** To spot macroeconomic tracking anomalies.
* **Healthy Sub-population Means:** To isolate precise, pathological biological signatures across highly reflective liver biomarkers (`AST`, `ALT`, `Bilirubin`, `GGT`, `FIB-4 Stiffness`).

---

## 🏗️ Pipeline Architecture

The pipeline is entirely refactored using highly structured, modular, **Object-Oriented Programming (OOP)** rules to match enterprise and clinical software engineering standards:
