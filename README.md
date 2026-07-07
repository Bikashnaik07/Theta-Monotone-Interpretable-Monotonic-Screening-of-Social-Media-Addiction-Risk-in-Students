# θ-Monotone: Interpretable Monotonic Screening of Social Media Addiction Risk in University Students

An interpretable, clinically-safe machine learning pipeline for screening social media addiction severity, built around a psychometric latent trait (θ), monotonic constraints, and full explainability (SHAP, fairness, and calibration analysis).

## Overview

Standard classifiers treat addiction severity (Low → Moderate → High) as a nominal, unordered category — which ignores its ordinal, progressive nature and can produce clinically nonsensical predictions (e.g., jumping from "Low" straight to "High"). This project fixes that by:

- Extracting a single latent psychological trait **θ** (via PCA on questionnaire items) that represents an individual's underlying "true addiction propensity," instead of treating each questionnaire item as an independent, fragmented feature.
- Training a **monotonic XGBoost** classifier that is structurally constrained so increased symptoms can never decrease predicted risk — guaranteeing zero "skip-level" errors.
- Validating the model with cross-validation, subgroup fairness analysis, calibration curves, and global/local SHAP explanations.

The pipeline was developed and validated across **two independent datasets and populations**:

1. **Phase 1 — SMA10 Dataset**: N=1029 Indian university students, 10-item questionnaire (SMAQ1–10), 3-class severity target.
2. **Phase 2 — Italian Adolescents Dataset**: A more complex, multi-scale dataset (BSMAS, RSES, CSIQ-A, STAI-Y1/Y2) with unsupervised K-Means cluster-based risk labeling, multi-directional monotonic constraints, and leakage-safe feature engineering.

## Key Results

| Phase | Dataset | Accuracy | Macro F1 |
|---|---|---|---|
| Phase 1 | SMA10 (N=1029) | 98.54% | 98.69% |
| Phase 2 | Italian Adolescents (N=258) | 94.23% | 94.24% |

- Zero skip-level misclassifications in both phases — the model always respects ordinal severity progression.
- Well-calibrated probability outputs (calibration curves closely follow the ideal diagonal).
- Minimal fairness gaps across gender and geography subgroups.
- SHAP confirms the model's reasoning is driven by behavioral/psychometric signals, not demographics, and that all monotonic constraint directions align with empirical feature impact.

## Methodology

1. **Data cleaning & stratified train/test split** (80/20).
2. **Latent trait (θ) extraction** — PCA fitted on training data only, first principal component retained as θ, then appended as an engineered feature.
3. **Class imbalance handling** — SMOTENC oversampling (training set only, preserves categorical integrity).
4. **Model training & comparison** — Logistic Regression, Ordinal Logistic Regression (MORD), XGBoost (Nominal), and XGBoost (Monotonic).
5. **Validation** — 5-fold cross-validation, subgroup fairness analysis, calibration curve analysis.
6. **Explainability** — Global and local SHAP analysis (waterfall plots for case-level transparency).
