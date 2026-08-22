# Project Brief: AI-Based Fraud Detection, Explainability, and Adversarial Robustness for Digital Banking Transactions

> Complete project context for Codex. Treat this file as the main source of truth for the project.

## 1. Project Overview

**Project Title:** AI-Based Fraud Detection, Explainability, and Adversarial Robustness for Digital Banking Transactions  
**Project Type:** Individual university project  
**Course:** Network & Internet Security (CST-8415)  
**Primary Environment:** Google Colab  
**Primary Language:** Python  
**Dataset:** PaySim synthetic financial transaction dataset

### Main Idea
Build an AI fraud detector for digital banking transactions, explain its decisions, test whether adversarial manipulation can fool it, compare multiple attacks, harden the model, and demonstrate the system through a Streamlit dashboard, simulated real-time transaction processing, and concept-drift analysis.

This is an experimental AI/ML security prototype. It is **not** a live banking system and does not use real customer banking data.

---

## 2. Exact Implementation Order

1. Fraud Detection
2. SHAP Explainability
3. Adversarial Attack
4. Multiple Attack Comparison
5. Model Hardening
6. Streamlit Dashboard
7. Real-Time Transaction Simulation
8. Concept Drift

```text
PaySim
  ↓
Fraud Detection
  ↓
SHAP Explainability
  ↓
Adversarial Attack
  ↓
Multiple Attack Comparison
  ↓
Hardening
  ↓
Streamlit Dashboard
  ↓
Real-Time Simulation
  ↓
Concept Drift
```

---

## 3. Objectives

1. Build a machine-learning model to detect fraudulent banking transactions.
2. Handle severe class imbalance correctly using SMOTE on training data only.
3. Evaluate using Precision, Recall, F1-score, and PR-AUC.
4. Explain model decisions using SHAP.
5. Generate adversarial evasion examples that attempt to bypass the detector.
6. Compare multiple compatible adversarial attacks.
7. Harden the detector using adversarial training.
8. Compare baseline and hardened robustness.
9. Provide a simple Streamlit prototype.
10. Simulate transactions arriving in real-time style using PaySim rows.
11. Study performance changes over simulated time using concept drift.
12. Produce clear figures, metrics, and security conclusions.

---

## 4. Dataset — PaySim

PaySim is a synthetic mobile-money transaction dataset used for fraud-detection research.

Typical columns:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

### Dataset Rules

- `isFraud` is the target label.
- Drop `isFlaggedFraud` because it may leak fraud-related information.
- Encode `type` correctly.
- `nameOrig` and `nameDest` are high-cardinality identifiers and should not be used in the baseline model without strong justification.
- Useful balance-difference features may be engineered.
- Never modify the original CSV in place.
- Never commit the PaySim CSV to GitHub.
- Store the dataset in Google Drive.

Example Colab path:

```text
/content/drive/MyDrive/AI_Fraud_Adversarial/data/paysim.csv
```

---

## 5. Development Workflow

### Codex
Use Codex to:
- create/edit project files,
- implement reusable Python modules,
- update the Colab notebook,
- run lightweight tests,
- debug errors.

Codex is not the main environment for full PaySim experiments.

### Google Colab
Use Google Colab to actually run:
- PaySim loading,
- preprocessing,
- SMOTE,
- model training,
- SHAP,
- adversarial attacks,
- hardening,
- evaluation,
- figures and metrics.

### GitHub
Use GitHub for source-code version control.

Do **not** upload:
- PaySim CSV,
- secrets,
- unnecessary large artifacts.

### Google Drive
Use Google Drive for:
- PaySim,
- saved models if needed,
- figures,
- metrics,
- persistent experiment outputs.

---

## 6. Phase 1 — Fraud Detection

### Goal
Build a reliable baseline fraud detector.

### Steps
1. Load PaySim from a configurable path.
2. Validate expected columns.
3. Inspect shape, missing values, transaction types, and class distribution.
4. Drop `isFlaggedFraud`.
5. Exclude unsuitable identifiers.
6. Encode `type`.
7. Engineer useful balance-difference features.
8. Split into training and untouched test sets.
9. Use stratification on `isFraud`.
10. Apply SMOTE **only** to training data.
11. Train an XGBoost baseline classifier.
12. Evaluate on untouched test data.

### Main Metrics
- Precision
- Recall
- F1-score
- PR-AUC
- Confusion matrix

Accuracy is supplementary only. Fraud **Recall** is especially important.

---

## 7. Phase 2 — SHAP Explainability

### Goal
Explain why the baseline model predicts a transaction as fraud or legitimate.

### Outputs

#### Global Explanation
- SHAP feature importance
- SHAP summary plot
- ranked important features

#### Local Explanation
Explain at least:
- one correctly detected fraud transaction,
- one correctly detected legitimate transaction.

Show which features push the decision toward fraud or legitimate.

### Rules
- Reuse the Phase 1 model.
- Reuse the same preprocessing and feature names.
- Do not retrain just for SHAP.
- Do not invent explanations; use actual SHAP values.

---

## 8. Phase 3 — Adversarial Evasion Attack

### Goal
Test whether constrained adversarial feature changes can make fraud bypass the baseline model.

### Tool
IBM Adversarial Robustness Toolbox (ART).

### Threat Model
Before attacking, define:
- mutable features,
- immutable features,
- permitted ranges,
- feature dependencies.

### Rules
Never modify:
- `isFraud`,
- leakage fields,
- identifiers,
- invalid categorical encodings.

Do not create impossible negative monetary values.

Use held-out **test fraud samples** for attack evaluation.

### Metrics
- Clean Recall
- Recall Under Attack
- Recall Drop
- Attack Success / Evasion Rate
- Number of samples attacked
- Successful evasion count
- Perturbation statistics

Adversarial **test** samples are evaluation-only and must never be used for training.

---

## 9. Phase 4 — Multiple Attack Comparison

### Goal
Compare multiple compatible adversarial attacks.

### Requirements
- At least 2 compatible attacks; 3 if practical.
- Verify estimator/attack compatibility.
- Use comparable evaluation samples and feature constraints.

### Compare
For each attack:
- Recall Under Attack
- Recall Drop
- Attack Success Rate
- Perturbation statistics
- Runtime if practical

Create:
- comparison table,
- comparison charts,
- strongest-attack analysis.

---

## 10. Phase 5 — Model Hardening

### Goal
Improve adversarial robustness using adversarial training.

### Critical Rule
Never train using adversarial test examples from Phases 3 or 4.

Correct flow:

```text
Training Fraud Samples
        ↓
Generate NEW Adversarial TRAIN Examples
        ↓
Add to Training Data
        ↓
Retrain
        ↓
Hardened Model
```

Then:

```text
Untouched Test Data
        ↓
Fresh Adversarial Attack
        ↓
Evaluate Hardened Model
```

### Compare Baseline vs Hardened
- Clean Precision
- Clean Recall
- Clean F1
- Clean PR-AUC
- Recall Under Attack
- Attack Success Rate
- Recall Recovery
- Clean-performance trade-off

Keep baseline and hardened models separate.

---

## 11. Phase 6 — Streamlit Dashboard

### Goal
Create a simple interactive demonstration.

### Dashboard Sections
1. Project Overview
2. Baseline Performance
3. SHAP Explainability
4. Attack Comparison
5. Baseline vs Hardened
6. Single Transaction Prediction
7. Real-Time Simulation section/page

### Single Transaction Prediction
Show:
- Fraud / Legitimate prediction
- fraud probability/risk score if available
- baseline result
- hardened result
- optional local SHAP explanation

Required notice:

> Research prototype using synthetic PaySim data. Not a live banking system.

---

## 12. Phase 7 — Real-Time Transaction Simulation

### Goal
Simulate transactions arriving one by one or in small batches.

This is **not** a live bank feed.

```text
PaySim Rows
    ↓
Simulation Engine
    ↓
Existing Preprocessing
    ↓
Hardened Model
    ↓
Streamlit Dashboard
```

### Display
For each transaction:
- transaction number,
- type,
- amount,
- fraud probability,
- prediction,
- actual label,
- correct/incorrect result.

### Running Metrics
- total processed,
- detected fraud,
- missed fraud,
- false positives,
- running Recall.

Optional:
- adjustable delay,
- Start/Stop controls,
- live risk chart.

---

## 13. Phase 8 — Concept Drift

### Goal
Study whether model performance changes across simulated time periods.

Use PaySim `step` information where methodologically appropriate.

### Main Experiment
1. Train on earlier-period transactions.
2. Evaluate on later-period transactions.
3. Do not use a random split for the main drift experiment.
4. Apply SMOTE only to the training period.
5. Evaluate several later windows if practical.

### Metrics
- Precision
- Recall
- F1
- PR-AUC

If performance degrades, simulate retraining with newer available training data and measure recovery.

### Limitation
PaySim is synthetic, so this demonstrates **simulated** temporal/concept drift, not proof of real-world banking drift.

---

## 14. Data-Leakage Rules

Always follow these rules:

1. Split before SMOTE.
2. SMOTE only training data.
3. Never SMOTE validation/test data.
4. Fit preprocessing on training data where required.
5. Keep final test data evaluation-only.
6. Never train on adversarial test examples.
7. Generate hardening examples only from training data.
8. Do not tune using final test results.
9. Do not overwrite the baseline model after hardening.
10. Keep baseline and hardened metrics separate.

---

## 15. Reproducibility and Colab Rules

- Use fixed random seeds.
- Provide `DEVELOPMENT_MODE` for quick tests.
- Provide `FULL_MODE` for final experiments.
- Do not simply use the first N rows if that removes meaningful fraud examples.
- Keep memory use reasonable for free Google Colab.
- Use configurable paths.
- Save actual results.
- Never fabricate metrics.
- Never type fake experimental values into final tables.

---

## 16. Recommended Project Structure

```text
AI_Fraud_Adversarial/
│
├── PROJECT_BRIEF.md
├── AGENTS.md
├── README.md
├── requirements-colab.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── fraud_adversarial_colab.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── explainability.py
│   ├── attack.py
│   ├── attack_comparison.py
│   ├── hardening.py
│   ├── dashboard_utils.py
│   ├── realtime_simulation.py
│   └── concept_drift.py
│
├── app.py
│
├── tests/
│   ├── test_smoke.py
│   └── test_data_leakage.py
│
└── outputs/
    ├── metrics/
    ├── figures/
    ├── models/
    └── shap/
```

---

## 17. Logical Security Architecture

### Data Storage Zone
- PaySim in Google Drive
- dataset separate from code
- original CSV not modified

### Trusted Model-Building Zone
- Google Colab
- preprocessing
- train/test separation
- training-only SMOTE
- baseline model training
- SHAP analysis

### Isolated Adversarial Testing Zone
- copies of evaluation fraud samples
- ART attack generation
- constrained perturbation
- attack metrics

### Blue-Team Hardening Zone
- training-side adversarial examples only
- adversarial training
- hardened model
- fresh test attack

### Results / Model Storage
- metrics
- figures
- baseline model
- hardened model

Real banking APIs, firewalls, authentication, and live payment processing are outside the implemented prototype unless explicitly added later.

---

## 18. Main Tools

| Tool | Purpose |
|---|---|
| Python | Main language |
| Google Colab | Main experiment environment |
| pandas / NumPy | Data handling |
| scikit-learn | Preprocessing, splitting, metrics |
| imbalanced-learn | SMOTE |
| XGBoost | Baseline fraud classifier |
| SHAP | Explainable AI |
| IBM ART | Adversarial attacks |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive dashboard |
| joblib | Save/load models |
| pytest | Lightweight testing |
| GitHub | Source-code version control |
| Google Drive | Dataset and persistent outputs |

---

## 19. Expected Final Outputs

- baseline fraud model,
- baseline metrics,
- SHAP global/local explanations,
- adversarial attack results,
- multiple-attack comparison,
- hardened model,
- before/after robustness comparison,
- Streamlit dashboard,
- real-time transaction simulation,
- concept-drift analysis,
- figures and tables,
- security recommendations.

---

## 20. Scope and Limitations

This is a university research prototype.

It does not claim to:
- process real bank transactions,
- connect to live banking APIs,
- replace production fraud-monitoring systems,
- resist every adversarial attack,
- prove real-world concept drift using PaySim.

The purpose is to demonstrate fraud detection, explainability, adversarial testing, hardening, and monitoring concepts in a reproducible Google Colab environment.

---

## 21. Working Rules for Codex

When working on this project:

- Read `PROJECT_BRIEF.md` and `AGENTS.md` first.
- Implement only one phase at a time.
- Do not rewrite working phases unnecessarily.
- Reuse preprocessing and saved artifacts consistently.
- Keep reusable logic under `src/`.
- Keep the Colab notebook as the main experiment interface.
- Test development mode before full experiments.
- Do not fabricate results.
- Explain major decisions clearly.
- Stop after the requested phase.
