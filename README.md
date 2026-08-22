# AI-Based Fraud Detection, Explainability, and Adversarial Robustness for Digital Banking Transactions

An experimental university project that uses the synthetic PaySim dataset to build and explain a fraud detector, evaluate constrained adversarial evasion, harden the model, and demonstrate monitoring concepts. It is a research prototype, not a live banking system.

## Project flow

```text
PaySim → Fraud Detection → SHAP Explainability → Adversarial Attack
       → Multiple Attack Comparison → Model Hardening → Streamlit Dashboard
       → Real-Time Transaction Simulation → Concept Drift
```

Implementation proceeds one phase at a time in this exact order:

1. Fraud Detection
2. SHAP Explainability
3. Adversarial Attack
4. Multiple Attack Comparison
5. Model Hardening
6. Streamlit Dashboard
7. Real-Time Transaction Simulation
8. Concept Drift

Phase 1 provides the executable baseline fraud-detection pipeline. No experimental results are committed; reported values must come from actual execution.

## Environments and data

Google Colab is the main experiment environment, and the Colab notebook is the main execution interface. GitHub stores code only. Store the PaySim CSV in Google Drive (the default configured path is `/content/drive/MyDrive/AI_Fraud_Adversarial/data/paysim.csv`) and never commit it to GitHub.

- `DEVELOPMENT_MODE`: fast, memory-aware checks on a representative subset while retaining fraud examples.
- `FULL_MODE`: final experiments on the intended dataset scope within available Colab resources.

The active mode and paths are configured in `src/config.py`.

## Directory structure

```text
AI_Fraud_Adversarial/
├── PROJECT_BRIEF.md
├── AGENTS.md
├── README.md
├── requirements-colab.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── fraud_adversarial_colab.ipynb
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
├── app.py
├── tests/
│   ├── test_smoke.py
│   └── test_data_leakage.py
└── outputs/
    ├── metrics/
    ├── figures/
    ├── models/
    └── shap/
```

## Results and artifacts

Actual metrics, figures, serialized models, and SHAP artifacts are saved under `outputs/metrics`, `outputs/figures`, `outputs/models`, and `outputs/shap`. Google Drive should be used for persistent Colab outputs. Baseline and hardened artifacts will use separate names and will never overwrite one another.

## Tests

After installing dependencies, run:

```bash
python -m pytest tests/test_smoke.py tests/test_data_leakage.py -q
```

For all Phase 1 unit tests, run:

```bash
python -m pytest tests/test_smoke.py tests/test_data_leakage.py tests/test_phase1.py -q
```
