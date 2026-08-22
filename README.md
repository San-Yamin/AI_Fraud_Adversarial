# AI-Based Fraud Detection and Adversarial Robustness

An end-to-end university research project for detecting fraudulent digital transactions,
explaining model decisions, testing adversarial evasion, improving robustness, and
monitoring model behaviour through an interactive dashboard.

> **Research notice:** This project uses synthetic PaySim data. It is an experimental
> prototype, not a live banking system, and it is not connected to real financial APIs.

## What this project demonstrates

The project follows one reproducible workflow:

```text
PaySim data
    ↓
Baseline fraud detection
    ↓
SHAP explainability
    ↓
Constrained adversarial evasion
    ↓
Multiple attack comparison
    ↓
Adversarial model hardening
    ↓
Streamlit dashboard
    ↓
Simulated transaction stream
    ↓
Step-based concept drift analysis
```

The main model family is XGBoost. Class imbalance is handled with SMOTE on training
data only. The primary evaluation metrics are Precision, Recall, F1, and PR-AUC, with
fraud Recall treated as especially important.

## Completed phases

| Phase | Component | Purpose |
|---|---|---|
| 1 | Baseline fraud detection | Preprocess PaySim, train XGBoost, and evaluate on untouched test data |
| 2 | SHAP explainability | Provide global feature importance and local transaction explanations |
| 3 | Adversarial evasion | Attack correctly detected test fraud using realistic feature constraints |
| 4 | Attack comparison | Compare compatible ART attacks on a common evaluation population |
| 5 | Model hardening | Generate training-side adversarial examples and train a separate hardened model |
| 6 | Streamlit dashboard | Present saved metrics, figures, explanations, and predictions |
| 7 | Real-time-style simulation | Replay a small PaySim sequence through the hardened model |
| 8 | Concept drift | Evaluate performance and feature-distribution changes over PaySim `step` windows |

All eight implementation phases are complete. Experimental values are never fabricated;
final metrics and figures come from actual notebook or dashboard execution.

## Important methodology safeguards

- `isFraud` is the target and is never used as an input feature.
- `isFlaggedFraud` is removed because it can leak fraud-related information.
- High-cardinality identifiers such as `nameOrig` and `nameDest` are excluded.
- The train/test split happens before SMOTE.
- SMOTE is applied only to training data.
- Held-out test transactions remain unchanged and evaluation-only.
- Phase 3 and Phase 4 adversarial test examples are never used for training.
- Phase 5 adversarial training examples originate only from training fraud.
- The baseline and hardened models are stored separately.
- Random seeds make sampling and evaluation reproducible.
- Adversarial changes follow explicit ranges, mutability rules, and feature dependencies.

## Technology stack

- Python
- pandas and NumPy
- scikit-learn and imbalanced-learn
- XGBoost
- SHAP
- IBM Adversarial Robustness Toolbox (ART)
- Matplotlib and Seaborn
- Streamlit
- Google Colab and Google Drive

## Repository structure

```text
AI_Fraud_Adversarial/
├── app.py                         # Streamlit dashboard
├── notebooks/
│   └── fraud_adversarial_colab.ipynb
├── src/
│   ├── config.py                  # Paths, seeds, run modes, and experiment limits
│   ├── data_loader.py             # Memory-aware PaySim loading
│   ├── preprocessing.py           # Feature engineering, encoding, and training-only SMOTE
│   ├── train.py                   # Split and baseline model training
│   ├── evaluate.py                # Clean and adversarial evaluation metrics
│   ├── visualization.py           # Experiment figures
│   ├── explainability.py          # Phase 2 SHAP workflow
│   ├── attack.py                  # Constrained ART attacks and threat model
│   ├── attack_comparison.py       # Phase 4 comparison workflow
│   ├── hardening.py               # Leakage-safe adversarial training
│   ├── dashboard_utils.py         # Artifact loading and transaction inference
│   ├── realtime_simulation.py     # Phase 7 stream simulation
│   └── concept_drift.py           # Phase 8 chronological analysis
├── tests/                         # Unit, leakage, attack, dashboard, and drift tests
├── outputs/                       # Local output structure; generated results are not committed
├── data/                          # Dataset documentation; PaySim CSV is not committed
├── requirements-colab.txt         # Full Colab experiment environment
├── requirements.txt               # Streamlit deployment environment
├── PROJECT_BRIEF.md                # Complete project specification
└── AGENTS.md                       # Permanent implementation rules
```

## Data and saved artifacts

PaySim is a synthetic mobile-money dataset. Download it separately and keep the original
CSV unchanged. The expected Colab location is:

```text
/content/drive/MyDrive/AI_Fraud_Adversarial/data/paysim.csv
```

The default persistent output directory is:

```text
/content/drive/MyDrive/AI_Fraud_Adversarial/outputs
```

Generated artifacts use the following structure:

```text
outputs/
├── models/       # Baseline model, hardened model, preprocessor, feature names
├── metrics/      # JSON and CSV evaluation results
├── figures/      # Performance, attack, hardening, simulation, and drift figures
└── shap/         # Global and local SHAP outputs
```

The dataset, serialized models, and full experimental outputs are intentionally not
committed to GitHub. They are stored in Google Drive when experiments run in Colab.

## Run the experiments in Google Colab

1. Open `notebooks/fraud_adversarial_colab.ipynb` in Google Colab.
2. Clone the repository to `/content/AI_Fraud_Adversarial` if it is not already present.
3. Mount Google Drive.
4. Place `paysim.csv` in the configured Drive data directory.
5. Run notebook cells in numerical order.
6. Keep the runtime alive when practical; deterministic reconstruction is used after a restart.

Install the Colab dependencies with:

```bash
pip install -r requirements-colab.txt
```

### Development and full modes

The run mode is controlled through `RUN_MODE`:

```bash
RUN_MODE=DEVELOPMENT_MODE
```

- `DEVELOPMENT_MODE` uses small, class-aware, reproducible samples for practical Colab testing.
- `FULL_MODE` uses the intended larger experiment scope within available memory.

Other paths and experiment budgets can be overridden with environment variables defined
in `src/config.py`.

## Run the dashboard locally

Create an environment and install the dashboard dependencies:

```bash
git clone https://github.com/San-Yamin/AI_Fraud_Adversarial.git
cd AI_Fraud_Adversarial
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Point the app to saved outputs and start Streamlit:

```bash
OUTPUT_DIR=/path/to/AI_Fraud_Adversarial/outputs \
PAYSIM_DATASET_PATH=/path/to/paysim.csv \
streamlit run app.py
```

The dataset path is needed only for the transaction simulation and concept-drift pages.
The overview and saved-results pages do not load the full PaySim dataset.

## Dashboard sections

The Streamlit application includes:

1. **Project Overview** — workflow and research scope.
2. **Baseline Performance** — saved Precision, Recall, F1, PR-AUC, confusion matrix, and PR curve.
3. **SHAP Explainability** — global importance, beeswarm, fraud waterfall, and legitimate waterfall.
4. **Attack Comparison** — attack table and evasion, recall, perturbation, and runtime figures.
5. **Baseline vs Hardened** — clean and adversarial robustness comparison.
6. **Single Transaction** — baseline and hardened predictions using the saved preprocessor.
7. **Real-Time Simulation** — start, pause, reset, monitor, and export a simulated PaySim stream.
8. **Concept Drift** — chronological metrics and feature-drift analysis across PaySim windows.

The dashboard reads existing artifacts and does not retrain either model.

## Streamlit Community Cloud deployment

The public app is deployed from `app.py` on the `main` branch. Streamlit Cloud uses the
root `requirements.txt` file.

The source repository does not contain generated artifacts. On Streamlit Cloud, the app
downloads the public deployment artifact bundle into temporary storage, validates the ZIP,
and loads the saved models, metrics, and figures from its extracted `outputs/` directory.
Set `DEPLOYMENT_ARTIFACT_FILE_ID` to replace the configured Google Drive bundle without a
code change. Do not publish the full PaySim CSV, secrets, credentials, or private data.

Recommended deployment settings:

- Entrypoint: `app.py`
- Python: 3.11
- Output path: resolved automatically from the downloaded deployment bundle
- Demo data: a small reproducible PaySim sample is still required for Phase 7 and Phase 8

## Testing

Install the complete development environment, then run all tests:

```bash
python -m pip install -r requirements-colab.txt
python -m pytest -q
```

The suite covers:

- Phase 1 preprocessing, splitting, training, and evaluation
- train/test leakage protections
- SHAP sampling and explanation helpers
- adversarial constraints and attack evaluation
- multiple-attack comparison
- leakage-safe model hardening
- dashboard artifact loading and prediction preprocessing
- real-time sequence creation, running metrics, and reset behaviour
- concept-drift windows, metrics, drift scores, and no-target-leakage rules

## Key output files

Important saved files include:

```text
outputs/models/baseline_model.joblib
outputs/models/hardened_model.joblib
outputs/models/baseline_preprocessor.joblib
outputs/models/feature_names.joblib
outputs/metrics/phase1_baseline_metrics.json
outputs/metrics/phase3_attack_metrics.json
outputs/metrics/phase4_attack_comparison.csv
outputs/metrics/phase4_attack_comparison.json
outputs/metrics/phase5_hardened_metrics.json
outputs/metrics/phase5_hardening_comparison.csv
outputs/metrics/phase7_simulation_results.csv
outputs/metrics/phase8_concept_drift.csv
outputs/metrics/phase8_concept_drift.json
```

SHAP plots and the remaining presentation figures are saved under `outputs/shap/` and
`outputs/figures/`.

## Limitations

- PaySim is synthetic, so findings do not establish real-world banking performance.
- Adversarial results depend on the defined threat model and practical Colab query budgets.
- Real-time processing is a replay simulation, not a production transaction feed.
- Concept drift is simulated through PaySim `step` ordering and is not proof of production drift.
- The application is a research demonstration and must not be used to make live financial decisions.

## Academic context

- **Project type:** Individual university project
- **Course:** Network & Internet Security (CST-8415)
- **Primary environment:** Google Colab
- **Dataset:** PaySim synthetic financial transactions

For the exact methodology and phase requirements, see `PROJECT_BRIEF.md`.
