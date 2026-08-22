"""Central, environment-overridable project configuration."""

from __future__ import annotations

import os
from pathlib import Path

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

DEVELOPMENT_MODE_LABEL = "DEVELOPMENT_MODE"
FULL_MODE_LABEL = "FULL_MODE"
RUN_MODE = os.getenv("RUN_MODE", DEVELOPMENT_MODE_LABEL).upper()
if RUN_MODE not in {DEVELOPMENT_MODE_LABEL, FULL_MODE_LABEL}:
    raise ValueError(
        f"RUN_MODE must be {DEVELOPMENT_MODE_LABEL!r} or {FULL_MODE_LABEL!r}"
    )
DEVELOPMENT_MODE = RUN_MODE == DEVELOPMENT_MODE_LABEL
FULL_MODE = RUN_MODE == FULL_MODE_LABEL

GOOGLE_DRIVE_PROJECT_DIR = Path(
    os.getenv("GOOGLE_DRIVE_PROJECT_DIR", "/content/drive/MyDrive/AI_Fraud_Adversarial")
)
DATASET_PATH = Path(
    os.getenv("PAYSIM_DATASET_PATH", str(GOOGLE_DRIVE_PROJECT_DIR / "data/paysim.csv"))
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(GOOGLE_DRIVE_PROJECT_DIR / "outputs")))
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
SHAP_DIR = OUTPUT_DIR / "shap"

TARGET_COLUMN = "isFraud"
LEAKAGE_COLUMN = "isFlaggedFraud"

# Phase 1 defaults. Development sampling is class-aware and reproducible.
DEVELOPMENT_NORMAL_SAMPLES = int(os.getenv("DEVELOPMENT_NORMAL_SAMPLES", "50000"))
DEVELOPMENT_FRAUD_SAMPLES = int(os.getenv("DEVELOPMENT_FRAUD_SAMPLES", "2000"))
CSV_CHUNK_SIZE = int(os.getenv("CSV_CHUNK_SIZE", "250000"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.20"))

# A minority/majority ratio of 0.10 avoids creating a huge 1:1 training set.
SMOTE_SAMPLING_STRATEGY = float(os.getenv("SMOTE_SAMPLING_STRATEGY", "0.10"))

PHASE1_FIGURES_DIR = FIGURES_DIR / "phase1"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "baseline_preprocessor.joblib"
BASELINE_METRICS_PATH = METRICS_DIR / "phase1_baseline_metrics.json"
