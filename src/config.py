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

# Phase 2 keeps global explanations bounded for free Colab memory.
SHAP_MAX_SAMPLES = int(os.getenv("SHAP_MAX_SAMPLES", "1000"))
SHAP_MAX_FRAUD_SAMPLES = int(os.getenv("SHAP_MAX_FRAUD_SAMPLES", "250"))
SHAP_MAX_DISPLAY = int(os.getenv("SHAP_MAX_DISPLAY", "15"))
SHAP_IMPORTANCE_PATH = SHAP_DIR / "global_feature_importance.csv"
SHAP_INTERPRETATION_PATH = SHAP_DIR / "phase2_interpretation.json"

# Phase 3: bounded, test-only targeted HopSkipJump defaults for free Colab.
ATTACK_SAMPLE_SIZE = int(os.getenv("ATTACK_SAMPLE_SIZE", "20"))
ATTACK_RELATIVE_BOUND = float(os.getenv("ATTACK_RELATIVE_BOUND", "0.10"))
ATTACK_MAX_ITER = int(os.getenv("ATTACK_MAX_ITER", "10"))
ATTACK_MAX_EVAL = int(os.getenv("ATTACK_MAX_EVAL", "500"))
ATTACK_INIT_EVAL = int(os.getenv("ATTACK_INIT_EVAL", "50"))
ATTACK_INIT_SIZE = int(os.getenv("ATTACK_INIT_SIZE", "30"))
PHASE3_FIGURES_DIR = FIGURES_DIR / "phase3"
PHASE3_METRICS_PATH = METRICS_DIR / "phase3_attack_metrics.json"
PHASE3_SAMPLES_PATH = METRICS_DIR / "phase3_attacked_samples.csv"

# Phase 4 uses the same population/bounds with practical attack-specific budgets.
PHASE4_ATTACK_SAMPLE_SIZE = int(
    os.getenv("PHASE4_ATTACK_SAMPLE_SIZE", str(ATTACK_SAMPLE_SIZE))
)
BOUNDARY_MAX_ITER = int(os.getenv("BOUNDARY_MAX_ITER", "50"))
BOUNDARY_NUM_TRIAL = int(os.getenv("BOUNDARY_NUM_TRIAL", "10"))
BOUNDARY_SAMPLE_SIZE = int(os.getenv("BOUNDARY_SAMPLE_SIZE", "10"))
ZOO_MAX_ITER = int(os.getenv("ZOO_MAX_ITER", "20"))
ZOO_LEARNING_RATE = float(os.getenv("ZOO_LEARNING_RATE", "0.05"))
ZOO_NB_PARALLEL = int(os.getenv("ZOO_NB_PARALLEL", "5"))
PHASE4_FIGURES_DIR = FIGURES_DIR / "phase4"
PHASE4_COMPARISON_CSV_PATH = METRICS_DIR / "phase4_attack_comparison.csv"
PHASE4_COMPARISON_JSON_PATH = METRICS_DIR / "phase4_attack_comparison.json"

# Phase 5 uses training-origin fraud only and a small common test population.
HARDENING_TRAIN_ATTACK_SAMPLE_SIZE = int(
    os.getenv("HARDENING_TRAIN_ATTACK_SAMPLE_SIZE", "20" if DEVELOPMENT_MODE else "50")
)
HARDENING_TEST_ATTACK_SAMPLE_SIZE = int(
    os.getenv("HARDENING_TEST_ATTACK_SAMPLE_SIZE", "10" if DEVELOPMENT_MODE else "20")
)
PHASE5_FIGURES_DIR = FIGURES_DIR / "phase5"
HARDENED_MODEL_PATH = MODELS_DIR / "hardened_model.joblib"
PHASE5_METRICS_PATH = METRICS_DIR / "phase5_hardened_metrics.json"
PHASE5_COMPARISON_CSV_PATH = METRICS_DIR / "phase5_hardening_comparison.csv"

# Optional robustness-improvement experiment. These defaults remain bounded for
# Colab while giving adversarial rows enough influence in the full SMOTE set.
ROBUST_HARDENING_TRAIN_SAMPLE_SIZE = int(
    os.getenv("ROBUST_HARDENING_TRAIN_SAMPLE_SIZE", "100" if DEVELOPMENT_MODE else "400")
)
ROBUST_HARDENING_TEST_SAMPLE_SIZE = int(
    os.getenv("ROBUST_HARDENING_TEST_SAMPLE_SIZE", "20" if DEVELOPMENT_MODE else "50")
)
ROBUST_HARDENING_ADVERSARIAL_WEIGHT = float(
    os.getenv("ROBUST_HARDENING_ADVERSARIAL_WEIGHT", "75.0")
)
ROBUST_HARDENING_HSJ_SHARE = float(
    os.getenv("ROBUST_HARDENING_HSJ_SHARE", "0.45")
)
ROBUST_HARDENING_BOUNDARY_SHARE = float(
    os.getenv("ROBUST_HARDENING_BOUNDARY_SHARE", "0.45")
)
