"""Leakage-safe adversarial training and Phase 5 comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.train import build_baseline_model


def select_adversarial_training_sources(
    baseline_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Select reproducible, correctly detected fraud from the training split only."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indices must match")
    fraud = X_train.loc[y_train == 1]
    detected = np.asarray(baseline_model.predict(fraud)).astype(int) == 1
    eligible = fraud.loc[detected]
    if eligible.empty:
        raise ValueError("No correctly detected training fraud is available for hardening")
    selected = eligible.sample(
        n=min(sample_size, len(eligible)), random_state=random_state
    ).copy()
    labels = y_train.loc[selected.index].astype("int8").copy()
    metadata = {
        "source_split": "training",
        "total_training_fraud": int((y_train == 1).sum()),
        "correctly_detected_training_fraud": int(detected.sum()),
        "selected_training_fraud": int(len(selected)),
        "random_seed": int(random_state),
        "phase3_or_phase4_test_samples_reused": False,
    }
    return selected, labels, metadata


def build_augmented_training_set(
    X_clean_training: pd.DataFrame,
    y_clean_training: Any,
    X_adversarial_training: pd.DataFrame,
    *,
    source_training_indices: pd.Index,
    untouched_test_indices: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Append label-1 adversarial TRAIN examples without accepting test rows."""
    if X_adversarial_training.empty:
        raise ValueError("At least one adversarial training example is required")
    if list(X_clean_training.columns) != list(X_adversarial_training.columns):
        raise ValueError("Clean and adversarial training feature columns differ")
    if not X_adversarial_training.index.equals(source_training_indices):
        raise ValueError("Adversarial rows must retain their training-source indices")
    if untouched_test_indices is not None:
        overlap = source_training_indices.intersection(untouched_test_indices)
        if len(overlap):
            raise ValueError("Adversarial training sources overlap the test split")

    clean_y = pd.Series(np.asarray(y_clean_training), dtype="int8").reset_index(drop=True)
    adversarial_y = pd.Series(
        np.ones(len(X_adversarial_training), dtype="int8"), name=clean_y.name
    )
    X_augmented = pd.concat(
        [X_clean_training.reset_index(drop=True), X_adversarial_training.reset_index(drop=True)],
        ignore_index=True,
    ).astype("float32")
    y_augmented = pd.concat([clean_y, adversarial_y], ignore_index=True)
    metadata = {
        "clean_training_rows": int(len(X_clean_training)),
        "adversarial_training_rows": int(len(X_adversarial_training)),
        "augmented_training_rows": int(len(X_augmented)),
        "adversarial_label": 1,
        "test_rows_used_for_training": 0,
    }
    return X_augmented, y_augmented, metadata


def train_hardened_model(
    X_augmented: pd.DataFrame,
    y_augmented: pd.Series,
    *,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Fit a new XGBoost instance with the baseline model configuration."""
    model = build_baseline_model(random_state=random_state)
    model.fit(X_augmented, y_augmented)
    return model


def select_common_correct_test_fraud(
    baseline_model: Any,
    hardened_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose the same untouched test fraud correctly detected by both models."""
    fraud = X_test.loc[y_test == 1]
    baseline_predictions = np.asarray(baseline_model.predict(fraud)).astype(int)
    hardened_predictions = np.asarray(hardened_model.predict(fraud)).astype(int)
    eligible = fraud.loc[(baseline_predictions == 1) & (hardened_predictions == 1)]
    if eligible.empty:
        raise ValueError("No common correctly detected test fraud is available")
    selected = eligible.sample(
        n=min(sample_size, len(eligible)), random_state=random_state
    ).copy()
    return selected, {
        "source_split": "untouched_test",
        "total_test_fraud": int(len(fraud)),
        "baseline_clean_fraud_recall": float(baseline_predictions.mean()),
        "hardened_clean_fraud_recall": float(hardened_predictions.mean()),
        "common_correctly_detected_test_fraud": int(len(eligible)),
        "selected_test_fraud": int(len(selected)),
        "used_for_training": False,
        "random_seed": int(random_state),
    }


def build_hardening_comparison(
    baseline_clean_metrics: dict[str, float],
    hardened_clean_metrics: dict[str, float],
    baseline_attacks: dict[str, dict[str, Any]],
    hardened_attacks: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build per-attack baseline-versus-hardened robustness comparisons."""
    if set(baseline_attacks) != set(hardened_attacks):
        raise ValueError("Baseline and hardened models must use the same attacks")
    rows = []
    for attack in baseline_attacks:
        baseline = baseline_attacks[attack]
        hardened = hardened_attacks[attack]
        if baseline["number_attacked"] != hardened["number_attacked"]:
            raise ValueError("Models must be attacked on comparable population sizes")
        rows.append(
            {
                "Attack": attack,
                "Baseline Clean Recall": float(baseline_clean_metrics["recall"]),
                "Baseline Recall Under Attack": float(baseline["recall_under_attack"]),
                "Baseline Recall Drop": float(baseline["recall_drop_on_attacked_sample"]),
                "Baseline Attack Success Rate": float(baseline["attack_success_rate"]),
                "Hardened Clean Recall": float(hardened_clean_metrics["recall"]),
                "Hardened Recall Under Attack": float(hardened["recall_under_attack"]),
                "Hardened Recall Drop": float(hardened["recall_drop_on_attacked_sample"]),
                "Hardened Attack Success Rate": float(hardened["attack_success_rate"]),
                "Recall Recovery": float(
                    hardened["recall_under_attack"] - baseline["recall_under_attack"]
                ),
                "Reduction in Attack Success Rate": float(
                    baseline["attack_success_rate"] - hardened["attack_success_rate"]
                ),
                "Clean Recall Change": float(
                    hardened_clean_metrics["recall"] - baseline_clean_metrics["recall"]
                ),
                "Clean Precision Change": float(
                    hardened_clean_metrics["precision"] - baseline_clean_metrics["precision"]
                ),
                "Clean F1 Change": float(
                    hardened_clean_metrics["f1_score"] - baseline_clean_metrics["f1_score"]
                ),
                "Clean PR-AUC Change": float(
                    hardened_clean_metrics["pr_auc"] - baseline_clean_metrics["pr_auc"]
                ),
            }
        )
    return pd.DataFrame(rows)


def save_phase5_outputs(
    hardened_model: Any,
    hardened_evaluation: dict[str, Any],
    comparison: pd.DataFrame,
    methodology: dict[str, Any],
    model_path: str | Path,
    metrics_path: str | Path,
    comparison_path: str | Path,
) -> None:
    """Save the new model and actual Phase 5 results without baseline overwrite."""
    model_output, metrics_output, comparison_output = map(
        Path, (model_path, metrics_path, comparison_path)
    )
    for path in (model_output, metrics_output, comparison_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hardened_model, model_output)
    comparison.to_csv(comparison_output, index=False)
    payload = {
        "hardened_clean_evaluation": {
            "metrics": hardened_evaluation["metrics"],
            "confusion_matrix": hardened_evaluation["confusion_matrix"],
            "classification_report": hardened_evaluation["classification_report"],
        },
        "robustness_comparison": comparison.to_dict(orient="records"),
        "methodology": methodology,
    }
    metrics_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
