"""Baseline evaluation and JSON-safe metric persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_binary_classifier(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Evaluate once on untouched test data; accuracy is supplementary."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    precision_curve, recall_curve, thresholds = precision_recall_curve(
        y_test, probabilities
    )
    return {
        "metrics": {
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "accuracy_supplementary": float(accuracy_score(y_test, predictions)),
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "predictions": predictions,
        "probabilities": probabilities,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "thresholds": thresholds,
    }


def save_metrics(result: dict[str, Any], path: str | Path) -> None:
    """Save only actual computed, JSON-safe evaluation values."""
    serializable = {
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        "classification_report": result["classification_report"],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def evaluate_adversarial_evasion(
    model: Any,
    X_clean: Any,
    X_adversarial: Any,
    *,
    full_test_clean_fraud_recall: float,
    attack_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Evaluate actual fraud-to-legitimate evasions on selected test fraud."""
    import numpy as np
    import pandas as pd

    clean_predictions = np.asarray(model.predict(X_clean)).astype(int)
    adversarial_predictions = np.asarray(model.predict(X_adversarial)).astype(int)
    clean_probabilities = np.asarray(model.predict_proba(X_clean))[:, 1]
    adversarial_probabilities = np.asarray(model.predict_proba(X_adversarial))[:, 1]
    if not np.all(clean_predictions == 1):
        raise ValueError("Attack population must be originally correctly detected fraud")
    successful = adversarial_predictions == 0
    differences = X_adversarial.to_numpy(dtype=float) - X_clean.to_numpy(dtype=float)
    l2 = np.linalg.norm(differences, ord=2, axis=1)
    linf = np.linalg.norm(differences, ord=np.inf, axis=1)
    count = len(X_clean)
    recall_under_attack = float(adversarial_predictions.mean())
    metrics = {
        "attack_name": "Targeted HopSkipJump",
        "attack_success_definition": (
            "successful fraud-to-legitimate evasions divided by originally "
            "correctly detected attacked test fraud"
        ),
        "full_test_clean_fraud_recall": float(full_test_clean_fraud_recall),
        "attacked_sample_clean_recall": float(clean_predictions.mean()),
        "recall_under_attack": recall_under_attack,
        "recall_drop_on_attacked_sample": float(
            clean_predictions.mean() - recall_under_attack
        ),
        "attack_success_rate": float(successful.mean()),
        "number_attacked": int(count),
        "successful_evasions": int(successful.sum()),
        "average_l2_perturbation": float(l2.mean()),
        "max_l2_perturbation": float(l2.max()),
        "average_linf_perturbation": float(linf.mean()),
        "max_linf_perturbation": float(linf.max()),
        "evaluation_only": True,
        "used_for_training": False,
    }
    if attack_metadata:
        metrics["attack_execution"] = attack_metadata
    samples = pd.DataFrame(
        {
            "source_test_index": X_clean.index,
            "actual_label": 1,
            "clean_prediction": clean_predictions,
            "adversarial_prediction": adversarial_predictions,
            "clean_fraud_probability": clean_probabilities,
            "adversarial_fraud_probability": adversarial_probabilities,
            "successful_evasion": successful,
            "l2_perturbation": l2,
            "linf_perturbation": linf,
        }
    )
    for feature in X_clean.columns:
        samples[f"clean_{feature}"] = X_clean[feature].to_numpy()
        samples[f"adversarial_{feature}"] = X_adversarial[feature].to_numpy()
    return metrics, samples


def save_attack_outputs(
    metrics: dict[str, Any],
    samples: Any,
    metrics_path: str | Path,
    samples_path: str | Path,
) -> None:
    """Save only actually computed attack metrics and sample outcomes."""
    metrics_output = Path(metrics_path)
    samples_output = Path(samples_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    samples.to_csv(samples_output, index=False)
