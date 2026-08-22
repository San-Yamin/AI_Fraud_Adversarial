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
