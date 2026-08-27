"""Reusable artifact loading and single-transaction inference for Phase 6."""

from __future__ import annotations

import json
import os
from collections import UserList
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from PIL import Image


PAYMENT_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
MODEL_INPUT_COLUMNS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


def fit_image_to_aspect_ratio(
    path: str | Path,
    aspect_ratio: tuple[int, int] = (16, 9),
    background: tuple[int, int, int] = (255, 255, 255),
    padding_fraction: float = 0.06,
) -> Image.Image:
    """Place a complete figure inside a padded fixed-ratio canvas without cropping."""
    ratio_width, ratio_height = aspect_ratio
    if ratio_width <= 0 or ratio_height <= 0:
        raise ValueError("aspect_ratio values must be positive")
    if not 0 <= padding_fraction < 0.5:
        raise ValueError("padding_fraction must be between 0 and 0.5")

    with Image.open(path) as source:
        figure = source.convert("RGB")

    target_ratio = ratio_width / ratio_height
    if figure.width / figure.height >= target_ratio:
        canvas_width = figure.width
        canvas_height = int(np.ceil(figure.width / target_ratio))
    else:
        canvas_height = figure.height
        canvas_width = int(np.ceil(figure.height * target_ratio))

    available_width = max(1, int(canvas_width * (1 - 2 * padding_fraction)))
    available_height = max(1, int(canvas_height * (1 - 2 * padding_fraction)))
    scale = min(
        available_width / figure.width,
        available_height / figure.height,
    )
    displayed_size = (
        max(1, int(figure.width * scale)),
        max(1, int(figure.height * scale)),
    )
    if displayed_size != figure.size:
        figure = figure.resize(displayed_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    offset = (
        (canvas_width - figure.width) // 2,
        (canvas_height - figure.height) // 2,
    )
    canvas.paste(figure, offset)
    return canvas


class _DashboardRemainderColsList(UserList):
    """Minimal load-compatible form of scikit-learn 1.5/1.6's private list."""

    def __init__(
        self, columns=(), *, future_dtype=None,
        warning_was_emitted=False, warning_enabled=False,
    ):
        super().__init__(columns)
        self.future_dtype = future_dtype
        self.warning_was_emitted = warning_was_emitted
        self.warning_enabled = warning_enabled


def install_sklearn_joblib_compatibility() -> bool:
    """Restore one removed private type needed only to read older preprocessors.

    Scikit-learn 1.5/1.6 serialized ``ColumnTransformer.transformers_`` with
    ``_RemainderColsList``. Newer releases removed that private class, causing
    otherwise valid fitted preprocessors to fail during joblib deserialization.
    """
    import sklearn.compose._column_transformer as column_transformer

    if hasattr(column_transformer, "_RemainderColsList"):
        return False
    column_transformer._RemainderColsList = _DashboardRemainderColsList
    return True


def default_output_directory(project_root: str | Path) -> Path:
    """Prefer explicit or full-mode artifacts before development outputs."""
    root = Path(project_root)
    candidates = []
    if os.getenv("OUTPUT_DIR"):
        candidates.append(Path(os.environ["OUTPUT_DIR"]))
    candidates.extend(
        [
            Path("/content/drive/MyDrive/AI_Fraud_Adversarial/outputs_full_mode"),
            Path("/content/drive/MyDrive/AI_Fraud_Adversarial/outputs"),
            root / "outputs",
        ]
    )
    return next((path for path in candidates if path.exists()), candidates[-1])


def default_dataset_path(project_root: str | Path) -> Path:
    """Resolve PaySim without loading it, supporting local and Colab demos."""
    root = Path(project_root)
    candidates = []
    if os.getenv("PAYSIM_DATASET_PATH"):
        candidates.append(Path(os.environ["PAYSIM_DATASET_PATH"]))
    candidates.extend(
        [
            Path("/content/drive/MyDrive/AI_Fraud_Adversarial/data/paysim.csv"),
            root / "data" / "paysim.csv",
        ]
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    """Return the complete Phase 1–8 dashboard artifact contract."""
    output = Path(output_dir)
    return {
        "baseline_model": output / "models" / "baseline_model.joblib",
        "hardened_model": output / "models" / "hardened_model.joblib",
        "preprocessor": output / "models" / "baseline_preprocessor.joblib",
        "feature_names": output / "models" / "feature_names.joblib",
        "phase1_metrics": output / "metrics" / "phase1_baseline_metrics.json",
        "phase4_csv": output / "metrics" / "phase4_attack_comparison.csv",
        "phase4_json": output / "metrics" / "phase4_attack_comparison.json",
        "phase5_metrics": output / "metrics" / "phase5_hardened_metrics.json",
        "phase5_csv": output / "metrics" / "phase5_hardening_comparison.csv",
        "phase1_confusion": output / "figures" / "phase1" / "baseline_confusion_matrix.png",
        "phase1_pr_curve": output / "figures" / "phase1" / "baseline_precision_recall_curve.png",
        "shap_importance_plot": output / "shap" / "global_feature_importance_bar.png",
        "shap_beeswarm": output / "shap" / "global_summary_beeswarm.png",
        "shap_fraud_waterfall": output / "shap" / "correct_fraud_waterfall.png",
        "shap_legitimate_waterfall": output / "shap" / "correct_legitimate_waterfall.png",
        "shap_importance_csv": output / "shap" / "global_feature_importance.csv",
        "phase4_success": output / "figures" / "phase4" / "attack_success_rate_comparison.png",
        "phase4_recall": output / "figures" / "phase4" / "recall_under_attack_comparison.png",
        "phase4_perturbation": output / "figures" / "phase4" / "perturbation_size_comparison.png",
        "phase4_runtime": output / "figures" / "phase4" / "runtime_comparison.png",
        "phase5_clean_recall": output / "figures" / "phase5" / "clean_recall_comparison.png",
        "phase5_attack_recall": output / "figures" / "phase5" / "recall_under_attack_comparison.png",
        "phase5_attack_success": output / "figures" / "phase5" / "attack_success_comparison.png",
        "phase5_confusion": output / "figures" / "phase5" / "hardened_confusion_matrix.png",
        "phase7_results": output / "metrics" / "phase7_simulation_results.csv",
        "phase8_csv": output / "metrics" / "phase8_concept_drift.csv",
        "phase8_json": output / "metrics" / "phase8_concept_drift.json",
        "phase8_fraud_rate": output / "figures" / "phase8" / "fraud_rate_over_time.png",
        "phase8_recall": output / "figures" / "phase8" / "recall_over_time.png",
        "phase8_f1": output / "figures" / "phase8" / "f1_over_time.png",
        "phase8_probability": output / "figures" / "phase8" / "mean_fraud_probability_over_time.png",
        "phase8_feature_drift": output / "figures" / "phase8" / "feature_drift_over_time.png",
    }


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object, raising a useful artifact-specific error."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return payload


def load_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {source}")
    return pd.read_csv(source)


def load_prediction_artifacts(paths: Mapping[str, Path]) -> tuple[Any, Any, Any, list[str]]:
    """Load both immutable model variants and their shared preprocessing artifacts."""
    required = ("baseline_model", "hardened_model", "preprocessor", "feature_names")
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing prediction artifacts: " + ", ".join(missing))
    install_sklearn_joblib_compatibility()
    baseline = joblib.load(paths["baseline_model"])
    hardened = joblib.load(paths["hardened_model"])
    preprocessor = joblib.load(paths["preprocessor"])
    names = list(joblib.load(paths["feature_names"]))
    for label, model in (("baseline", baseline), ("hardened", hardened)):
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError(f"The {label} model does not implement predict_proba()")
    return baseline, hardened, preprocessor, names


def available_transaction_types(preprocessor: Any) -> list[str]:
    """Read fitted one-hot categories, with PaySim values as a safe UI fallback."""
    try:
        encoder = preprocessor.named_transformers_["transaction_type"]
        values = [str(value) for value in encoder.categories_[0]]
        return values or list(PAYMENT_TYPES)
    except (AttributeError, KeyError, IndexError, TypeError):
        return list(PAYMENT_TYPES)


def build_raw_transaction(values: Mapping[str, Any]) -> pd.DataFrame:
    """Validate and engineer exactly the raw features used in Phase 1."""
    missing = sorted(set(MODEL_INPUT_COLUMNS) - set(values))
    if missing:
        raise ValueError(f"Missing transaction fields: {missing}")
    row = {name: values[name] for name in MODEL_INPUT_COLUMNS}
    if int(row["step"]) < 1:
        raise ValueError("step must be at least 1")
    monetary = MODEL_INPUT_COLUMNS[2:]
    if any(float(row[name]) < 0 for name in monetary):
        raise ValueError("Transaction amounts and balances cannot be negative")
    frame = pd.DataFrame([row])
    frame["sender_balance_change"] = frame["oldbalanceOrg"] - frame["newbalanceOrig"]
    frame["receiver_balance_change"] = frame["newbalanceDest"] - frame["oldbalanceDest"]
    sender = frame["oldbalanceOrg"].to_numpy(dtype=np.float32)
    amount = frame["amount"].to_numpy(dtype=np.float32)
    frame["amount_to_sender_balance"] = np.divide(
        amount, sender, out=np.zeros_like(amount), where=sender != 0
    )
    return frame


def transform_transaction(
    values: Mapping[str, Any], preprocessor: Any, feature_names: list[str]
) -> pd.DataFrame:
    """Apply the saved fitted preprocessor and enforce saved column order."""
    raw = build_raw_transaction(values)
    transformed = preprocessor.transform(raw)
    encoded = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    missing = sorted(set(feature_names) - set(encoded.columns))
    extra = sorted(set(encoded.columns) - set(feature_names))
    if missing or extra:
        raise ValueError(f"Saved feature mismatch; missing={missing}, extra={extra}")
    return encoded.loc[:, feature_names].astype("float32")


def risk_label(probability: float) -> str:
    """Return a presentation-only probability band, not a model class."""
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


def predict_transaction(
    values: Mapping[str, Any], baseline: Any, hardened: Any,
    preprocessor: Any, feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Return actual predictions from both saved models without retraining."""
    encoded = transform_transaction(values, preprocessor, feature_names)
    results = {}
    for label, model in (("Baseline", baseline), ("Hardened", hardened)):
        probability = float(model.predict_proba(encoded)[0, 1])
        prediction = int(model.predict(encoded)[0])
        results[label] = {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": probability,
            "risk_label": risk_label(probability),
        }
    return results
