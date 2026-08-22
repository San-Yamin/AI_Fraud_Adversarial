"""Chunked PaySim step-window concept-drift analysis for the hardened model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import CSV_CHUNK_SIZE, RANDOM_SEED
from src.dashboard_utils import MODEL_INPUT_COLUMNS


DRIFT_INPUT_COLUMNS = (*MODEL_INPUT_COLUMNS, "isFraud")
DRIFT_DTYPES = {
    "step": "int16", "type": "category", "amount": "float32",
    "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
    "oldbalanceDest": "float32", "newbalanceDest": "float32", "isFraud": "int8",
}
DEFAULT_WINDOWS = 8
DEFAULT_DRIFT_FEATURES = 5
DEFAULT_FEATURE_SAMPLE = 2000


def build_step_window_edges(
    minimum_step: int, maximum_step: int, n_windows: int = DEFAULT_WINDOWS
) -> np.ndarray:
    """Create reproducible equal-width boundaries over observed PaySim steps."""
    if minimum_step > maximum_step:
        raise ValueError("minimum_step cannot exceed maximum_step")
    if n_windows < 2:
        raise ValueError("n_windows must be at least 2")
    return np.linspace(float(minimum_step), float(maximum_step + 1), n_windows + 1)


def assign_step_windows(steps: Any, edges: np.ndarray) -> np.ndarray:
    """Assign every integer step to exactly one chronological window."""
    values = np.asarray(steps, dtype=float)
    return np.searchsorted(edges[1:-1], values, side="right").astype(int)


def kolmogorov_smirnov_statistic(reference: Any, current: Any) -> float:
    """Compute the explainable two-sample KS distance without extra dependencies."""
    reference_values = np.sort(np.asarray(reference, dtype=float))
    current_values = np.sort(np.asarray(current, dtype=float))
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    if not len(reference_values) or not len(current_values):
        return float("nan")
    support = np.unique(np.concatenate([reference_values, current_values]))
    reference_cdf = np.searchsorted(reference_values, support, side="right") / len(reference_values)
    current_cdf = np.searchsorted(current_values, support, side="right") / len(current_values)
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def drift_level(score: float) -> str:
    """Apply transparent heuristic KS bands; these are not production thresholds."""
    if not np.isfinite(score):
        return "Unavailable"
    if score < 0.10:
        return "Low"
    if score < 0.20:
        return "Moderate"
    return "High"


def select_drift_features(
    feature_names: list[str], importance: pd.DataFrame | None, *, max_features: int = 5
) -> list[str]:
    """Select top saved SHAP features, excluding step (the window definition)."""
    available = set(feature_names) - {"step"}
    ordered: list[str] = []
    if importance is not None and "feature" in importance:
        ordered.extend(str(value) for value in importance["feature"])
    ordered.extend(feature_names)
    selected = []
    for feature in ordered:
        if feature in available and feature not in selected:
            selected.append(feature)
        if len(selected) == max_features:
            break
    if not selected:
        raise ValueError("No encoded features are available for drift measurement")
    return selected


def _transform_chunk(
    chunk: pd.DataFrame, preprocessor: Any, feature_names: list[str]
) -> pd.DataFrame:
    """Apply the saved preprocessor while keeping isFraud outside model input."""
    raw = chunk.loc[:, MODEL_INPUT_COLUMNS].copy()
    raw["sender_balance_change"] = raw["oldbalanceOrg"] - raw["newbalanceOrig"]
    raw["receiver_balance_change"] = raw["newbalanceDest"] - raw["oldbalanceDest"]
    sender = raw["oldbalanceOrg"].to_numpy(dtype=np.float32)
    amount = raw["amount"].to_numpy(dtype=np.float32)
    raw["amount_to_sender_balance"] = np.divide(
        amount, sender, out=np.zeros_like(amount), where=sender != 0
    )
    values = preprocessor.transform(raw)
    transformed = pd.DataFrame(values, columns=preprocessor.get_feature_names_out())
    if set(transformed.columns) != set(feature_names):
        raise ValueError("Saved preprocessor output does not match saved feature names")
    return transformed.loc[:, feature_names].astype("float32")


def _empty_accumulator(n_windows: int) -> list[dict[str, Any]]:
    return [
        {
            "count": 0, "fraud": 0, "tp": 0, "fp": 0, "fn": 0,
            "probability_sum": 0.0, "labels": [], "probabilities": [],
            "feature_sample": None, "step_min": None, "step_max": None,
        }
        for _ in range(n_windows)
    ]


def _update_accumulator(
    accumulator: dict[str, Any], chunk: pd.DataFrame, encoded: pd.DataFrame,
    model: Any, drift_features: list[str], sample_size: int, rng: np.random.Generator,
) -> None:
    labels = chunk["isFraud"].to_numpy(dtype=np.int8)
    probabilities = np.asarray(model.predict_proba(encoded), dtype=float)[:, 1]
    predictions = np.asarray(model.predict(encoded), dtype=np.int8)
    accumulator["count"] += len(chunk)
    accumulator["fraud"] += int(labels.sum())
    accumulator["tp"] += int(((labels == 1) & (predictions == 1)).sum())
    accumulator["fp"] += int(((labels == 0) & (predictions == 1)).sum())
    accumulator["fn"] += int(((labels == 1) & (predictions == 0)).sum())
    accumulator["probability_sum"] += float(probabilities.sum())
    accumulator["labels"].append(labels.copy())
    accumulator["probabilities"].append(probabilities.astype(np.float32))
    observed_min, observed_max = int(chunk["step"].min()), int(chunk["step"].max())
    accumulator["step_min"] = (
        observed_min if accumulator["step_min"] is None
        else min(accumulator["step_min"], observed_min)
    )
    accumulator["step_max"] = (
        observed_max if accumulator["step_max"] is None
        else max(accumulator["step_max"], observed_max)
    )
    candidates = encoded.loc[:, drift_features].copy()
    candidates["_sample_key"] = rng.random(len(candidates))
    retained = accumulator["feature_sample"]
    combined = candidates if retained is None else pd.concat([retained, candidates], ignore_index=True)
    accumulator["feature_sample"] = combined.nsmallest(sample_size, "_sample_key")


def _build_summary(
    accumulators: list[dict[str, Any]], drift_features: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    reference = accumulators[0]["feature_sample"]
    if reference is None:
        raise ValueError("The earliest chronological window contains no transactions")
    rows, feature_scores = [], {}
    for index, accumulator in enumerate(accumulators):
        count, fraud = accumulator["count"], accumulator["fraud"]
        if not count:
            raise ValueError(f"Chronological window {index + 1} is empty")
        precision_denominator = accumulator["tp"] + accumulator["fp"]
        precision = accumulator["tp"] / precision_denominator if precision_denominator else 0.0
        recall = accumulator["tp"] / fraud if fraud else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(recall) and precision + recall else float("nan")
        )
        labels = np.concatenate(accumulator["labels"])
        probabilities = np.concatenate(accumulator["probabilities"])
        pr_auc = (
            float(average_precision_score(labels, probabilities))
            if np.unique(labels).size == 2 else float("nan")
        )
        current = accumulator["feature_sample"]
        scores = {
            feature: kolmogorov_smirnov_statistic(reference[feature], current[feature])
            for feature in drift_features
        }
        drift_score = float(np.nanmean(list(scores.values())))
        feature_scores[f"Window {index + 1}"] = scores
        rows.append(
            {
                "Window": f"Window {index + 1}",
                "Step Range": f"{accumulator['step_min']}–{accumulator['step_max']}",
                "Transaction Count": int(count), "Fraud Count": int(fraud),
                "Fraud Rate": float(fraud / count), "Precision": float(precision),
                "Recall": float(recall), "F1": float(f1), "PR-AUC": pr_auc,
                "Mean Fraud Probability": float(accumulator["probability_sum"] / count),
                "Drift Score": drift_score, "Drift Level": drift_level(drift_score),
            }
        )
    return pd.DataFrame(rows), feature_scores


def analyze_concept_drift(
    dataset_path: str | Path, model: Any, preprocessor: Any, feature_names: list[str],
    *, importance: pd.DataFrame | None = None, n_windows: int = DEFAULT_WINDOWS,
    max_drift_features: int = DEFAULT_DRIFT_FEATURES,
    feature_sample_size: int = DEFAULT_FEATURE_SAMPLE,
    random_state: int = RANDOM_SEED, chunk_size: int = CSV_CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the unchanged hardened model over chunked chronological windows."""
    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {source}")
    header = pd.read_csv(source, nrows=0)
    missing = sorted(set(DRIFT_INPUT_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"PaySim file is missing drift columns: {missing}")
    minimum_step, maximum_step = None, None
    for steps in pd.read_csv(source, usecols=["step"], dtype={"step": "int16"}, chunksize=chunk_size):
        chunk_min, chunk_max = int(steps["step"].min()), int(steps["step"].max())
        minimum_step = chunk_min if minimum_step is None else min(minimum_step, chunk_min)
        maximum_step = chunk_max if maximum_step is None else max(maximum_step, chunk_max)
    edges = build_step_window_edges(minimum_step, maximum_step, n_windows)
    drift_features = select_drift_features(
        feature_names, importance, max_features=max_drift_features
    )
    accumulators = _empty_accumulator(n_windows)
    rngs = [np.random.default_rng(random_state + index) for index in range(n_windows)]
    for chunk in pd.read_csv(
        source, usecols=DRIFT_INPUT_COLUMNS, dtype=DRIFT_DTYPES, chunksize=chunk_size,
    ):
        assignments = assign_step_windows(chunk["step"], edges)
        for window_index in np.unique(assignments):
            mask = assignments == window_index
            window_chunk = chunk.loc[mask].reset_index(drop=True)
            encoded = _transform_chunk(window_chunk, preprocessor, feature_names)
            _update_accumulator(
                accumulators[int(window_index)], window_chunk, encoded, model,
                drift_features, feature_sample_size, rngs[int(window_index)],
            )
    summary, feature_scores = _build_summary(accumulators, drift_features)
    metadata = {
        "method": "fixed equal-width step ranges over observed minimum/maximum step",
        "n_windows": int(n_windows), "minimum_step": int(minimum_step),
        "maximum_step": int(maximum_step), "window_edges": edges.tolist(),
        "drift_metric": "mean two-sample Kolmogorov–Smirnov statistic",
        "reference_window": "Window 1", "drift_features": drift_features,
        "feature_sample_size_per_window": int(feature_sample_size),
        "random_seed": int(random_state), "model_retrained": False,
        "target_used_as_feature": False, "feature_drift_scores": feature_scores,
    }
    return summary, metadata


def interpret_drift(summary: pd.DataFrame) -> dict[str, Any]:
    """Create cautious, data-derived experimental interpretation."""
    maximum_score = float(summary["Drift Score"].max())
    potential = summary.loc[summary["Drift Level"].isin(["Moderate", "High"]), "Window"].tolist()
    valid_recall = summary.loc[summary["Recall"].notna(), "Recall"]
    degraded = False
    recall_change = float("nan")
    if len(valid_recall) >= 4:
        early = float(valid_recall.iloc[:2].mean())
        late = float(valid_recall.iloc[-2:].mean())
        recall_change = late - early
        degraded = recall_change < -0.05
    return {
        "maximum_drift_score": maximum_score,
        "maximum_drift_level": drift_level(maximum_score),
        "potential_drift_windows": potential,
        "meaningful_feature_drift_observed": bool(maximum_score >= 0.10),
        "recall_change_late_vs_early": recall_change,
        "performance_degradation_observed": bool(degraded),
        "threshold_note": (
            "KS bands are descriptive heuristics: low <0.10, moderate 0.10–<0.20, "
            "high >=0.20. Recall degradation uses a >0.05 late-vs-early decrease."
        ),
        "scope_note": "PaySim-based simulated temporal analysis, not production drift evidence.",
    }


def save_drift_outputs(
    summary: pd.DataFrame, metadata: dict[str, Any], interpretation: dict[str, Any],
    csv_path: str | Path, json_path: str | Path,
) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    csv_output, json_output = Path(csv_path), Path(json_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_output, index=False)
    json_output.write_text(
        json.dumps(json_safe(
            {"summary": summary.to_dict(orient="records"), "metadata": metadata,
             "interpretation": interpretation}
        ), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def plot_drift_summary(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save the five requested presentation-friendly chronological figures."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("Fraud Rate", "Fraud Rate Over PaySim Step Windows", "Fraud rate", "fraud_rate_over_time.png"),
        ("Recall", "Fraud Recall Over PaySim Step Windows", "Recall", "recall_over_time.png"),
        ("F1", "F1 Over PaySim Step Windows", "F1", "f1_over_time.png"),
        ("Mean Fraud Probability", "Mean Fraud Probability Over Time", "Mean probability", "mean_fraud_probability_over_time.png"),
        ("Drift Score", "Feature Drift vs Early Reference Window", "Mean KS statistic", "feature_drift_over_time.png"),
    )
    paths = []
    for column, title, ylabel, filename in specifications:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary["Window"], summary[column], marker="o")
        if column == "Drift Score":
            ax.axhline(0.10, color="orange", linestyle="--", label="Moderate threshold")
            ax.axhline(0.20, color="red", linestyle="--", label="High threshold")
            ax.legend()
        ax.set(title=title, xlabel="Chronological window", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = directory / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
