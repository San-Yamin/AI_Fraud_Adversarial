"""Deterministic, target-safe PaySim transaction-stream simulation for Phase 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CSV_CHUNK_SIZE, RANDOM_SEED
from src.dashboard_utils import MODEL_INPUT_COLUMNS, transform_transaction


SIMULATION_COLUMNS = (*MODEL_INPUT_COLUMNS, "isFraud")
SIMULATION_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
}
RESULT_COLUMNS = (
    "transaction_sequence", "source_index", "step", "type", "amount",
    "fraud_probability", "prediction", "predicted_label", "actual_label",
    "actual_label_name", "correct", "status",
)


def _class_counts(sample_size: int, fraud_fraction: float) -> tuple[int, int]:
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2 to include both classes")
    if not 0 < fraud_fraction < 1:
        raise ValueError("fraud_fraction must be in (0, 1)")
    fraud = min(sample_size - 1, max(1, round(sample_size * fraud_fraction)))
    return sample_size - fraud, fraud


def create_simulation_sequence(
    data: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
    fraud_fraction: float = 0.20,
) -> pd.DataFrame:
    """Create a seeded class-aware sequence from already bounded PaySim rows."""
    missing = sorted(set(SIMULATION_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Simulation source is missing columns: {missing}")
    legitimate_count, fraud_count = _class_counts(sample_size, fraud_fraction)
    legitimate = data.loc[data["isFraud"] == 0]
    fraud = data.loc[data["isFraud"] == 1]
    if len(legitimate) < legitimate_count or len(fraud) < fraud_count:
        raise ValueError("Simulation source does not contain enough rows from both classes")
    selected = pd.concat(
        [
            legitimate.sample(n=legitimate_count, random_state=random_state),
            fraud.sample(n=fraud_count, random_state=random_state + 1),
        ]
    )
    selected = selected.sample(frac=1.0, random_state=random_state + 2).copy()
    selected.insert(0, "source_index", selected.index.to_numpy())
    return selected.reset_index(drop=True).loc[:, ("source_index", *SIMULATION_COLUMNS)]


def load_paysim_simulation_sequence(
    path: str | Path,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
    fraud_fraction: float = 0.20,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """Scan PaySim in chunks and retain only a small deterministic class sample."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {source}")
    header = pd.read_csv(source, nrows=0)
    missing = sorted(set(SIMULATION_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"PaySim file is missing simulation columns: {missing}")
    legitimate_count, fraud_count = _class_counts(sample_size, fraud_fraction)
    rng = np.random.default_rng(random_state)
    retained = {0: None, 1: None}
    limits = {0: legitimate_count, 1: fraud_count}
    row_offset = 0
    for chunk in pd.read_csv(
        source, usecols=SIMULATION_COLUMNS, dtype=SIMULATION_DTYPES,
        chunksize=chunk_size,
    ):
        chunk = chunk.copy()
        chunk["source_index"] = np.arange(row_offset, row_offset + len(chunk))
        chunk["_sample_key"] = rng.random(len(chunk))
        row_offset += len(chunk)
        for label in (0, 1):
            candidates = chunk.loc[chunk["isFraud"] == label]
            if candidates.empty:
                continue
            combined = (
                candidates if retained[label] is None
                else pd.concat([retained[label], candidates], ignore_index=True)
            )
            retained[label] = combined.nsmallest(limits[label], "_sample_key")
    if any(retained[label] is None or len(retained[label]) < limits[label] for label in (0, 1)):
        raise ValueError("PaySim does not contain enough legitimate and fraud rows")
    sequence = pd.concat([retained[0], retained[1]], ignore_index=True)
    sequence = sequence.sample(frac=1.0, random_state=random_state + 2)
    return sequence.drop(columns="_sample_key").reset_index(drop=True).loc[
        :, ("source_index", *SIMULATION_COLUMNS)
    ]


def preprocess_simulation_rows(
    rows: pd.DataFrame, preprocessor: Any, feature_names: list[str]
) -> pd.DataFrame:
    """Transform stream rows one-by-one while never passing the target to preprocessing."""
    encoded = []
    for _, row in rows.iterrows():
        values = {column: row[column] for column in MODEL_INPUT_COLUMNS}
        encoded.append(transform_transaction(values, preprocessor, feature_names).iloc[0])
    if not encoded:
        return pd.DataFrame(columns=feature_names, dtype="float32")
    return pd.DataFrame(encoded).reset_index(drop=True).loc[:, feature_names].astype("float32")


def classify_outcome(actual: int, prediction: int) -> str:
    if actual == 1 and prediction == 1:
        return "Correct Detection"
    if actual == 1 and prediction == 0:
        return "Missed Fraud"
    if actual == 0 and prediction == 1:
        return "False Positive"
    return "Correct Legitimate"


def predict_simulation_batch(
    rows: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    feature_names: list[str],
    *,
    sequence_start: int = 1,
) -> pd.DataFrame:
    """Generate actual hardened-model predictions for the next stream batch."""
    if "isFraud" not in rows:
        raise ValueError("Simulation rows require isFraud for evaluation")
    encoded = preprocess_simulation_rows(rows, preprocessor, feature_names)
    probabilities = np.asarray(model.predict_proba(encoded))[:, 1]
    predictions = np.asarray(model.predict(encoded)).astype(int)
    actual = rows["isFraud"].to_numpy(dtype=int)
    records = pd.DataFrame(
        {
            "transaction_sequence": np.arange(sequence_start, sequence_start + len(rows)),
            "source_index": rows["source_index"].to_numpy(),
            "step": rows["step"].to_numpy(),
            "type": rows["type"].astype(str).to_numpy(),
            "amount": rows["amount"].to_numpy(dtype=float),
            "fraud_probability": probabilities.astype(float),
            "prediction": predictions,
            "predicted_label": np.where(predictions == 1, "Fraud", "Legitimate"),
            "actual_label": actual,
            "actual_label_name": np.where(actual == 1, "Fraud", "Legitimate"),
            "correct": predictions == actual,
            "status": [classify_outcome(a, p) for a, p in zip(actual, predictions)],
        }
    )
    return records.loc[:, RESULT_COLUMNS]


def calculate_running_metrics(results: pd.DataFrame) -> dict[str, float | int]:
    """Calculate cumulative fraud-detection counters with safe zero division."""
    if results.empty:
        return {
            "total_processed": 0, "actual_fraud": 0, "detected_fraud": 0,
            "missed_fraud": 0, "false_positives": 0, "true_positives": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
    actual = results["actual_label"].to_numpy(dtype=int)
    predicted = results["prediction"].to_numpy(dtype=int)
    true_positives = int(((actual == 1) & (predicted == 1)).sum())
    missed = int(((actual == 1) & (predicted == 0)).sum())
    false_positives = int(((actual == 0) & (predicted == 1)).sum())
    actual_fraud = int((actual == 1).sum())
    predicted_fraud = int((predicted == 1).sum())
    precision = true_positives / predicted_fraud if predicted_fraud else 0.0
    recall = true_positives / actual_fraud if actual_fraud else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_processed": int(len(results)), "actual_fraud": actual_fraud,
        "detected_fraud": true_positives, "missed_fraud": missed,
        "false_positives": false_positives, "true_positives": true_positives,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def add_running_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Attach running recall and prediction counts for live plotting."""
    enriched = results.copy()
    actual_fraud = enriched["actual_label"].eq(1).cumsum()
    true_positive = (
        enriched["actual_label"].eq(1) & enriched["prediction"].eq(1)
    ).cumsum()
    enriched["running_recall"] = np.divide(
        true_positive, actual_fraud,
        out=np.zeros(len(enriched), dtype=float), where=actual_fraud.to_numpy() != 0,
    )
    enriched["running_predicted_fraud"] = enriched["prediction"].eq(1).cumsum()
    enriched["running_predicted_legitimate"] = enriched["prediction"].eq(0).cumsum()
    return enriched


def reset_simulation_state(sequence: pd.DataFrame) -> dict[str, Any]:
    """Return a fresh deterministic state without modifying the source sequence."""
    return {
        "sequence": sequence.copy(deep=True), "position": 0, "running": False,
        "results": pd.DataFrame(columns=RESULT_COLUMNS), "next_due": 0.0,
    }


def save_simulation_results(results: pd.DataFrame, path: str | Path) -> Path:
    """Export computed results only; no source data or model artifacts are modified."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.loc[:, RESULT_COLUMNS].to_csv(output, index=False)
    return output
