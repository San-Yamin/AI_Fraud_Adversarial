"""Memory-aware PaySim loading, validation, and descriptive inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CSV_CHUNK_SIZE,
    DATASET_PATH,
    DEVELOPMENT_FRAUD_SAMPLES,
    DEVELOPMENT_MODE_LABEL,
    DEVELOPMENT_NORMAL_SAMPLES,
    RANDOM_SEED,
    RUN_MODE,
    TARGET_COLUMN,
    FULL_MODE_LABEL,
)

REQUIRED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

PAYSIM_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def validate_required_columns(columns: Any) -> None:
    """Raise a clear error when the input does not match the PaySim schema."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"PaySim file is missing required columns: {missing}")


def _read_full_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=REQUIRED_COLUMNS, dtype=PAYSIM_DTYPES)


def _retain_smallest_random_keys(
    retained: pd.DataFrame | None, candidates: pd.DataFrame, limit: int
) -> pd.DataFrame:
    combined = candidates if retained is None else pd.concat([retained, candidates])
    return combined.nsmallest(limit, "_sample_key")


def _read_development_subset(
    path: Path,
    *,
    normal_samples: int,
    fraud_samples: int,
    random_seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Select representative rows across the whole file without loading it all.

    Each row receives a seeded random priority. Keeping the smallest priorities
    per class gives a reproducible class-aware sample and avoids the temporal
    bias caused by selecting the first N PaySim rows.
    """
    rng = np.random.default_rng(random_seed)
    retained_normal: pd.DataFrame | None = None
    retained_fraud: pd.DataFrame | None = None

    reader = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        dtype=PAYSIM_DTYPES,
        chunksize=chunk_size,
    )
    for chunk in reader:
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        retained_normal = _retain_smallest_random_keys(
            retained_normal, chunk.loc[chunk[TARGET_COLUMN] == 0], normal_samples
        )
        retained_fraud = _retain_smallest_random_keys(
            retained_fraud, chunk.loc[chunk[TARGET_COLUMN] == 1], fraud_samples
        )

    if retained_normal is None or retained_fraud is None or retained_fraud.empty:
        raise ValueError("Development sampling requires both legitimate and fraud rows")
    result = pd.concat([retained_normal, retained_fraud], ignore_index=True)
    result = result.drop(columns="_sample_key")
    return result.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def load_paysim(
    path: str | Path = DATASET_PATH,
    *,
    run_mode: str = RUN_MODE,
    normal_samples: int = DEVELOPMENT_NORMAL_SAMPLES,
    fraud_samples: int = DEVELOPMENT_FRAUD_SAMPLES,
    random_seed: int = RANDOM_SEED,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """Load PaySim in full mode or as a reproducible class-aware dev subset."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {dataset_path}")
    if run_mode not in {DEVELOPMENT_MODE_LABEL, FULL_MODE_LABEL}:
        raise ValueError(
            f"run_mode must be {DEVELOPMENT_MODE_LABEL!r} or {FULL_MODE_LABEL!r}"
        )

    header = pd.read_csv(dataset_path, nrows=0)
    validate_required_columns(header.columns)
    if run_mode == DEVELOPMENT_MODE_LABEL:
        data = _read_development_subset(
            dataset_path,
            normal_samples=normal_samples,
            fraud_samples=fraud_samples,
            random_seed=random_seed,
            chunk_size=chunk_size,
        )
    else:
        data = _read_full_dataset(dataset_path)
    validate_required_columns(data.columns)
    return data


def summarize_paysim(data: pd.DataFrame) -> dict[str, Any]:
    """Return the required dataset inspection summary without changing data."""
    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    return {
        "shape": tuple(data.shape),
        "columns": data.columns.tolist(),
        "missing_values": data.isna().sum().to_dict(),
        "transaction_types": data["type"].value_counts(dropna=False).to_dict(),
        "class_counts": {"legitimate": int(counts[0]), "fraud": int(counts[1])},
        "fraud_percentage": float(counts[1] / total * 100.0) if total else 0.0,
    }
