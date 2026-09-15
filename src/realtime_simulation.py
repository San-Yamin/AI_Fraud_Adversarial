"""Deterministic, target-safe PaySim transaction-stream simulation for Phase 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CSV_CHUNK_SIZE, RANDOM_SEED
from src.dashboard_utils import MODEL_INPUT_COLUMNS, transform_transaction


SIMULATION_COLUMNS = (*MODEL_INPUT_COLUMNS, "isFraud")

NUMERIC_SIMULATION_COLUMNS = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
)

RESULT_COLUMNS = (
    "transaction_sequence",
    "source_index",
    "step",
    "type",
    "amount",
    "fraud_probability",
    "prediction",
    "predicted_label",
    "actual_label",
    "actual_label_name",
    "correct",
    "status",
)


def _class_counts(
    sample_size: int,
    fraud_fraction: float,
) -> tuple[int, int]:
    """Return requested legitimate/fraud sample counts."""

    if sample_size < 2:
        raise ValueError(
            "sample_size must be at least 2 to include both classes"
        )

    if not 0 < fraud_fraction < 1:
        raise ValueError(
            "fraud_fraction must be in (0, 1)"
        )

    fraud = min(
        sample_size - 1,
        max(1, round(sample_size * fraud_fraction)),
    )

    legitimate = sample_size - fraud

    return legitimate, fraud


def _clean_simulation_rows(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only model-required PaySim columns and safely convert numeric fields.

    nameOrig and nameDest are intentionally excluded because they contain
    identifiers such as C661060758 and are not model features.
    """

    missing = sorted(
        set(SIMULATION_COLUMNS) - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Simulation source is missing columns: {missing}"
        )

    cleaned = data.loc[:, SIMULATION_COLUMNS].copy()

    # Transaction type remains categorical/text.
    cleaned["type"] = (
        cleaned["type"]
        .astype("string")
        .str.strip()
    )

    # Convert only genuine numeric PaySim fields.
    for column in NUMERIC_SIMULATION_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Remove malformed rows.
    cleaned = cleaned.dropna(
        subset=list(SIMULATION_COLUMNS)
    ).copy()

    if cleaned.empty:
        raise ValueError(
            "No valid PaySim rows remain after numeric conversion. "
            "Check that the CSV uses the standard PaySim columns: "
            "step, type, amount, nameOrig, oldbalanceOrg, "
            "newbalanceOrig, nameDest, oldbalanceDest, "
            "newbalanceDest, isFraud, isFlaggedFraud."
        )

    cleaned["step"] = cleaned["step"].astype("int32")

    cleaned["amount"] = cleaned["amount"].astype(
        "float32"
    )

    cleaned["oldbalanceOrg"] = cleaned[
        "oldbalanceOrg"
    ].astype("float32")

    cleaned["newbalanceOrig"] = cleaned[
        "newbalanceOrig"
    ].astype("float32")

    cleaned["oldbalanceDest"] = cleaned[
        "oldbalanceDest"
    ].astype("float32")

    cleaned["newbalanceDest"] = cleaned[
        "newbalanceDest"
    ].astype("float32")

    cleaned["isFraud"] = cleaned["isFraud"].astype(
        "int8"
    )

    # Ensure only valid binary labels remain.
    cleaned = cleaned.loc[
        cleaned["isFraud"].isin([0, 1])
    ].copy()

    return cleaned


def create_simulation_sequence(
    data: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
    fraud_fraction: float = 0.20,
) -> pd.DataFrame:
    """
    Create a deterministic class-aware PaySim simulation sequence.
    """

    cleaned = _clean_simulation_rows(data)

    legitimate_count, fraud_count = _class_counts(
        sample_size,
        fraud_fraction,
    )

    legitimate = cleaned.loc[
        cleaned["isFraud"] == 0
    ]

    fraud = cleaned.loc[
        cleaned["isFraud"] == 1
    ]

    if (
        len(legitimate) < legitimate_count
        or len(fraud) < fraud_count
    ):
        raise ValueError(
            "Simulation source does not contain enough "
            "valid rows from both classes"
        )

    selected = pd.concat(
        [
            legitimate.sample(
                n=legitimate_count,
                random_state=random_state,
            ),
            fraud.sample(
                n=fraud_count,
                random_state=random_state + 1,
            ),
        ]
    )

    selected = selected.sample(
        frac=1.0,
        random_state=random_state + 2,
    ).copy()

    selected.insert(
        0,
        "source_index",
        selected.index.to_numpy(),
    )

    return (
        selected
        .reset_index(drop=True)
        .loc[:, ("source_index", *SIMULATION_COLUMNS)]
    )


def load_paysim_simulation_sequence(
    path: str | Path,
    *,
    sample_size: int,
    random_state: int = RANDOM_SEED,
    fraud_fraction: float = 0.20,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Read PaySim in chunks and retain a small deterministic sample.

    nameOrig/nameDest are never loaded into the simulation model features.
    """

    source = Path(path).expanduser()

    if not source.is_file():
        raise FileNotFoundError(
            f"PaySim CSV not found: {source}"
        )

    # Read header only.
    header = pd.read_csv(
        source,
        nrows=0,
    )

    missing = sorted(
        set(SIMULATION_COLUMNS)
        - set(header.columns)
    )

    if missing:
        raise ValueError(
            f"PaySim file is missing simulation columns: {missing}"
        )

    legitimate_count, fraud_count = _class_counts(
        sample_size,
        fraud_fraction,
    )

    rng = np.random.default_rng(
        random_state
    )

    retained: dict[
        int,
        pd.DataFrame | None,
    ] = {
        0: None,
        1: None,
    }

    limits = {
        0: legitimate_count,
        1: fraud_count,
    }

    source_row_offset = 0

    # Important:
    # read ONLY these columns.
    # nameOrig/nameDest are not loaded.
    for raw_chunk in pd.read_csv(
        source,
        usecols=list(SIMULATION_COLUMNS),
        chunksize=chunk_size,
        low_memory=False,
    ):

        original_chunk_length = len(
            raw_chunk
        )

        if original_chunk_length == 0:
            continue

        raw_chunk = raw_chunk.copy()

        # Track original dataset row position.
        raw_chunk["source_index"] = np.arange(
            source_row_offset,
            source_row_offset
            + original_chunk_length,
            dtype=np.int64,
        )

        source_row_offset += (
            original_chunk_length
        )

        source_index = raw_chunk[
            "source_index"
        ].copy()

        # Clean and type-check fields.
        cleaned = _clean_simulation_rows(
            raw_chunk
        )

        cleaned["source_index"] = (
            source_index
            .loc[cleaned.index]
            .to_numpy()
        )

        cleaned["_sample_key"] = (
            rng.random(len(cleaned))
        )

        for label in (0, 1):

            candidates = cleaned.loc[
                cleaned["isFraud"]
                == label
            ]

            if candidates.empty:
                continue

            if retained[label] is None:
                combined = candidates
            else:
                combined = pd.concat(
                    [
                        retained[label],
                        candidates,
                    ],
                    ignore_index=True,
                )

            retained[label] = (
                combined.nsmallest(
                    limits[label],
                    "_sample_key",
                )
            )

    if any(
        retained[label] is None
        or len(retained[label])
        < limits[label]
        for label in (0, 1)
    ):
        raise ValueError(
            "PaySim does not contain enough valid "
            "legitimate and fraud rows after cleaning"
        )

    sequence = pd.concat(
        [
            retained[0],
            retained[1],
        ],
        ignore_index=True,
    )

    sequence = sequence.sample(
        frac=1.0,
        random_state=random_state + 2,
    )

    sequence = (
        sequence
        .drop(columns="_sample_key")
        .reset_index(drop=True)
    )

    return sequence.loc[
        :,
        (
            "source_index",
            *SIMULATION_COLUMNS,
        ),
    ]


def preprocess_simulation_rows(
    rows: pd.DataFrame,
    preprocessor: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Convert simulation rows to saved model features.

    PaySim account IDs and isFraud are never passed into preprocessing.
    """

    encoded_rows: list[pd.Series] = []

    for _, row in rows.iterrows():

        # Explicit typing prevents identifiers such as
        # C661060758 from becoming numeric model inputs.
        values = {
            "step": int(
                row["step"]
            ),

            "type": str(
                row["type"]
            ),

            "amount": float(
                row["amount"]
            ),

            "oldbalanceOrg": float(
                row["oldbalanceOrg"]
            ),

            "newbalanceOrig": float(
                row["newbalanceOrig"]
            ),

            "oldbalanceDest": float(
                row["oldbalanceDest"]
            ),

            "newbalanceDest": float(
                row["newbalanceDest"]
            ),
        }

        transformed = transform_transaction(
            values,
            preprocessor,
            feature_names,
        )

        encoded_rows.append(
            transformed.iloc[0]
        )

    if not encoded_rows:
        return pd.DataFrame(
            columns=feature_names,
            dtype="float32",
        )

    encoded = (
        pd.DataFrame(encoded_rows)
        .reset_index(drop=True)
    )

    return (
        encoded
        .loc[:, feature_names]
        .astype("float32")
    )


def classify_outcome(
    actual: int,
    prediction: int,
) -> str:
    """Return readable prediction outcome."""

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
    """
    Generate predictions for one simulation batch.
    """

    if "isFraud" not in rows:
        raise ValueError(
            "Simulation rows require isFraud "
            "for evaluation"
        )

    encoded = preprocess_simulation_rows(
        rows,
        preprocessor,
        feature_names,
    )

    probabilities = np.asarray(
        model.predict_proba(encoded)
    )[:, 1]

    predictions = np.asarray(
        model.predict(encoded)
    ).astype(int)

    actual = rows[
        "isFraud"
    ].to_numpy(
        dtype=int
    )

    records = pd.DataFrame(
        {
            "transaction_sequence":
                np.arange(
                    sequence_start,
                    sequence_start
                    + len(rows),
                ),

            "source_index":
                rows[
                    "source_index"
                ].to_numpy(),

            "step":
                rows[
                    "step"
                ].to_numpy(),

            "type":
                rows[
                    "type"
                ].astype(
                    str
                ).to_numpy(),

            "amount":
                rows[
                    "amount"
                ].to_numpy(
                    dtype=float
                ),

            "fraud_probability":
                probabilities.astype(
                    float
                ),

            "prediction":
                predictions,

            "predicted_label":
                np.where(
                    predictions == 1,
                    "Fraud",
                    "Legitimate",
                ),

            "actual_label":
                actual,

            "actual_label_name":
                np.where(
                    actual == 1,
                    "Fraud",
                    "Legitimate",
                ),

            "correct":
                predictions == actual,

            "status":
                [
                    classify_outcome(
                        a,
                        p,
                    )
                    for a, p in zip(
                        actual,
                        predictions,
                    )
                ],
        }
    )

    return records.loc[
        :,
        RESULT_COLUMNS,
    ]


def calculate_running_metrics(
    results: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Calculate cumulative fraud-detection metrics.
    """

    if results.empty:
        return {
            "total_processed": 0,
            "actual_fraud": 0,
            "detected_fraud": 0,
            "missed_fraud": 0,
            "false_positives": 0,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    actual = results[
        "actual_label"
    ].to_numpy(
        dtype=int
    )

    predicted = results[
        "prediction"
    ].to_numpy(
        dtype=int
    )

    true_positives = int(
        (
            (actual == 1)
            & (predicted == 1)
        ).sum()
    )

    missed = int(
        (
            (actual == 1)
            & (predicted == 0)
        ).sum()
    )

    false_positives = int(
        (
            (actual == 0)
            & (predicted == 1)
        ).sum()
    )

    actual_fraud = int(
        (actual == 1).sum()
    )

    predicted_fraud = int(
        (predicted == 1).sum()
    )

    precision = (
        true_positives
        / predicted_fraud
        if predicted_fraud
        else 0.0
    )

    recall = (
        true_positives
        / actual_fraud
        if actual_fraud
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "total_processed":
            int(len(results)),

        "actual_fraud":
            actual_fraud,

        "detected_fraud":
            true_positives,

        "missed_fraud":
            missed,

        "false_positives":
            false_positives,

        "true_positives":
            true_positives,

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),
    }


def add_running_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach cumulative metrics used by live charts.
    """

    enriched = results.copy()

    actual_fraud = (
        enriched[
            "actual_label"
        ]
        .eq(1)
        .cumsum()
    )

    true_positive = (
        (
            enriched[
                "actual_label"
            ].eq(1)
        )
        & (
            enriched[
                "prediction"
            ].eq(1)
        )
    ).cumsum()

    enriched[
        "running_recall"
    ] = np.divide(
        true_positive,
        actual_fraud,
        out=np.zeros(
            len(enriched),
            dtype=float,
        ),
        where=(
            actual_fraud
            .to_numpy()
            != 0
        ),
    )

    enriched[
        "running_predicted_fraud"
    ] = (
        enriched[
            "prediction"
        ]
        .eq(1)
        .cumsum()
    )

    enriched[
        "running_predicted_legitimate"
    ] = (
        enriched[
            "prediction"
        ]
        .eq(0)
        .cumsum()
    )

    return enriched


def reset_simulation_state(
    sequence: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create a fresh Real-Time Simulation state.
    """

    return {
        "sequence":
            sequence.copy(
                deep=True
            ),

        "position":
            0,

        "running":
            False,

        "results":
            pd.DataFrame(
                columns=RESULT_COLUMNS
            ),

        "next_due":
            0.0,
    }


def save_simulation_results(
    results: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Save simulation results to CSV.
    """

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.loc[
        :,
        RESULT_COLUMNS,
    ].to_csv(
        output,
        index=False,
    )

    return output