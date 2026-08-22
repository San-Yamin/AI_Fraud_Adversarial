"""Leakage-safe Phase 1 cleaning, feature engineering, and encoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import LEAKAGE_COLUMN, RANDOM_SEED, SMOTE_SAMPLING_STRATEGY, TARGET_COLUMN

IDENTIFIER_COLUMNS = ("nameOrig", "nameDest")
RAW_FEATURE_COLUMNS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
ENGINEERED_FEATURE_COLUMNS = (
    "sender_balance_change",
    "receiver_balance_change",
    "amount_to_sender_balance",
)
NUMERIC_FEATURE_COLUMNS = tuple(
    column for column in RAW_FEATURE_COLUMNS if column != "type"
) + ENGINEERED_FEATURE_COLUMNS
CATEGORICAL_FEATURE_COLUMNS = ("type",)


def prepare_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target, remove leakage/IDs, and engineer safe features.

    ``sender_balance_change`` and ``receiver_balance_change`` express the
    observed before/after balance movement. ``amount_to_sender_balance``
    normalizes transaction size and uses zero when the original balance is zero.
    """
    required = set(RAW_FEATURE_COLUMNS) | {TARGET_COLUMN, LEAKAGE_COLUMN} | set(
        IDENTIFIER_COLUMNS
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Cannot preprocess data; missing columns: {missing}")

    y = data[TARGET_COLUMN].astype("int8").copy()
    X = data.loc[:, RAW_FEATURE_COLUMNS].copy()
    X["sender_balance_change"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["receiver_balance_change"] = X["newbalanceDest"] - X["oldbalanceDest"]
    sender_balance = X["oldbalanceOrg"].to_numpy(dtype=np.float32)
    amount = X["amount"].to_numpy(dtype=np.float32)
    X["amount_to_sender_balance"] = np.divide(
        amount,
        sender_balance,
        out=np.zeros_like(amount, dtype=np.float32),
        where=sender_balance != 0,
    )
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create a train-fitted transformer with readable one-hot names."""
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURE_COLUMNS)),
            (
                "transaction_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on training data and only transform untouched test data."""
    preprocessor = build_preprocessor()
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out().tolist()
    transformed_train = pd.DataFrame(train_values, columns=names, index=X_train.index)
    transformed_test = pd.DataFrame(test_values, columns=names, index=X_test.index)
    return transformed_train.astype("float32"), transformed_test.astype("float32"), preprocessor


def resample_training_data(
    X_train: Any,
    y_train: Any,
    *,
    random_state: int = RANDOM_SEED,
    sampling_strategy: float | str | dict[Any, Any] = SMOTE_SAMPLING_STRATEGY,
    **smote_kwargs: Any,
) -> tuple[Any, Any]:
    """Apply SMOTE exclusively to already-split training data.

    Validation and test arrays are deliberately absent from this API, making
    accidental resampling of held-out data harder.
    """
    sampler = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        **smote_kwargs,
    )
    return sampler.fit_resample(X_train, y_train)
