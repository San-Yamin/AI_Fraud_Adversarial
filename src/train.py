"""Reproducible splitting and baseline XGBoost training for Phase 1."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TEST_SIZE

if TYPE_CHECKING:
    from xgboost import XGBClassifier


def stratified_train_test_split(
    X: Any,
    y: Any,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[Any, Any, Any, Any]:
    """Split before any fitting or SMOTE, preserving fraud prevalence."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_baseline_model(random_state: int = RANDOM_SEED) -> "XGBClassifier":
    """Return a modest Colab-compatible baseline without test-set tuning."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def train_baseline_model(X_train: Any, y_train: Any) -> "XGBClassifier":
    """Fit and return a new baseline model."""
    model = build_baseline_model()
    model.fit(X_train, y_train)
    return model
