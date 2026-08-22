"""Dataset-free checks for deterministic Phase 2 explanation utilities."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.explainability import (
    SHAP_MAX_FRAUD_SAMPLES,
    SHAP_MAX_SAMPLES,
    build_interpretation,
    compute_shap_values,
    create_tree_explainer,
    prepare_shap_evaluation_sample,
    rank_global_importance,
    select_correct_examples,
)


class FixedModel:
    def predict(self, X):
        return np.array([0, 1, 0, 1, 1, 0])[: len(X)]

    def predict_proba(self, X):
        predictions = self.predict(X)
        fraud = np.where(predictions == 1, 0.9, 0.1)
        return np.column_stack([1 - fraud, fraud])


class FakeExplanation:
    feature_names = ["amount", "type_TRANSFER"]
    values = np.array([[2.0, -1.0], [4.0, 1.0]])


def test_shap_sampling_config_names_are_aligned():
    assert config.SHAP_MAX_SAMPLES == SHAP_MAX_SAMPLES
    assert config.SHAP_MAX_FRAUD_SAMPLES == SHAP_MAX_FRAUD_SAMPLES
    assert config.SHAP_MAX_SAMPLES > 0
    assert 0 <= config.SHAP_MAX_FRAUD_SAMPLES <= config.SHAP_MAX_SAMPLES
    assert config.SHAP_MAX_DISPLAY > 0


def test_shap_sample_is_bounded_reproducible_and_test_only():
    X = pd.DataFrame({"a": range(20)}, index=range(100, 120))
    y = pd.Series([0] * 15 + [1] * 5, index=X.index)
    first_X, first_y = prepare_shap_evaluation_sample(
        X, y, max_samples=8, max_fraud_samples=3, random_state=42
    )
    second_X, second_y = prepare_shap_evaluation_sample(
        X, y, max_samples=8, max_fraud_samples=3, random_state=42
    )
    assert first_X.index.tolist() == second_X.index.tolist()
    assert first_y.index.tolist() == second_y.index.tolist()
    assert len(first_X) == 8
    assert first_y.value_counts().to_dict() == {0: 5, 1: 3}
    assert set(first_X.index).issubset(X.index)


def test_correct_examples_are_selected_automatically():
    X = pd.DataFrame({"a": range(6)})
    y = pd.Series([0, 1, 1, 1, 0, 0])
    selected = select_correct_examples(FixedModel(), X, y, random_state=42)
    assert selected["fraud"]["actual_label"] == 1
    assert selected["fraud"]["predicted_label"] == 1
    assert selected["legitimate"]["actual_label"] == 0
    assert selected["legitimate"]["predicted_label"] == 0


def test_rank_and_interpretation_come_from_shap_values():
    ranked = rank_global_importance(FakeExplanation())
    assert ranked.iloc[0]["feature"] == "amount"
    table = pd.DataFrame(
        {
            "feature": ["amount", "type_TRANSFER"],
            "feature_value": [100.0, 1.0],
            "shap_value": [2.0, -1.0],
            "direction": ["toward fraud", "toward legitimate"],
        }
    )
    interpretation = build_interpretation(
        {"actual_label": 1, "predicted_label": 1, "fraud_probability": 0.9}, table
    )
    assert interpretation["top_features_toward_fraud"][0]["feature"] == "amount"
    assert (
        interpretation["top_features_toward_legitimate"][0]["feature"]
        == "type_TRANSFER"
    )


def test_tree_explainer_returns_fraud_class_feature_matrix():
    X = pd.DataFrame(
        {"amount": [1.0, 2.0, 8.0, 9.0], "type_TRANSFER": [0, 0, 1, 1]}
    )
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    explanation = compute_shap_values(create_tree_explainer(model), X)
    assert explanation.values.shape == X.shape
    assert explanation.feature_names == X.columns.tolist()
