"""Dataset-free tests for Phase 6 artifact paths and prediction preprocessing."""

import numpy as np

from src.dashboard_utils import (
    artifact_paths,
    build_raw_transaction,
    predict_transaction,
    risk_label,
    transform_transaction,
)


VALUES = {
    "step": 1, "type": "TRANSFER", "amount": 100.0,
    "oldbalanceOrg": 500.0, "newbalanceOrig": 400.0,
    "oldbalanceDest": 20.0, "newbalanceDest": 120.0,
}


class IdentityPreprocessor:
    def transform(self, frame):
        return frame[["amount", "sender_balance_change", "receiver_balance_change"]].to_numpy()

    def get_feature_names_out(self):
        return np.array(["amount", "sender_balance_change", "receiver_balance_change"])


class ProbabilityModel:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return np.array([[1 - self.probability, self.probability]])

    def predict(self, frame):
        return np.array([int(self.probability >= 0.5)])


def test_raw_transaction_recomputes_phase1_engineered_features():
    frame = build_raw_transaction(VALUES)
    assert frame.loc[0, "sender_balance_change"] == 100.0
    assert frame.loc[0, "receiver_balance_change"] == 100.0
    assert frame.loc[0, "amount_to_sender_balance"] == 0.2


def test_saved_feature_order_and_dual_model_predictions_are_used():
    names = ["receiver_balance_change", "amount", "sender_balance_change"]
    encoded = transform_transaction(VALUES, IdentityPreprocessor(), names)
    assert encoded.columns.tolist() == names
    result = predict_transaction(
        VALUES, ProbabilityModel(0.2), ProbabilityModel(0.8),
        IdentityPreprocessor(), names,
    )
    assert result["Baseline"]["prediction_label"] == "Legitimate"
    assert result["Hardened"]["prediction_label"] == "Fraud"
    assert result["Baseline"]["fraud_probability"] == 0.2
    assert result["Hardened"]["fraud_probability"] == 0.8


def test_artifact_contract_and_probability_risk_bands(tmp_path):
    paths = artifact_paths(tmp_path)
    assert paths["baseline_model"].name == "baseline_model.joblib"
    assert paths["hardened_model"].name == "hardened_model.joblib"
    assert paths["phase5_csv"].name == "phase5_hardening_comparison.csv"
    assert risk_label(0.1) == "Low"
    assert risk_label(0.5) == "Medium"
    assert risk_label(0.9) == "High"
