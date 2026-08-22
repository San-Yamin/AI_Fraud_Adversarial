"""Dataset-free Phase 3 threat-model, constraint, and metric tests."""

import numpy as np
import pandas as pd

from src.attack import (
    build_feature_threat_model,
    select_correctly_detected_test_fraud,
    validate_adversarial_constraints,
    verify_art_compatibility,
)
from src.evaluate import evaluate_adversarial_evasion


FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "sender_balance_change",
    "receiver_balance_change",
    "amount_to_sender_balance",
    "type_PAYMENT",
    "type_TRANSFER",
]


def encoded_rows() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "oldbalanceOrg": [500.0, 600.0, 700.0, 800.0],
            "newbalanceOrig": [400.0, 400.0, 400.0, 400.0],
            "oldbalanceDest": [0.0, 100.0, 200.0, 300.0],
            "newbalanceDest": [100.0, 300.0, 500.0, 700.0],
            "type_PAYMENT": [0.0, 0.0, 1.0, 0.0],
            "type_TRANSFER": [1.0, 1.0, 0.0, 1.0],
        }
    )
    frame["sender_balance_change"] = frame["oldbalanceOrg"] - frame["newbalanceOrig"]
    frame["receiver_balance_change"] = frame["newbalanceDest"] - frame["oldbalanceDest"]
    frame["amount_to_sender_balance"] = frame["amount"] / frame["oldbalanceOrg"]
    return frame.loc[:, FEATURES].astype("float32")


class AmountModel:
    n_features_in_ = len(FEATURES)

    def predict_proba(self, X):
        fraud = (X["amount"].to_numpy() >= 250).astype(float) * 0.8 + 0.1
        return np.column_stack([1 - fraud, fraud])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_art_compatibility_and_threat_model_are_explicit():
    X = encoded_rows()
    compatibility = verify_art_compatibility(AmountModel(), X.columns.tolist())
    assert compatibility["compatible"] is True
    assert compatibility["attack"] == "HopSkipJump"
    threat_model = build_feature_threat_model(X)
    threat = threat_model.set_index("feature_name")
    assert bool(threat.loc["amount", "mutable"]) is True
    assert threat.loc["sender_balance_change", "attack_handling"] == "recomputed only"
    assert bool(threat.loc["step", "mutable"]) is False
    assert bool(threat.loc["type_TRANSFER", "mutable"]) is False
    assert threat.loc["isFraud", "attack_handling"] == "excluded and protected"


def test_attack_population_is_correct_test_fraud_only_and_reproducible():
    X = encoded_rows()
    y = pd.Series([0, 0, 1, 1], index=X.index)
    first, first_y, metadata = select_correctly_detected_test_fraud(
        AmountModel(), X, y, sample_size=1, random_state=42
    )
    second, second_y, _ = select_correctly_detected_test_fraud(
        AmountModel(), X, y, sample_size=1, random_state=42
    )
    assert first.index.tolist() == second.index.tolist()
    assert first_y.eq(1).all() and second_y.eq(1).all()
    assert AmountModel().predict(first).tolist() == [1]
    assert metadata["selected_attacked_samples"] == 1


def test_constraint_validation_enforces_dependencies_and_protected_features():
    clean = encoded_rows().iloc[[2]].copy()
    adversarial = clean.copy()
    adversarial["amount"] *= 0.95
    adversarial["oldbalanceOrg"] *= 1.05
    adversarial["sender_balance_change"] = (
        adversarial["oldbalanceOrg"] - adversarial["newbalanceOrig"]
    )
    adversarial["amount_to_sender_balance"] = (
        adversarial["amount"] / adversarial["oldbalanceOrg"]
    )
    validate_adversarial_constraints(clean, adversarial, relative_bound=0.10)


def test_attack_metrics_use_actual_prediction_changes():
    clean = encoded_rows().iloc[[2, 3]].copy()
    adversarial = clean.copy()
    adversarial.loc[adversarial.index[0], "amount"] = 240.0
    adversarial.loc[adversarial.index[0], "amount_to_sender_balance"] = 240.0 / 700.0
    metrics, samples = evaluate_adversarial_evasion(
        AmountModel(),
        clean,
        adversarial,
        full_test_clean_fraud_recall=1.0,
    )
    assert metrics["number_attacked"] == 2
    assert metrics["successful_evasions"] == 1
    assert metrics["attack_success_rate"] == 0.5
    assert metrics["recall_under_attack"] == 0.5
    assert samples["successful_evasion"].tolist() == [True, False]
