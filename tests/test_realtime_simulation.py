"""Dataset-free Phase 7 sequence, leakage, prediction, metric, and reset tests."""

import numpy as np
import pandas as pd

from src.realtime_simulation import (
    add_running_metrics,
    calculate_running_metrics,
    create_simulation_sequence,
    load_paysim_simulation_sequence,
    predict_simulation_batch,
    preprocess_simulation_rows,
    reset_simulation_state,
)


def paysim_rows() -> pd.DataFrame:
    rows = []
    for index in range(20):
        fraud = int(index >= 15)
        amount = float(100 + index * 50)
        rows.append(
            {
                "step": index + 1, "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": amount, "oldbalanceOrg": amount + 500,
                "newbalanceOrig": 500.0, "oldbalanceDest": 0.0,
                "newbalanceDest": amount, "isFraud": fraud,
            }
        )
    return pd.DataFrame(rows)


class InspectingPreprocessor:
    def __init__(self):
        self.seen_columns = []

    def transform(self, frame):
        self.seen_columns.append(frame.columns.tolist())
        return frame[["amount", "sender_balance_change"]].to_numpy()

    def get_feature_names_out(self):
        return np.array(["amount", "sender_balance_change"])


class AmountModel:
    def predict_proba(self, frame):
        fraud = (frame["amount"].to_numpy() >= 800).astype(float) * 0.8 + 0.1
        return np.column_stack([1 - fraud, fraud])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def test_sequence_is_class_aware_deterministic_and_resettable():
    data = paysim_rows()
    first = create_simulation_sequence(data, sample_size=10, random_state=42)
    second = create_simulation_sequence(data, sample_size=10, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    assert first["isFraud"].value_counts().to_dict() == {0: 8, 1: 2}
    state = reset_simulation_state(first)
    state["position"] = 5
    reset = reset_simulation_state(state["sequence"])
    assert reset["position"] == 0
    assert reset["results"].empty
    pd.testing.assert_frame_equal(reset["sequence"], first)


def test_chunked_csv_sequence_is_small_and_reproducible(tmp_path):
    source = tmp_path / "paysim.csv"
    paysim_rows().to_csv(source, index=False)
    first = load_paysim_simulation_sequence(
        source, sample_size=10, random_state=7, chunk_size=4
    )
    second = load_paysim_simulation_sequence(
        source, sample_size=10, random_state=7, chunk_size=4
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10
    assert set(first["isFraud"]) == {0, 1}


def test_target_is_never_passed_to_preprocessor_and_predictions_are_actual():
    sequence = create_simulation_sequence(paysim_rows(), sample_size=10, random_state=42)
    preprocessor = InspectingPreprocessor()
    names = ["amount", "sender_balance_change"]
    encoded = preprocess_simulation_rows(sequence.iloc[:2], preprocessor, names)
    assert encoded.columns.tolist() == names
    assert all("isFraud" not in columns for columns in preprocessor.seen_columns)

    results = predict_simulation_batch(
        sequence, AmountModel(), preprocessor, names, sequence_start=1
    )
    assert len(results) == 10
    assert results["fraud_probability"].between(0, 1).all()
    assert set(results["status"]).issubset(
        {"Correct Detection", "Missed Fraud", "False Positive", "Correct Legitimate"}
    )


def test_running_metrics_and_charts_use_cumulative_predictions():
    results = pd.DataFrame(
        {
            "actual_label": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 1],
            "transaction_sequence": [1, 2, 3, 4],
        }
    )
    metrics = calculate_running_metrics(results)
    assert metrics == {
        "total_processed": 4, "actual_fraud": 2, "detected_fraud": 1,
        "missed_fraud": 1, "false_positives": 1, "true_positives": 1,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }
    enriched = add_running_metrics(results)
    assert enriched["running_recall"].tolist() == [0.0, 1.0, 0.5, 0.5]
    assert enriched["running_predicted_fraud"].tolist() == [0, 1, 1, 2]
