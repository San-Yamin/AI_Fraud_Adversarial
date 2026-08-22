"""Dataset-free chronological-window and drift tests for final Phase 8."""

import json

import numpy as np
import pandas as pd

from src.concept_drift import (
    analyze_concept_drift,
    assign_step_windows,
    build_step_window_edges,
    drift_level,
    interpret_drift,
    kolmogorov_smirnov_statistic,
    plot_drift_summary,
    save_drift_outputs,
    select_drift_features,
)


class InspectingPreprocessor:
    def __init__(self):
        self.seen_columns = []

    def transform(self, frame):
        self.seen_columns.append(frame.columns.tolist())
        return frame[["step", "amount", "sender_balance_change"]].to_numpy()

    def get_feature_names_out(self):
        return np.array(["step", "amount", "sender_balance_change"])


class AmountModel:
    def predict_proba(self, frame):
        probability = np.where(frame["amount"].to_numpy() >= 500, 0.9, 0.1)
        return np.column_stack([1 - probability, probability])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def temporal_rows() -> pd.DataFrame:
    rows = []
    for step in range(1, 9):
        for offset in range(6):
            fraud = int(offset == 0 and step % 2 == 0)
            amount = 800.0 if fraud else 100.0 + step
            rows.append(
                {
                    "step": step, "type": "TRANSFER" if fraud else "PAYMENT",
                    "amount": amount, "oldbalanceOrg": amount + 500,
                    "newbalanceOrig": 500.0, "oldbalanceDest": 0.0,
                    "newbalanceDest": amount, "isFraud": fraud,
                }
            )
    return pd.DataFrame(rows)


def test_fixed_step_windows_are_chronological_and_complete():
    edges = build_step_window_edges(1, 8, 8)
    assignments = assign_step_windows(np.arange(1, 9), edges)
    assert assignments.tolist() == list(range(8))


def test_ks_metric_thresholds_and_shap_selection_are_explicit():
    assert kolmogorov_smirnov_statistic([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0
    assert kolmogorov_smirnov_statistic([0, 0, 0], [1, 1, 1]) == 1.0
    assert drift_level(0.05) == "Low"
    assert drift_level(0.15) == "Moderate"
    assert drift_level(0.25) == "High"
    importance = pd.DataFrame({"feature": ["step", "amount", "sender_balance_change"]})
    selected = select_drift_features(
        ["step", "amount", "sender_balance_change"], importance, max_features=2
    )
    assert selected == ["amount", "sender_balance_change"]


def test_chunked_analysis_uses_target_only_for_metrics_and_handles_no_fraud(tmp_path):
    source = tmp_path / "paysim.csv"
    temporal_rows().to_csv(source, index=False)
    preprocessor = InspectingPreprocessor()
    summary, metadata = analyze_concept_drift(
        source, AmountModel(), preprocessor,
        ["step", "amount", "sender_balance_change"],
        importance=pd.DataFrame({"feature": ["amount", "sender_balance_change"]}),
        n_windows=8, feature_sample_size=20, chunk_size=7,
    )
    assert len(summary) == 8
    assert summary["Transaction Count"].sum() == len(temporal_rows())
    assert summary["Recall"].isna().any()
    assert summary["PR-AUC"].isna().any()
    assert all("isFraud" not in columns for columns in preprocessor.seen_columns)
    assert metadata["model_retrained"] is False
    assert metadata["target_used_as_feature"] is False
    assert "step" not in metadata["drift_features"]


def test_interpretation_outputs_and_figures_are_data_derived(tmp_path):
    summary = pd.DataFrame(
        {
            "Window": [f"Window {i}" for i in range(1, 5)],
            "Step Range": ["1–1", "2–2", "3–3", "4–4"],
            "Transaction Count": [10] * 4, "Fraud Count": [1] * 4,
            "Fraud Rate": [0.1] * 4, "Precision": [1.0] * 4,
            "Recall": [1.0, 1.0, 0.8, 0.7], "F1": [1.0, 1.0, 0.8, 0.7],
            "PR-AUC": [1.0] * 4, "Mean Fraud Probability": [0.1, 0.1, 0.2, 0.3],
            "Drift Score": [0.0, 0.05, 0.15, 0.25],
            "Drift Level": ["Low", "Low", "Moderate", "High"],
        }
    )
    interpretation = interpret_drift(summary)
    assert interpretation["meaningful_feature_drift_observed"] is True
    assert interpretation["performance_degradation_observed"] is True
    csv_path, json_path = tmp_path / "drift.csv", tmp_path / "drift.json"
    save_drift_outputs(summary, {"method": "test"}, interpretation, csv_path, json_path)
    assert pd.read_csv(csv_path).shape[0] == 4
    assert json.loads(json_path.read_text())["interpretation"]["maximum_drift_level"] == "High"
    figures = plot_drift_summary(summary, tmp_path / "figures")
    assert len(figures) == 5
    assert all(path.is_file() for path in figures)
