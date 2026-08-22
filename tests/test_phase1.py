"""Dataset-free unit tests for the Phase 1 pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_paysim, summarize_paysim
from src.evaluate import evaluate_binary_classifier
from src.preprocessing import fit_transform_train_test, prepare_features_and_target
from src.train import stratified_train_test_split


def tiny_paysim_frame() -> pd.DataFrame:
    rows = 20
    return pd.DataFrame(
        {
            "step": np.arange(rows),
            "type": ["PAYMENT", "TRANSFER"] * 10,
            "amount": np.linspace(10, 200, rows),
            "nameOrig": [f"C{i}" for i in range(rows)],
            "oldbalanceOrg": np.linspace(0, 1000, rows),
            "newbalanceOrig": np.linspace(0, 800, rows),
            "nameDest": [f"M{i}" for i in range(rows)],
            "oldbalanceDest": np.linspace(0, 300, rows),
            "newbalanceDest": np.linspace(10, 500, rows),
            "isFraud": [0] * 16 + [1] * 4,
            "isFlaggedFraud": [0] * rows,
        }
    )


def test_loader_validates_and_samples_across_classes(tmp_path):
    csv_path = tmp_path / "paysim.csv"
    tiny_paysim_frame().to_csv(csv_path, index=False)
    loaded = load_paysim(
        csv_path,
        run_mode="DEVELOPMENT_MODE",
        normal_samples=6,
        fraud_samples=3,
        chunk_size=5,
    )
    summary = summarize_paysim(loaded)
    assert summary["shape"] == (9, 11)
    assert summary["class_counts"] == {"legitimate": 6, "fraud": 3}
    assert summary["fraud_percentage"] > 0


def test_preprocessing_separates_target_leakage_and_identifiers():
    X, y = prepare_features_and_target(tiny_paysim_frame())
    assert y.name == "isFraud"
    assert set(y.unique()) == {0, 1}
    assert "isFraud" not in X
    assert "isFlaggedFraud" not in X
    assert "nameOrig" not in X
    assert "nameDest" not in X
    assert np.isfinite(X["amount_to_sender_balance"]).all()
    assert X.loc[0, "amount_to_sender_balance"] == 0


def test_split_is_disjoint_and_preprocessor_is_train_fitted():
    X, y = prepare_features_and_target(tiny_paysim_frame())
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.25
    )
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(X)
    assert set(y_train.index) == set(X_train.index)
    assert set(y_test.index) == set(X_test.index)

    encoded_train, encoded_test, preprocessor = fit_transform_train_test(
        X_train, X_test
    )
    assert encoded_train.columns.tolist() == encoded_test.columns.tolist()
    assert encoded_train.columns.tolist() == preprocessor.get_feature_names_out().tolist()
    assert any(name.startswith("type_") for name in encoded_train.columns)


def test_metric_calculation_on_tiny_model():
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression(random_state=42).fit(X, y)
    result = evaluate_binary_classifier(model, X, y)
    assert set(result["metrics"]) == {
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "accuracy_supplementary",
    }
    assert len(result["confusion_matrix"]) == 2
