"""Small dependency and ML workflow checks that do not require PaySim."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification


def test_core_imports_and_tiny_binary_dataset():
    import art  # noqa: F401
    import joblib  # noqa: F401
    import matplotlib  # noqa: F401
    import seaborn  # noqa: F401
    import shap  # noqa: F401
    import streamlit  # noqa: F401

    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(4)])
    assert frame.shape == (40, 4)
    assert set(np.unique(y)) == {0, 1}


def test_smote_xgboost_prediction_and_shap():
    shap = pytest.importorskip("shap")
    try:
        from xgboost import XGBClassifier
    except Exception as error:  # Native OpenMP may be absent outside Colab.
        pytest.skip(f"XGBoost runtime unavailable in this environment: {error}")
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    assert np.bincount(y_resampled)[0] == np.bincount(y_resampled)[1]

    model = XGBClassifier(
        n_estimators=4,
        max_depth=2,
        learning_rate=0.2,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(X_resampled, y_resampled)
    predictions = model.predict(X[:3])
    assert predictions.shape == (3,)

    explanation = shap.TreeExplainer(model)(X[:2])
    assert explanation.values.shape[0] == 2
