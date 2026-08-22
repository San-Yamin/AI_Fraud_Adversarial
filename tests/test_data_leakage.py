"""Guards for the training-only SMOTE contract."""

import inspect

import numpy as np

from src.preprocessing import resample_training_data


def test_smote_helper_accepts_no_validation_or_test_arguments():
    parameters = inspect.signature(resample_training_data).parameters
    assert "X_test" not in parameters
    assert "y_test" not in parameters
    assert "X_validation" not in parameters
    assert "y_validation" not in parameters


def test_training_smote_does_not_mutate_held_out_data():
    X_train = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [1.1]])
    y_train = np.array([0, 0, 0, 0, 1, 1])
    X_test = np.array([[99.0], [100.0]])
    y_test = np.array([0, 1])
    X_test_before = X_test.copy()
    y_test_before = y_test.copy()

    X_resampled, y_resampled = resample_training_data(
        X_train, y_train, sampling_strategy=1.0, k_neighbors=1
    )

    assert len(X_resampled) > len(X_train)
    assert np.bincount(y_resampled).tolist() == [4, 4]
    np.testing.assert_array_equal(X_test, X_test_before)
    np.testing.assert_array_equal(y_test, y_test_before)
