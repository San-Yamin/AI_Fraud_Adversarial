"""Dataset-free leakage, augmentation, and comparison tests for Phase 5."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from src.hardening import (
    build_augmented_training_set,
    build_hardening_comparison,
    select_adversarial_training_sources,
    select_common_correct_test_fraud,
)


class ThresholdModel:
    def __init__(self, threshold=2.0):
        self.threshold = threshold

    def predict(self, X):
        return (X["amount"].to_numpy() >= self.threshold).astype(int)


def test_training_sources_are_fraud_only_and_reproducible():
    X = pd.DataFrame({"amount": [0.0, 2.0, 3.0, 4.0]}, index=[10, 11, 12, 13])
    y = pd.Series([0, 1, 1, 1], index=X.index)
    first, labels, metadata = select_adversarial_training_sources(
        ThresholdModel(), X, y, sample_size=2, random_state=42
    )
    second, _, _ = select_adversarial_training_sources(
        ThresholdModel(), X, y, sample_size=2, random_state=42
    )
    assert first.index.tolist() == second.index.tolist()
    assert labels.eq(1).all()
    assert metadata["source_split"] == "training"
    assert metadata["phase3_or_phase4_test_samples_reused"] is False


def test_augmentation_rejects_test_source_overlap_and_preserves_fraud_label():
    clean = pd.DataFrame({"amount": [1.0, 2.0]})
    adversarial = pd.DataFrame({"amount": [1.8]}, index=[12])
    with pytest.raises(ValueError, match="overlap"):
        build_augmented_training_set(
            clean, [0, 1], adversarial,
            source_training_indices=adversarial.index,
            untouched_test_indices=pd.Index([12, 20]),
        )
    X_augmented, y_augmented, metadata = build_augmented_training_set(
        clean, [0, 1], adversarial,
        source_training_indices=adversarial.index,
        untouched_test_indices=pd.Index([20]),
    )
    assert len(X_augmented) == 3
    assert y_augmented.iloc[-1] == 1
    assert metadata["test_rows_used_for_training"] == 0


def test_common_test_population_is_detected_by_both_models():
    X = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 1, 1])
    selected, metadata = select_common_correct_test_fraud(
        ThresholdModel(2.0), ThresholdModel(3.0), X, y,
        sample_size=2, random_state=42,
    )
    assert np.all(ThresholdModel(2.0).predict(selected) == 1)
    assert np.all(ThresholdModel(3.0).predict(selected) == 1)
    assert metadata["source_split"] == "untouched_test"
    assert metadata["used_for_training"] is False


def test_hardening_comparison_uses_actual_model_results():
    baseline_clean = {"precision": .8, "recall": .7, "f1_score": .75, "pr_auc": .81}
    hardened_clean = {"precision": .79, "recall": .72, "f1_score": .755, "pr_auc": .82}
    baseline = {"HSJ": {
        "number_attacked": 10, "recall_under_attack": .4,
        "recall_drop_on_attacked_sample": .6, "attack_success_rate": .6,
    }}
    hardened = {"HSJ": {
        "number_attacked": 10, "recall_under_attack": .7,
        "recall_drop_on_attacked_sample": .3, "attack_success_rate": .3,
    }}
    table = build_hardening_comparison(
        baseline_clean, hardened_clean, baseline, hardened
    )
    assert table.loc[0, "Recall Recovery"] == pytest.approx(.3)
    assert table.loc[0, "Reduction in Attack Success Rate"] == pytest.approx(.3)
    assert table.loc[0, "Clean Recall Change"] == pytest.approx(.02)


def test_notebook_contains_only_phase5_cells_38_through_46():
    path = Path(__file__).parents[1] / "notebooks/fraud_adversarial_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    for number in range(38, 47):
        assert any(f"Cell {number} —" in text for text in markdown)
    phase6 = next(text for text in markdown if text.startswith("# PHASE 6"))
    assert "Implemented" in phase6

    cell38 = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "phase5_attack_parameters" in "".join(cell.get("source", []))
    )
    assert "getattr(" in cell38
    assert "project_config.HARDENING_TRAIN_ATTACK_SAMPLE_SIZE" not in cell38
    assert "project_config.HARDENING_TEST_ATTACK_SAMPLE_SIZE" not in cell38
