"""Memory-aware global and local Tree SHAP explainability for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config

# Keep Phase 2 compatible with a Colab checkout that still has the completed
# Phase 1 config. New constants remain configurable when the updated config is
# present, while safe memory-aware defaults prevent an import-time failure.
RANDOM_SEED = config.RANDOM_SEED
SHAP_MAX_SAMPLES = getattr(config, "SHAP_MAX_SAMPLES", 1000)
SHAP_MAX_FRAUD_SAMPLES = getattr(config, "SHAP_MAX_FRAUD_SAMPLES", 250)


def load_phase1_artifacts(
    model_path: str | Path,
    preprocessor_path: str | Path,
    feature_names_path: str | Path,
) -> tuple[Any, Any, list[str]]:
    """Load, validate, and return the existing Phase 1 artifacts."""
    paths = [Path(model_path), Path(preprocessor_path), Path(feature_names_path)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 1 artifacts: {missing}")
    model = joblib.load(paths[0])
    preprocessor = joblib.load(paths[1])
    feature_names = list(joblib.load(paths[2]))
    if not feature_names:
        raise ValueError("Saved feature_names.joblib is empty")
    model_features = getattr(model, "n_features_in_", len(feature_names))
    if int(model_features) != len(feature_names):
        raise ValueError(
            "Baseline model and saved feature names have different feature counts"
        )
    return model, preprocessor, feature_names


def transform_with_saved_preprocessor(
    preprocessor: Any, X_raw: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    """Transform held-out raw features without refitting the preprocessor."""
    values = preprocessor.transform(X_raw)
    if values.shape[1] != len(feature_names):
        raise ValueError("Transformed data does not match saved feature names")
    return pd.DataFrame(values, columns=feature_names, index=X_raw.index).astype(
        "float32"
    )


def prepare_shap_evaluation_sample(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    max_samples: int = SHAP_MAX_SAMPLES,
    max_fraud_samples: int = SHAP_MAX_FRAUD_SAMPLES,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible, bounded test-only sample with fraud representation.

    Up to ``max_fraud_samples`` held-out fraud rows are selected first; remaining
    capacity is filled with held-out legitimate rows. This sample is intended
    for explanation rather than population prevalence estimation.
    """
    if max_samples < 1 or max_fraud_samples < 0:
        raise ValueError("SHAP sample limits must be positive")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indices must match")

    fraud_indices = y_test.index[y_test == 1]
    legitimate_indices = y_test.index[y_test == 0]
    fraud_count = min(len(fraud_indices), max_fraud_samples, max_samples)
    legitimate_count = min(len(legitimate_indices), max_samples - fraud_count)
    if legitimate_count == 0 and len(legitimate_indices) and fraud_count == max_samples:
        fraud_count -= 1
        legitimate_count = 1

    fraud_selected = y_test.loc[fraud_indices].sample(
        n=fraud_count, random_state=random_state
    ).index
    legitimate_selected = y_test.loc[legitimate_indices].sample(
        n=legitimate_count, random_state=random_state
    ).index
    selected = fraud_selected.append(legitimate_selected)
    selected = pd.Series(selected).sample(frac=1.0, random_state=random_state).tolist()
    return X_test.loc[selected].copy(), y_test.loc[selected].copy()


def create_tree_explainer(model: Any) -> Any:
    """Create the tree-specific explainer without training or changing the model."""
    return shap.TreeExplainer(model, model_output="raw")


def _positive_class_explanation(explanation: Any) -> Any:
    """Normalize binary/multi-output SHAP results to the fraud output."""
    values = np.asarray(explanation.values)
    if values.ndim <= 2:
        return explanation
    base_values = np.asarray(explanation.base_values)
    fraud_base = base_values[..., -1] if base_values.ndim > 1 else base_values
    return shap.Explanation(
        values=values[..., -1],
        base_values=fraud_base,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def compute_shap_values(explainer: Any, X: pd.DataFrame) -> Any:
    """Compute SHAP values for the supplied bounded held-out sample."""
    explanation = explainer(X, check_additivity=False)
    return _positive_class_explanation(explanation)


def rank_global_importance(explanation: Any) -> pd.DataFrame:
    """Rank features by mean absolute SHAP magnitude."""
    values = np.asarray(explanation.values)
    names = list(explanation.feature_names)
    return (
        pd.DataFrame(
            {
                "feature": names,
                "mean_absolute_shap": np.abs(values).mean(axis=0),
                "mean_signed_shap": values.mean(axis=0),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    )


def select_correct_examples(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Reproducibly select one correct fraud and one correct legitimate row."""
    predictions = np.asarray(model.predict(X_test)).astype(int)
    probabilities = np.asarray(model.predict_proba(X_test))[:, 1]
    actual = y_test.to_numpy(dtype=int)
    rng = np.random.default_rng(random_state)
    selections: dict[str, dict[str, Any]] = {}
    for label, name in ((1, "fraud"), (0, "legitimate")):
        eligible = np.flatnonzero((actual == label) & (predictions == label))
        if not len(eligible):
            raise ValueError(f"No correctly detected {name} transaction is available")
        position = int(rng.choice(eligible))
        selections[name] = {
            "position": position,
            "index": X_test.index[position],
            "actual_label": int(actual[position]),
            "predicted_label": int(predictions[position]),
            "fraud_probability": float(probabilities[position]),
        }
    return selections


def explain_selected_row(
    explainer: Any,
    X_test: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    """Explain one selected row and produce a data-derived direction table."""
    row = X_test.iloc[[selection["position"]]]
    explanation = compute_shap_values(explainer, row)[0]
    values = np.asarray(explanation.values)
    table = pd.DataFrame(
        {
            "feature": row.columns,
            "feature_value": row.iloc[0].to_numpy(),
            "shap_value": values,
            "direction": np.where(
                values > 0,
                "toward fraud",
                np.where(values < 0, "toward legitimate", "neutral"),
            ),
        }
    ).sort_values("shap_value", key=lambda series: series.abs(), ascending=False)
    return explanation, table.reset_index(drop=True)


def save_global_plots(
    explanation: Any,
    output_dir: str | Path,
    *,
    max_display: int = 15,
) -> tuple[Path, Path]:
    """Save global SHAP importance and beeswarm figures."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bar_path = directory / "global_feature_importance_bar.png"
    beeswarm_path = directory / "global_summary_beeswarm.png"

    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.gcf().savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.gcf().savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    return bar_path, beeswarm_path


def save_waterfall_plot(
    explanation: Any,
    path: str | Path,
    *,
    max_display: int = 15,
) -> Path:
    """Save a local SHAP waterfall plot."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.gcf().savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    return output_path


def build_interpretation(
    selection: dict[str, Any], table: pd.DataFrame, *, top_n: int = 5
) -> dict[str, Any]:
    """Build an interpretation only from computed predictions and SHAP values."""
    toward_fraud = table.loc[table["shap_value"] > 0].nlargest(top_n, "shap_value")
    toward_legitimate = table.loc[table["shap_value"] < 0].nsmallest(
        top_n, "shap_value"
    )
    return {
        **selection,
        "top_features_toward_fraud": toward_fraud[
            ["feature", "feature_value", "shap_value"]
        ].to_dict(orient="records"),
        "top_features_toward_legitimate": toward_legitimate[
            ["feature", "feature_value", "shap_value"]
        ].to_dict(orient="records"),
        "shap_output_space": "raw model margin; positive pushes toward fraud",
    }


def save_phase2_outputs(
    output_dir: str | Path,
    importance: pd.DataFrame,
    local_tables: dict[str, pd.DataFrame],
    interpretations: dict[str, dict[str, Any]],
    sample_labels: pd.Series,
) -> dict[str, Path]:
    """Persist computed tables and interpretations without fabricated values."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"importance": directory / "global_feature_importance.csv"}
    importance.to_csv(paths["importance"], index=False)
    for name, table in local_tables.items():
        path = directory / f"local_{name}_explanation.csv"
        table.to_csv(path, index=False)
        paths[f"local_{name}"] = path
    interpretation_path = directory / "phase2_interpretation.json"
    payload = {
        "global_sample": {
            "rows": int(len(sample_labels)),
            "class_counts": {
                str(key): int(value)
                for key, value in sample_labels.value_counts().sort_index().items()
            },
            "note": "Class-aware held-out explanation sample; not a prevalence estimate.",
        },
        "local_explanations": interpretations,
    }
    interpretation_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    paths["interpretation"] = interpretation_path
    return paths
