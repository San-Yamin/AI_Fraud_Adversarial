"""Phase 1 class, confusion-matrix, and precision-recall figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_original_class_distribution(y: Any, output_dir: str | Path) -> Any:
    counts = pd.Series(y).value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Legitimate", "Fraud"], y=counts.values, ax=ax)
    ax.set(title="Original Fraud Class Distribution", ylabel="Transactions")
    _save(fig, Path(output_dir) / "original_class_distribution.png")
    return fig


def plot_smote_distributions(y_before: Any, y_after: Any, output_dir: str | Path) -> Any:
    before = pd.Series(y_before).value_counts().reindex([0, 1], fill_value=0)
    after = pd.Series(y_after).value_counts().reindex([0, 1], fill_value=0)
    frame = pd.DataFrame({"Before SMOTE": before, "After SMOTE": after})
    fig, ax = plt.subplots(figsize=(7, 4))
    frame.T.plot(kind="bar", ax=ax)
    ax.set(title="Training Distribution Before and After SMOTE", ylabel="Transactions")
    ax.legend(["Legitimate", "Fraud"])
    _save(fig, Path(output_dir) / "training_distribution_smote.png")
    return fig


def plot_confusion_matrix(matrix: Any, output_dir: str | Path) -> Any:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set(title="Baseline Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    _save(fig, Path(output_dir) / "baseline_confusion_matrix.png")
    return fig


def plot_precision_recall(result: dict[str, Any], output_dir: str | Path) -> Any:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["recall_curve"], result["precision_curve"])
    ax.set(
        title=f"Baseline Precision-Recall Curve (AP={result['metrics']['pr_auc']:.4f})",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    _save(fig, Path(output_dir) / "baseline_precision_recall_curve.png")
    return fig
