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


def plot_attack_probabilities(samples: pd.DataFrame, output_dir: str | Path) -> Any:
    """Compare clean and adversarial fraud probabilities per attacked sample."""
    frame = samples.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.index, frame["clean_fraud_probability"], marker="o", label="Clean")
    ax.plot(
        frame.index,
        frame["adversarial_fraud_probability"],
        marker="x",
        label="Adversarial",
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold")
    ax.set(
        title="Fraud Probability Before and After Targeted HopSkipJump",
        xlabel="Attacked test-fraud sample",
        ylabel="Fraud probability",
        ylim=(0, 1.02),
    )
    ax.legend()
    _save(fig, Path(output_dir) / "attack_fraud_probabilities.png")
    return fig


def plot_attack_perturbations(samples: pd.DataFrame, output_dir: str | Path) -> Any:
    """Show encoded-space L2 perturbation sizes and successful evasions."""
    frame = samples.reset_index(drop=True)
    colors = frame["successful_evasion"].map({True: "tab:red", False: "tab:blue"})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(frame.index, frame["l2_perturbation"], color=colors)
    ax.set(
        title="Constrained Adversarial Perturbation Size",
        xlabel="Attacked test-fraud sample",
        ylabel="Encoded-space L2 distance",
    )
    _save(fig, Path(output_dir) / "attack_perturbation_sizes.png")
    return fig


def plot_attack_comparison(table: pd.DataFrame, output_dir: str | Path) -> list[Any]:
    """Save success, recall, perturbation, and runtime comparison figures."""
    directory = Path(output_dir)
    specifications = (
        (
            "Attack Success Rate",
            "Attack Success Rate Comparison",
            "Evasion rate",
            "attack_success_rate_comparison.png",
        ),
        (
            "Recall Under Attack",
            "Fraud Recall Under Attack",
            "Recall",
            "recall_under_attack_comparison.png",
        ),
        (
            "Mean Perturbation",
            "Mean Encoded-Space L2 Perturbation",
            "Mean L2 distance",
            "perturbation_size_comparison.png",
        ),
        (
            "Runtime Seconds",
            "Attack Runtime Comparison",
            "Runtime (seconds)",
            "runtime_comparison.png",
        ),
    )
    figures = []
    for column, title, ylabel, filename in specifications:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=table, x="Attack", y=column, ax=ax)
        ax.set(title=title, xlabel="Attack", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=15)
        _save(fig, directory / filename)
        figures.append(fig)
    return figures


def plot_hardening_comparison(
    comparison: pd.DataFrame,
    hardened_confusion_matrix: Any,
    output_dir: str | Path,
) -> list[Any]:
    """Save clean, adversarial, attack-success, and hardened-CM figures."""
    directory = Path(output_dir)
    figures = []

    clean = comparison.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Baseline", "Hardened"],
        [clean["Baseline Clean Recall"], clean["Hardened Clean Recall"]],
    )
    ax.set(title="Baseline vs Hardened Clean Recall", ylabel="Recall", ylim=(0, 1.05))
    _save(fig, directory / "clean_recall_comparison.png")
    figures.append(fig)

    for baseline_column, hardened_column, title, filename in (
        (
            "Baseline Recall Under Attack", "Hardened Recall Under Attack",
            "Recall Under Fresh Attack", "recall_under_attack_comparison.png",
        ),
        (
            "Baseline Attack Success Rate", "Hardened Attack Success Rate",
            "Attack Success Before vs After Hardening", "attack_success_comparison.png",
        ),
    ):
        frame = comparison[["Attack", baseline_column, hardened_column]].melt(
            id_vars="Attack", var_name="Model", value_name="Value"
        )
        frame["Model"] = frame["Model"].str.replace(
            " Recall Under Attack", "", regex=False
        ).str.replace(" Attack Success Rate", "", regex=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=frame, x="Attack", y="Value", hue="Model", ax=ax)
        ax.set(title=title, xlabel="Attack", ylabel="Rate", ylim=(0, 1.05))
        _save(fig, directory / filename)
        figures.append(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        hardened_confusion_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"], ax=ax,
    )
    ax.set(title="Hardened Model Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    _save(fig, directory / "hardened_confusion_matrix.png")
    figures.append(fig)
    return figures
