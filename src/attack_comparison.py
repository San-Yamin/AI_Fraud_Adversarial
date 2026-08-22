"""Fair Phase 4 comparison of constrained ART evasion attacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.attack import generate_constrained_comparison_attack
from src.evaluate import evaluate_adversarial_evasion


def run_comparison_attack(
    attack_name: str,
    model: Any,
    X_clean: pd.DataFrame,
    *,
    full_test_clean_fraud_recall: float,
    parameters: dict[str, Any],
    relative_bound: float,
    random_state: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Generate and evaluate one attack on the shared test-only population."""
    X_adversarial, execution = generate_constrained_comparison_attack(
        attack_name,
        model,
        X_clean,
        parameters,
        relative_bound=relative_bound,
        random_state=random_state,
        verbose=verbose,
    )
    metrics, samples = evaluate_adversarial_evasion(
        model,
        X_clean,
        X_adversarial,
        full_test_clean_fraud_recall=full_test_clean_fraud_recall,
        attack_metadata=execution,
        attack_name=attack_name,
    )
    return X_adversarial, metrics, samples


def build_comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the required comparable metric table from actual results."""
    if len(results) < 2:
        raise ValueError("At least two attack results are required for comparison")
    rows = []
    for name, metrics in results.items():
        execution = metrics.get("attack_execution", {})
        rows.append(
            {
                "Attack": name,
                "Clean Recall": metrics["attacked_sample_clean_recall"],
                "Recall Under Attack": metrics["recall_under_attack"],
                "Recall Drop": metrics["recall_drop_on_attacked_sample"],
                "Attack Success Rate": metrics["attack_success_rate"],
                "Number Attacked": metrics["number_attacked"],
                "Successful Evasions": metrics["successful_evasions"],
                "Mean Perturbation": metrics["average_l2_perturbation"],
                "Maximum Perturbation": metrics["max_l2_perturbation"],
                "Runtime Seconds": execution.get("runtime_seconds"),
                "Query Count": execution.get("total_model_queries"),
            }
        )
    table = pd.DataFrame(rows)
    if table["Number Attacked"].nunique() != 1:
        raise ValueError("Attacks were not evaluated on comparable population sizes")
    return table


def _ties_for_extreme(
    table: pd.DataFrame, column: str, *, largest: bool, tolerance: float = 1e-12
) -> list[str]:
    extreme = table[column].max() if largest else table[column].min()
    matches = (table[column] - extreme).abs() <= tolerance
    return table.loc[matches, "Attack"].tolist()


def identify_comparison_findings(table: pd.DataFrame) -> dict[str, Any]:
    """Identify leaders while retaining ties and avoiding overstated differences."""
    findings = {
        "strongest_by_evasion_rate": _ties_for_extreme(
            table, "Attack Success Rate", largest=True
        ),
        "largest_recall_drop": _ties_for_extreme(table, "Recall Drop", largest=True),
        "largest_mean_perturbation": _ties_for_extreme(
            table, "Mean Perturbation", largest=True
        ),
        "largest_maximum_perturbation": _ties_for_extreme(
            table, "Maximum Perturbation", largest=True
        ),
        "fastest_attack": _ties_for_extreme(table, "Runtime Seconds", largest=False),
    }
    success_spread = float(
        table["Attack Success Rate"].max() - table["Attack Success Rate"].min()
    )
    findings["evasion_rate_spread"] = success_spread
    findings["interpretation_note"] = (
        "Evasion rates are tied or nearly tied; do not overstate attack differences."
        if success_spread <= 0.01
        else "Attack rankings apply only to this population, threat model, and query budget."
    )
    return findings


def save_comparison_outputs(
    table: pd.DataFrame,
    findings: dict[str, Any],
    compatibility: pd.DataFrame,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    """Save computed comparison results and methodological context."""
    csv_output = Path(csv_path)
    json_output = Path(json_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_output, index=False)
    payload = {
        "comparison": table.to_dict(orient="records"),
        "findings": findings,
        "compatibility": compatibility.to_dict(orient="records"),
        "methodology": {
            "population": "same correctly detected held-out test fraud for every attack",
            "target_class": 0,
            "constraints": "same Phase 3 non-negative ±10% monetary bounds and dependencies",
            "evaluation_only": True,
            "used_for_training": False,
        },
    }
    json_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
