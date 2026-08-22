"""Dataset-free tests for fair Phase 4 comparison and ranking."""

from copy import deepcopy

from src.attack import verify_comparison_attacks
from src.attack_comparison import build_comparison_table, identify_comparison_findings


def metric(success, recall, perturbation, runtime, queries):
    return {
        "attacked_sample_clean_recall": 1.0,
        "recall_under_attack": recall,
        "recall_drop_on_attacked_sample": 1.0 - recall,
        "attack_success_rate": success,
        "number_attacked": 20,
        "successful_evasions": int(success * 20),
        "average_l2_perturbation": perturbation,
        "max_l2_perturbation": perturbation * 2,
        "attack_execution": {
            "runtime_seconds": runtime,
            "total_model_queries": queries,
        },
    }


def test_three_selected_attacks_are_art_black_box_compatible():
    compatibility = verify_comparison_attacks().set_index("attack")
    assert compatibility.index.tolist() == ["HopSkipJump", "BoundaryAttack", "ZooAttack"]
    assert compatibility["compatible"].all()
    assert not compatibility["gradient_required"].any()


def test_comparison_table_and_findings_use_actual_metrics():
    results = {
        "HopSkipJump": metric(0.40, 0.60, 2.0, 10.0, 1000),
        "BoundaryAttack": metric(0.30, 0.70, 3.0, 8.0, 800),
        "ZooAttack": metric(0.40, 0.60, 4.0, 12.0, 1200),
    }
    table = build_comparison_table(deepcopy(results))
    findings = identify_comparison_findings(table)
    assert findings["strongest_by_evasion_rate"] == ["HopSkipJump", "ZooAttack"]
    assert findings["largest_recall_drop"] == ["HopSkipJump", "ZooAttack"]
    assert findings["largest_mean_perturbation"] == ["ZooAttack"]
    assert findings["fastest_attack"] == ["BoundaryAttack"]
    assert table["Number Attacked"].nunique() == 1
