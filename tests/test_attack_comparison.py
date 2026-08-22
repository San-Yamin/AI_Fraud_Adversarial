"""Dataset-free tests for fair Phase 4 comparison and ranking."""

from copy import deepcopy
import json
from pathlib import Path

from src import config
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


def test_phase4_config_names_match_notebook_contract():
    expected = {
        "ATTACK_INIT_EVAL": 50,
        "ATTACK_INIT_SIZE": 30,
        "ATTACK_MAX_EVAL": 500,
        "ATTACK_MAX_ITER": 10,
        "ATTACK_RELATIVE_BOUND": 0.10,
        "PHASE4_ATTACK_SAMPLE_SIZE": config.ATTACK_SAMPLE_SIZE,
        "BOUNDARY_MAX_ITER": 50,
        "BOUNDARY_NUM_TRIAL": 10,
        "BOUNDARY_SAMPLE_SIZE": 10,
        "ZOO_LEARNING_RATE": 0.05,
        "ZOO_MAX_ITER": 20,
        "ZOO_NB_PARALLEL": 5,
    }
    for name, default in expected.items():
        assert hasattr(config, name), name
        value = getattr(config, name)
        assert type(value) is type(default)


def test_cell29_uses_compatible_config_lookup_not_brittle_direct_import():
    notebook_path = Path(__file__).parents[1] / "notebooks/fraud_adversarial_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell29_code = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "phase4_attack_parameters" in "".join(cell["source"])
    )
    assert "from src.config import" not in cell29_code
    assert "from src import config as project_config" in cell29_code
    for name in (
        "ATTACK_INIT_EVAL",
        "ATTACK_INIT_SIZE",
        "ATTACK_MAX_EVAL",
        "ATTACK_MAX_ITER",
        "ATTACK_RELATIVE_BOUND",
        "BOUNDARY_MAX_ITER",
        "BOUNDARY_NUM_TRIAL",
        "BOUNDARY_SAMPLE_SIZE",
        "PHASE4_ATTACK_SAMPLE_SIZE",
        "ZOO_LEARNING_RATE",
        "ZOO_MAX_ITER",
        "ZOO_NB_PARALLEL",
    ):
        assert name in cell29_code


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
