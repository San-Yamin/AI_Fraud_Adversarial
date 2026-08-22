"""Constrained, test-only targeted HopSkipJump evasion for Phase 3."""

from __future__ import annotations

import time
from typing import Any

import art
import numpy as np
import pandas as pd
from art.attacks.evasion import BoundaryAttack, HopSkipJump, ZooAttack
from art.estimators.classification import BlackBoxClassifier

from src.config import (
    ATTACK_INIT_EVAL,
    ATTACK_INIT_SIZE,
    ATTACK_MAX_EVAL,
    ATTACK_MAX_ITER,
    ATTACK_RELATIVE_BOUND,
    ATTACK_SAMPLE_SIZE,
    RANDOM_SEED,
)

DIRECT_MUTABLE_FEATURES = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
DERIVED_FEATURES = (
    "sender_balance_change",
    "receiver_balance_change",
    "amount_to_sender_balance",
)


def verify_art_compatibility(model: Any, feature_names: list[str]) -> dict[str, Any]:
    """Verify the actual model interface and ART components used by Phase 3."""
    missing_features = sorted(
        (set(DIRECT_MUTABLE_FEATURES) | set(DERIVED_FEATURES)) - set(feature_names)
    )
    if missing_features:
        raise ValueError(f"Encoded matrix is missing attack features: {missing_features}")
    for method in ("predict", "predict_proba"):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"Baseline model must implement {method}()")
    model_features = int(getattr(model, "n_features_in_", len(feature_names)))
    if model_features != len(feature_names):
        raise ValueError("Model feature count does not match saved feature names")
    requirements = [item.__name__ for item in HopSkipJump._estimator_requirements]
    return {
        "art_version": art.__version__,
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "model_feature_count": model_features,
        "art_estimator": "BlackBoxClassifier",
        "attack": "HopSkipJump",
        "attack_type": "targeted decision-based black-box",
        "estimator_requirements": requirements,
        "compatible": True,
    }


def build_feature_threat_model(
    X_reference: pd.DataFrame,
    *,
    relative_bound: float = ATTACK_RELATIVE_BOUND,
) -> pd.DataFrame:
    """Describe direct, dependent, and protected encoded features."""
    if not 0 < relative_bound <= 1:
        raise ValueError("relative_bound must be in (0, 1]")
    records: list[dict[str, Any]] = []
    for feature in X_reference.columns:
        observed_min = float(X_reference[feature].min())
        observed_max = float(X_reference[feature].max())
        if feature in DIRECT_MUTABLE_FEATURES:
            mutable = True
            handling = "directly attacked"
            valid_range = (
                f"per-row [max(0, clean×{1-relative_bound:.2f}), "
                f"clean×{1+relative_bound:.2f}]"
            )
            dependency = {
                "amount": "amount_to_sender_balance",
                "oldbalanceOrg": "sender_balance_change, amount_to_sender_balance",
                "newbalanceOrig": "sender_balance_change",
                "oldbalanceDest": "receiver_balance_change",
                "newbalanceDest": "receiver_balance_change",
            }[feature]
            reason = "Bounded non-negative monetary value"
        elif feature in DERIVED_FEATURES:
            mutable = True
            handling = "recomputed only"
            valid_range = "deterministically derived from bounded raw monetary values"
            dependency = {
                "sender_balance_change": "oldbalanceOrg - newbalanceOrig",
                "receiver_balance_change": "newbalanceDest - oldbalanceDest",
                "amount_to_sender_balance": "amount / oldbalanceOrg; 0 when denominator is 0",
            }[feature]
            reason = "Cannot be attacked independently; relationship is enforced"
        elif feature == "step":
            mutable = False
            handling = "protected"
            valid_range = f"fixed clean value; observed [{observed_min:g}, {observed_max:g}]"
            dependency = "none"
            reason = "Transaction time/context is outside the attacker capability"
        elif feature.startswith("type_"):
            mutable = False
            handling = "protected"
            valid_range = "fixed clean one-hot value in {0, 1}"
            dependency = "entire transaction-type one-hot group"
            reason = "Protects categorical validity; no invalid/fractional encoding"
        else:
            mutable = False
            handling = "protected"
            valid_range = f"fixed clean value; observed [{observed_min:g}, {observed_max:g}]"
            dependency = "none"
            reason = "Not included in the documented attacker capability"
        records.append(
            {
                "feature_name": feature,
                "mutable": mutable,
                "attack_handling": handling,
                "valid_range": valid_range,
                "reason": reason,
                "dependencies": dependency,
            }
        )
    for excluded, reason in (
        ("isFraud", "Target label; never an attack input"),
        ("isFlaggedFraud", "Leakage field dropped before modeling"),
        ("nameOrig", "Identifier excluded from the baseline"),
        ("nameDest", "Identifier excluded from the baseline"),
    ):
        records.append(
            {
                "feature_name": excluded,
                "mutable": False,
                "attack_handling": "excluded and protected",
                "valid_range": "not present in encoded model input",
                "reason": reason,
                "dependencies": "none",
            }
        )
    return pd.DataFrame(records)


def select_correctly_detected_test_fraud(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    sample_size: int = ATTACK_SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Select a reproducible subset only from correctly detected test fraud."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indices must match")
    fraud_mask = y_test == 1
    X_fraud = X_test.loc[fraud_mask]
    predictions = np.asarray(model.predict(X_fraud)).astype(int)
    eligible = X_fraud.loc[predictions == 1]
    if eligible.empty:
        raise ValueError("No correctly detected test fraud is available to attack")
    selected = eligible.sample(
        n=min(sample_size, len(eligible)), random_state=random_state
    ).copy()
    labels = y_test.loc[selected.index].copy()
    metadata = {
        "total_test_fraud": int(fraud_mask.sum()),
        "correctly_detected_test_fraud": int((predictions == 1).sum()),
        "full_test_clean_fraud_recall": float((predictions == 1).mean()),
        "selected_attacked_samples": int(len(selected)),
    }
    return selected, labels, metadata


def _sample_bounds(clean_row: pd.Series, relative_bound: float) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_row.loc[list(DIRECT_MUTABLE_FEATURES)].to_numpy(dtype=np.float32)
    lower = np.maximum(0.0, clean * (1.0 - relative_bound)).astype(np.float32)
    upper = np.maximum(lower, clean * (1.0 + relative_bound)).astype(np.float32)
    return lower, upper


def _decode_normalized_variables(
    normalized: np.ndarray,
    clean_row: pd.Series,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    """Decode bounded attack variables and enforce all feature dependencies."""
    normalized = np.asarray(normalized, dtype=np.float32)
    raw = lower + np.clip(normalized, 0.0, 1.0) * (upper - lower)
    decoded = pd.DataFrame(
        np.repeat(clean_row.to_numpy(dtype=np.float32)[None, :], len(raw), axis=0),
        columns=clean_row.index,
    )
    decoded.loc[:, list(DIRECT_MUTABLE_FEATURES)] = raw
    decoded["sender_balance_change"] = (
        decoded["oldbalanceOrg"] - decoded["newbalanceOrig"]
    )
    decoded["receiver_balance_change"] = (
        decoded["newbalanceDest"] - decoded["oldbalanceDest"]
    )
    sender = decoded["oldbalanceOrg"].to_numpy(dtype=np.float32)
    amount = decoded["amount"].to_numpy(dtype=np.float32)
    decoded["amount_to_sender_balance"] = np.divide(
        amount,
        sender,
        out=np.zeros_like(amount, dtype=np.float32),
        where=sender != 0,
    )
    return decoded.astype("float32")


def _clean_normalized_point(lower: np.ndarray, upper: np.ndarray, clean: np.ndarray) -> np.ndarray:
    span = upper - lower
    return np.divide(
        clean - lower,
        span,
        out=np.zeros_like(clean, dtype=np.float32),
        where=span != 0,
    ).reshape(1, -1)


def generate_constrained_hopskipjump(
    model: Any,
    X_attack: pd.DataFrame,
    *,
    relative_bound: float = ATTACK_RELATIVE_BOUND,
    max_iter: int = ATTACK_MAX_ITER,
    max_eval: int = ATTACK_MAX_EVAL,
    init_eval: int = ATTACK_INIT_EVAL,
    init_size: int = ATTACK_INIT_SIZE,
    random_state: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run targeted HSJ per row in a dependency-safe five-variable space."""
    if X_attack.empty:
        raise ValueError("X_attack must contain at least one row")
    compatibility = verify_art_compatibility(model, X_attack.columns.tolist())
    adversarial_rows: list[pd.Series] = []
    query_counts: list[int] = []
    started = time.perf_counter()

    for offset, (_, clean_row) in enumerate(X_attack.iterrows()):
        lower, upper = _sample_bounds(clean_row, relative_bound)
        clean_variables = clean_row.loc[list(DIRECT_MUTABLE_FEATURES)].to_numpy(
            dtype=np.float32
        )
        clean_normalized = _clean_normalized_point(lower, upper, clean_variables)
        query_count = [0]

        def predict_fn(normalized: np.ndarray) -> np.ndarray:
            decoded = _decode_normalized_variables(normalized, clean_row, lower, upper)
            query_count[0] += len(decoded)
            return np.asarray(model.predict_proba(decoded), dtype=np.float32)

        estimator = BlackBoxClassifier(
            predict_fn=predict_fn,
            input_shape=(len(DIRECT_MUTABLE_FEATURES),),
            nb_classes=2,
            clip_values=(0.0, 1.0),
        )
        attack = HopSkipJump(
            classifier=estimator,
            targeted=True,
            norm=2,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            init_size=init_size,
            batch_size=64,
            verbose=verbose,
        )
        np.random.seed(random_state + offset)
        adversarial_normalized = attack.generate(
            x=clean_normalized,
            y=np.array([0], dtype=np.int64),
        )
        decoded = _decode_normalized_variables(
            adversarial_normalized, clean_row, lower, upper
        ).iloc[0]
        decoded.name = clean_row.name
        adversarial_rows.append(decoded)
        query_counts.append(query_count[0])

    X_adversarial = pd.DataFrame(adversarial_rows).loc[:, X_attack.columns]
    validate_adversarial_constraints(
        X_attack, X_adversarial, relative_bound=relative_bound
    )
    return X_adversarial.astype("float32"), {
        **compatibility,
        "runtime_seconds": float(time.perf_counter() - started),
        "total_model_queries": int(sum(query_counts)),
        "queries_per_sample": query_counts,
        "relative_bound": float(relative_bound),
        "random_seed": int(random_state),
        "attack_parameters": {
            "target_class": 0,
            "norm": 2,
            "max_iter": int(max_iter),
            "max_eval": int(max_eval),
            "init_eval": int(init_eval),
            "init_size": int(init_size),
        },
    }


def validate_adversarial_constraints(
    X_clean: pd.DataFrame,
    X_adversarial: pd.DataFrame,
    *,
    relative_bound: float = ATTACK_RELATIVE_BOUND,
    tolerance: float = 1e-4,
) -> None:
    """Reject negative, out-of-bound, dependent, or protected modifications."""
    if not X_clean.columns.equals(X_adversarial.columns):
        raise ValueError("Clean and adversarial feature columns differ")
    if not X_clean.index.equals(X_adversarial.index):
        raise ValueError("Clean and adversarial sample indices differ")
    direct = list(DIRECT_MUTABLE_FEATURES)
    clean_values = X_clean[direct].to_numpy(dtype=float)
    adversarial_values = X_adversarial[direct].to_numpy(dtype=float)
    lower = np.maximum(0.0, clean_values * (1.0 - relative_bound))
    upper = clean_values * (1.0 + relative_bound)
    if np.any(adversarial_values < lower - tolerance) or np.any(
        adversarial_values > upper + tolerance
    ):
        raise ValueError("Adversarial monetary values exceed the threat-model bounds")
    if np.any(adversarial_values < -tolerance):
        raise ValueError("Adversarial monetary values must be non-negative")

    protected = [
        name
        for name in X_clean.columns
        if name not in DIRECT_MUTABLE_FEATURES and name not in DERIVED_FEATURES
    ]
    if not np.allclose(
        X_clean[protected].to_numpy(),
        X_adversarial[protected].to_numpy(),
        atol=tolerance,
        rtol=0,
    ):
        raise ValueError("An immutable feature was changed")
    if not np.allclose(
        X_adversarial["sender_balance_change"],
        X_adversarial["oldbalanceOrg"] - X_adversarial["newbalanceOrig"],
        atol=tolerance,
    ):
        raise ValueError("Sender balance dependency is inconsistent")
    if not np.allclose(
        X_adversarial["receiver_balance_change"],
        X_adversarial["newbalanceDest"] - X_adversarial["oldbalanceDest"],
        atol=tolerance,
    ):
        raise ValueError("Receiver balance dependency is inconsistent")
    sender = X_adversarial["oldbalanceOrg"].to_numpy(dtype=float)
    expected_ratio = np.divide(
        X_adversarial["amount"].to_numpy(dtype=float),
        sender,
        out=np.zeros_like(sender),
        where=sender != 0,
    )
    if not np.allclose(
        X_adversarial["amount_to_sender_balance"], expected_ratio, atol=tolerance
    ):
        raise ValueError("Amount-to-sender dependency is inconsistent")


def verify_comparison_attacks() -> pd.DataFrame:
    """Report actual ART requirements for the three Phase 4 attacks."""
    rows = []
    details = {
        "HopSkipJump": (
            HopSkipJump,
            "decision-based",
            "Uses predicted class decisions; no gradients required",
        ),
        "BoundaryAttack": (
            BoundaryAttack,
            "decision-based",
            "Uses predicted class decisions; no gradients required",
        ),
        "ZooAttack": (
            ZooAttack,
            "score-based finite-difference",
            "Uses class scores from predict_proba; no model gradients required",
        ),
    }
    for name, (attack_class, attack_type, reason) in details.items():
        requirements = [item.__name__ for item in attack_class._estimator_requirements]
        rows.append(
            {
                "attack": name,
                "art_version": art.__version__,
                "attack_type": attack_type,
                "estimator": "BlackBoxClassifier",
                "estimator_requirements": ", ".join(requirements),
                "gradient_required": False,
                "compatible": requirements == ["BaseEstimator", "ClassifierMixin"],
                "compatibility_reason": reason,
            }
        )
    result = pd.DataFrame(rows)
    if not result["compatible"].all():
        raise RuntimeError("One or more selected ART attacks is incompatible")
    return result


def _build_comparison_attack(
    attack_name: str,
    estimator: BlackBoxClassifier,
    parameters: dict[str, Any],
    *,
    verbose: bool,
) -> Any:
    """Instantiate one verified Phase 4 ART attack."""
    if attack_name == "HopSkipJump":
        return HopSkipJump(
            classifier=estimator,
            targeted=True,
            norm=2,
            max_iter=int(parameters["max_iter"]),
            max_eval=int(parameters["max_eval"]),
            init_eval=int(parameters["init_eval"]),
            init_size=int(parameters["init_size"]),
            batch_size=64,
            verbose=verbose,
        )
    if attack_name == "BoundaryAttack":
        return BoundaryAttack(
            estimator=estimator,
            targeted=True,
            max_iter=int(parameters["max_iter"]),
            num_trial=int(parameters["num_trial"]),
            sample_size=int(parameters["sample_size"]),
            init_size=int(parameters["init_size"]),
            batch_size=64,
            verbose=verbose,
        )
    if attack_name == "ZooAttack":
        return ZooAttack(
            classifier=estimator,
            targeted=True,
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            binary_search_steps=1,
            initial_const=0.001,
            abort_early=True,
            use_resize=False,
            use_importance=False,
            nb_parallel=int(parameters["nb_parallel"]),
            batch_size=1,
            verbose=verbose,
        )
    raise ValueError(f"Unsupported comparison attack: {attack_name}")


def generate_constrained_comparison_attack(
    attack_name: str,
    model: Any,
    X_attack: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    relative_bound: float = ATTACK_RELATIVE_BOUND,
    random_state: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one ART attack in the identical Phase 3 constrained latent space."""
    compatibility = verify_art_compatibility(model, X_attack.columns.tolist())
    selected = verify_comparison_attacks().set_index("attack")
    if attack_name not in selected.index:
        raise ValueError(f"Attack is not in the verified comparison set: {attack_name}")
    adversarial_rows: list[pd.Series] = []
    query_counts: list[int] = []
    started = time.perf_counter()

    for offset, (_, clean_row) in enumerate(X_attack.iterrows()):
        lower, upper = _sample_bounds(clean_row, relative_bound)
        clean_variables = clean_row.loc[list(DIRECT_MUTABLE_FEATURES)].to_numpy(
            dtype=np.float32
        )
        clean_normalized = _clean_normalized_point(lower, upper, clean_variables)
        query_count = [0]

        def predict_fn(normalized: np.ndarray) -> np.ndarray:
            decoded = _decode_normalized_variables(normalized, clean_row, lower, upper)
            query_count[0] += len(decoded)
            return np.asarray(model.predict_proba(decoded), dtype=np.float32)

        estimator = BlackBoxClassifier(
            predict_fn=predict_fn,
            input_shape=(len(DIRECT_MUTABLE_FEATURES),),
            nb_classes=2,
            clip_values=(0.0, 1.0),
        )
        attack = _build_comparison_attack(
            attack_name, estimator, parameters, verbose=verbose
        )
        np.random.seed(random_state + offset)
        adversarial_normalized = attack.generate(
            x=clean_normalized,
            y=np.array([0], dtype=np.int64),
        )
        decoded = _decode_normalized_variables(
            adversarial_normalized, clean_row, lower, upper
        ).iloc[0]
        decoded.name = clean_row.name
        adversarial_rows.append(decoded)
        query_counts.append(query_count[0])

    X_adversarial = pd.DataFrame(adversarial_rows).loc[:, X_attack.columns]
    validate_adversarial_constraints(
        X_attack, X_adversarial, relative_bound=relative_bound
    )
    return X_adversarial.astype("float32"), {
        **compatibility,
        "attack": attack_name,
        "attack_type": selected.loc[attack_name, "attack_type"],
        "runtime_seconds": float(time.perf_counter() - started),
        "total_model_queries": int(sum(query_counts)),
        "queries_per_sample": query_counts,
        "relative_bound": float(relative_bound),
        "random_seed": int(random_state),
        "attack_parameters": parameters,
        "target_class": 0,
        "evaluation_only": True,
    }
