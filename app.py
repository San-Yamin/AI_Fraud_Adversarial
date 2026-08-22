"""Presentation dashboard for completed Phases 1–5 (no Phase 7 simulation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    artifact_paths,
    available_transaction_types,
    default_output_directory,
    load_csv,
    load_json,
    load_prediction_artifacts,
    predict_transaction,
)


PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="AI Fraud Adversarial Robustness",
    page_icon="🛡️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .prototype-notice {padding: .8rem 1rem; border-radius: .5rem;
        background: #fff3cd; color: #664d03; font-weight: 650; margin-bottom: 1rem;}
    .workflow {padding: 1rem; border-radius: .6rem; background: #f2f6fc;
        text-align: center; font-size: 1.05rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def show_notice() -> None:
    st.markdown(
        '<div class="prototype-notice">Research prototype using synthetic PaySim data. '
        "Not a live banking system.</div>",
        unsafe_allow_html=True,
    )


def show_image(path: Path, caption: str) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure unavailable: `{path}`")


def metric_cards(metrics: dict) -> None:
    columns = st.columns(4)
    definitions = (
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1_score"),
        ("PR-AUC", "pr_auc"),
    )
    for column, (label, key) in zip(columns, definitions):
        value = metrics.get(key)
        column.metric(label, f"{float(value):.4f}" if value is not None else "Unavailable")


@st.cache_resource(show_spinner="Loading saved models…")
def cached_prediction_artifacts(output_path: str):
    return load_prediction_artifacts(artifact_paths(output_path))


default_outputs = default_output_directory(PROJECT_ROOT)
with st.sidebar:
    st.title("Fraud Security Lab")
    output_input = st.text_input("Saved outputs directory", str(default_outputs))
    page = st.radio(
        "Dashboard section",
        (
            "Project Overview",
            "Baseline Performance",
            "SHAP Explainability",
            "Attack Comparison",
            "Baseline vs Hardened",
            "Single Transaction",
            "Real-Time Simulation",
        ),
    )
    st.caption("The dashboard reads saved artifacts only. It does not retrain models.")

paths = artifact_paths(Path(output_input).expanduser())

st.title("AI-Based Fraud Detection and Adversarial Robustness")
show_notice()

if page == "Project Overview":
    st.header("Project Overview")
    st.write(
        "A security-focused machine-learning prototype that detects fraudulent digital "
        "banking transactions, explains model decisions, evaluates constrained adversarial "
        "evasion, compares attacks, and measures adversarial hardening."
    )
    st.info(
        "PaySim is a synthetic mobile-money transaction dataset. No real customer or "
        "banking data is used."
    )
    st.markdown(
        '<div class="workflow">Fraud Detection &nbsp;→&nbsp; SHAP &nbsp;→&nbsp; '
        "Adversarial Attack &nbsp;→&nbsp; Multiple Attack Comparison &nbsp;→&nbsp; Hardening</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Completed experiment stages")
    st.write(
        "Baseline XGBoost detection · Tree SHAP explainability · constrained ART attacks · "
        "multi-attack comparison · leakage-safe adversarial training"
    )

elif page == "Baseline Performance":
    st.header("Baseline Model Performance")
    try:
        payload = load_json(paths["phase1_metrics"])
        metric_cards(payload.get("metrics", {}))
        if "confusion_matrix" in payload:
            st.subheader("Saved confusion matrix values")
            st.dataframe(
                pd.DataFrame(
                    payload["confusion_matrix"],
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=["Predicted Legitimate", "Predicted Fraud"],
                ),
                use_container_width=True,
            )
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))
    left, right = st.columns(2)
    with left:
        show_image(paths["phase1_confusion"], "Baseline confusion matrix")
    with right:
        show_image(paths["phase1_pr_curve"], "Baseline precision–recall curve")

elif page == "SHAP Explainability":
    st.header("SHAP Explainability")
    st.caption("These are saved Tree SHAP results from the baseline model; SHAP is not recomputed here.")
    tabs = st.tabs(["Global importance", "Beeswarm", "Fraud example", "Legitimate example"])
    with tabs[0]:
        show_image(paths["shap_importance_plot"], "Global SHAP feature importance")
        try:
            st.dataframe(load_csv(paths["shap_importance_csv"]), use_container_width=True)
        except (FileNotFoundError, ValueError) as error:
            st.warning(str(error))
    with tabs[1]:
        show_image(paths["shap_beeswarm"], "SHAP beeswarm summary")
    with tabs[2]:
        show_image(paths["shap_fraud_waterfall"], "Correctly detected fraud transaction")
    with tabs[3]:
        show_image(paths["shap_legitimate_waterfall"], "Correctly detected legitimate transaction")

elif page == "Attack Comparison":
    st.header("Multiple Attack Comparison")
    try:
        comparison = load_csv(paths["phase4_csv"])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
    figures = (
        ("phase4_success", "Attack success rate comparison"),
        ("phase4_recall", "Recall under attack comparison"),
        ("phase4_perturbation", "Perturbation size comparison"),
        ("phase4_runtime", "Attack runtime comparison"),
    )
    for row_start in range(0, len(figures), 2):
        for column, (key, caption) in zip(st.columns(2), figures[row_start:row_start + 2]):
            with column:
                show_image(paths[key], caption)
    st.caption("Adversarial test transactions were evaluation-only and were never used for training.")

elif page == "Baseline vs Hardened":
    st.header("Baseline vs Hardened Model")
    try:
        hardened_payload = load_json(paths["phase5_metrics"])
        hardened_clean = hardened_payload.get("hardened_clean_evaluation", {}).get(
            "metrics", {}
        )
        if hardened_clean:
            st.subheader("Hardened clean-test performance")
            metric_cards(hardened_clean)
        comparison = load_csv(paths["phase5_csv"])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if not comparison.empty:
            row = comparison.iloc[0]
            cards = st.columns(4)
            for card, title, key in (
                (cards[0], "Baseline clean recall", "Baseline Clean Recall"),
                (cards[1], "Hardened clean recall", "Hardened Clean Recall"),
                (cards[2], "Recall recovery", "Recall Recovery"),
                (cards[3], "Attack-success reduction", "Reduction in Attack Success Rate"),
            ):
                card.metric(title, f"{float(row[key]):.4f}" if key in row else "Unavailable")
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(str(error))
    figures = (
        ("phase5_clean_recall", "Baseline vs hardened clean recall"),
        ("phase5_attack_recall", "Recall under fresh attack"),
        ("phase5_attack_success", "Attack success before and after hardening"),
        ("phase5_confusion", "Hardened model confusion matrix"),
    )
    for row_start in range(0, len(figures), 2):
        for column, (key, caption) in zip(st.columns(2), figures[row_start:row_start + 2]):
            with column:
                show_image(paths[key], caption)

elif page == "Single Transaction":
    st.header("Single Transaction Prediction")
    st.write("Enter PaySim-style values. The saved Phase 1 preprocessor is applied to both models.")
    try:
        baseline_model, hardened_model, preprocessor, feature_names = cached_prediction_artifacts(
            str(Path(output_input).expanduser())
        )
        transaction_types = available_transaction_types(preprocessor)
        with st.form("transaction_form"):
            left, right = st.columns(2)
            with left:
                transaction_type = st.selectbox("Transaction type", transaction_types)
                step = st.number_input("Step", min_value=1, value=1, step=1)
                amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
                oldbalance_org = st.number_input("Sender old balance", min_value=0.0, value=5000.0)
            with right:
                newbalance_orig = st.number_input("Sender new balance", min_value=0.0, value=4000.0)
                oldbalance_dest = st.number_input("Receiver old balance", min_value=0.0, value=0.0)
                newbalance_dest = st.number_input("Receiver new balance", min_value=0.0, value=1000.0)
            submitted = st.form_submit_button("Evaluate transaction", type="primary")
        if submitted:
            values = {
                "step": int(step), "type": transaction_type, "amount": amount,
                "oldbalanceOrg": oldbalance_org, "newbalanceOrig": newbalance_orig,
                "oldbalanceDest": oldbalance_dest, "newbalanceDest": newbalance_dest,
            }
            predictions = predict_transaction(
                values, baseline_model, hardened_model, preprocessor, feature_names
            )
            for column, model_name in zip(st.columns(2), ("Baseline", "Hardened")):
                result = predictions[model_name]
                with column:
                    st.subheader(model_name)
                    st.metric("Prediction", result["prediction_label"])
                    st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
                    st.metric("Risk band", result["risk_label"])
            st.caption("Low/Medium/High is a presentation band derived from probability, not a separately trained model.")
    except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as error:
        st.error(str(error))
        st.info("Set the sidebar path to the outputs directory containing both models and preprocessing artifacts.")

else:
    st.header("Real-Time Transaction Simulation")
    st.info("Planned for Phase 7 — not implemented in this dashboard phase.")
    st.write(
        "The future page will simulate PaySim rows arriving over time. It will not represent "
        "a live banking feed."
    )

st.divider()
st.caption("CST-8415 research prototype · synthetic PaySim data · saved experimental artifacts only")
