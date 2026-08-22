"""Presentation dashboard for completed Phases 1–8."""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    artifact_paths,
    available_transaction_types,
    default_dataset_path,
    default_output_directory,
    load_csv,
    load_json,
    load_prediction_artifacts,
    predict_transaction,
)
from src.realtime_simulation import (
    add_running_metrics,
    calculate_running_metrics,
    load_paysim_simulation_sequence,
    predict_simulation_batch,
    reset_simulation_state,
    save_simulation_results,
)
from src.concept_drift import (
    analyze_concept_drift,
    interpret_drift,
    plot_drift_summary,
    save_drift_outputs,
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


@st.cache_data(show_spinner="Preparing a small reproducible PaySim sequence…")
def cached_simulation_sequence(dataset_path: str, sample_size: int, seed: int):
    return load_paysim_simulation_sequence(
        dataset_path, sample_size=sample_size, random_state=seed,
    )


@st.cache_data(show_spinner="Evaluating chronological PaySim windows…")
def cached_concept_drift(dataset_path: str, output_path: str, n_windows: int):
    drift_paths = artifact_paths(output_path)
    _, hardened, preprocessor, feature_names = load_prediction_artifacts(drift_paths)
    importance = (
        load_csv(drift_paths["shap_importance_csv"])
        if drift_paths["shap_importance_csv"].is_file() else None
    )
    summary, metadata = analyze_concept_drift(
        dataset_path, hardened, preprocessor, feature_names,
        importance=importance, n_windows=n_windows,
    )
    interpretation = interpret_drift(summary)
    save_drift_outputs(
        summary, metadata, interpretation,
        drift_paths["phase8_csv"], drift_paths["phase8_json"],
    )
    plot_drift_summary(summary, Path(output_path) / "figures" / "phase8")
    return summary, metadata, interpretation


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
            "Concept Drift",
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

elif page == "Real-Time Simulation":
    st.header("Real-Time Transaction Simulation")
    st.warning(
        "Simulated transaction stream using synthetic PaySim data. "
        "Not connected to any live banking system."
    )
    st.caption(
        "A small class-aware sequence is read from PaySim in chunks. The actual isFraud "
        "label is retained only for evaluation and is never passed to the model."
    )

    dataset_input = st.text_input(
        "PaySim CSV path", str(default_dataset_path(PROJECT_ROOT)), key="simulation_dataset"
    )
    control_columns = st.columns(4)
    with control_columns[0]:
        transaction_count = st.number_input(
            "Transactions", min_value=10, max_value=500, value=50, step=10
        )
    with control_columns[1]:
        delay_seconds = st.slider(
            "Delay (seconds)", min_value=0.0, max_value=3.0, value=0.5, step=0.1
        )
    with control_columns[2]:
        batch_size = st.number_input(
            "Batch size", min_value=1, max_value=10, value=1, step=1
        )
    with control_columns[3]:
        deterministic = st.checkbox("Deterministic", value=True)
        requested_seed = st.number_input("Random seed", value=42, step=1)

    button_columns = st.columns(4)
    start_clicked = button_columns[0].button("▶ Start", type="primary", use_container_width=True)
    pause_clicked = button_columns[1].button("⏸ Stop / Pause", use_container_width=True)
    reset_clicked = button_columns[2].button("↺ Reset", use_container_width=True)
    export_clicked = button_columns[3].button("Save CSV", use_container_width=True)

    try:
        _, hardened_model, preprocessor, feature_names = cached_prediction_artifacts(
            str(Path(output_input).expanduser())
        )
        if start_clicked:
            selected_seed = int(requested_seed) if deterministic else int(time.time_ns() % (2**31 - 1))
            configuration = (str(Path(dataset_input).expanduser()), int(transaction_count), selected_seed)
            if (
                "phase7_state" not in st.session_state
                or st.session_state.get("phase7_configuration") != configuration
            ):
                sequence = cached_simulation_sequence(*configuration)
                st.session_state.phase7_state = reset_simulation_state(sequence)
                st.session_state.phase7_configuration = configuration
            st.session_state.phase7_state["running"] = True
            st.session_state.phase7_state["next_due"] = 0.0
        if pause_clicked and "phase7_state" in st.session_state:
            st.session_state.phase7_state["running"] = False
        if reset_clicked and "phase7_state" in st.session_state:
            sequence = st.session_state.phase7_state["sequence"]
            st.session_state.phase7_state = reset_simulation_state(sequence)

        @st.fragment(run_every=0.25)
        def render_live_simulation() -> None:
            if "phase7_state" not in st.session_state:
                st.info("Choose the controls and press Start to prepare the simulation sequence.")
                return
            state = st.session_state.phase7_state
            now = time.monotonic()
            if state["running"] and now >= state["next_due"]:
                start = int(state["position"])
                end = min(start + int(batch_size), len(state["sequence"]))
                if start < end:
                    batch = state["sequence"].iloc[start:end]
                    batch_results = predict_simulation_batch(
                        batch, hardened_model, preprocessor, feature_names,
                        sequence_start=start + 1,
                    )
                    state["results"] = (
                        batch_results.reset_index(drop=True)
                        if state["results"].empty
                        else pd.concat([state["results"], batch_results], ignore_index=True)
                    )
                    state["position"] = end
                    state["next_due"] = now + float(delay_seconds)
                if state["position"] >= len(state["sequence"]):
                    state["running"] = False

            results = state["results"]
            metrics = calculate_running_metrics(results)
            progress = state["position"] / len(state["sequence"])
            st.progress(progress, text=f"Processed {state['position']} of {len(state['sequence'])}")
            st.caption("Status: " + ("Running" if state["running"] else "Paused / complete"))

            first_row = st.columns(6)
            for card, label, key in zip(
                first_row,
                ("Processed", "Actual fraud", "Detected fraud", "Missed fraud", "False positives", "True positives"),
                ("total_processed", "actual_fraud", "detected_fraud", "missed_fraud", "false_positives", "true_positives"),
            ):
                card.metric(label, metrics[key])
            second_row = st.columns(3)
            second_row[0].metric("Running precision", f"{metrics['precision']:.3f}")
            second_row[1].metric("Running recall", f"{metrics['recall']:.3f}")
            second_row[2].metric("Running F1", f"{metrics['f1']:.3f}")

            if results.empty:
                return
            latest = results.iloc[-1]
            st.subheader("Latest transaction")
            latest_columns = st.columns(7)
            latest_values = (
                ("Sequence", int(latest["transaction_sequence"])),
                ("Step", int(latest["step"])),
                ("Type", latest["type"]),
                ("Amount", f"{latest['amount']:,.2f}"),
                ("Fraud probability", f"{latest['fraud_probability']:.2%}"),
                ("Prediction", latest["predicted_label"]),
                ("Outcome", latest["status"]),
            )
            for column, (label, value) in zip(latest_columns, latest_values):
                column.metric(label, value)
            st.caption(f"Actual label: {latest['actual_label_name']}")

            enriched = add_running_metrics(results)
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.subheader("Fraud probability over sequence")
                st.line_chart(
                    enriched.set_index("transaction_sequence")[["fraud_probability"]]
                )
                st.subheader("Running recall")
                st.line_chart(
                    enriched.set_index("transaction_sequence")[["running_recall"]]
                )
            with chart_right:
                st.subheader("Running prediction counts")
                st.line_chart(
                    enriched.set_index("transaction_sequence")[[
                        "running_predicted_fraud", "running_predicted_legitimate"
                    ]]
                )
                st.subheader("Recent transactions")
                st.dataframe(
                    results.tail(15)[[
                        "transaction_sequence", "step", "type", "amount",
                        "fraud_probability", "predicted_label", "actual_label_name", "status",
                    ]],
                    use_container_width=True, hide_index=True,
                )

        render_live_simulation()

        if export_clicked:
            if "phase7_state" not in st.session_state or st.session_state.phase7_state["results"].empty:
                st.warning("Process at least one transaction before saving results.")
            else:
                saved_path = save_simulation_results(
                    st.session_state.phase7_state["results"], paths["phase7_results"]
                )
                st.success(f"Saved actual simulation results to {saved_path}")
        if "phase7_state" in st.session_state and not st.session_state.phase7_state["results"].empty:
            csv_bytes = st.session_state.phase7_state["results"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download current results", csv_bytes,
                file_name="phase7_simulation_results.csv", mime="text/csv",
            )
    except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as error:
        st.error(str(error))
        st.info("Check the PaySim path and the saved hardened-model artifact directory.")

else:
    st.header("Concept Drift")
    st.warning(
        "Concept drift analysis is simulated using PaySim step-based chronological windows."
    )
    st.caption(
        "The saved hardened model is evaluated without retraining. Fixed equal-width step "
        "ranges are compared with Window 1 as the feature-distribution reference."
    )
    drift_dataset = st.text_input(
        "PaySim CSV path", str(default_dataset_path(PROJECT_ROOT)), key="drift_dataset"
    )
    n_windows = st.slider("Chronological windows", min_value=8, max_value=12, value=8)
    run_drift = st.button("Run concept drift analysis", type="primary")
    refresh_drift = st.button("Clear cached analysis and rerun")
    if refresh_drift:
        cached_concept_drift.clear()
        run_drift = True
    if run_drift:
        try:
            drift_summary, drift_metadata, drift_interpretation = cached_concept_drift(
                str(Path(drift_dataset).expanduser()),
                str(Path(output_input).expanduser()), int(n_windows),
            )
            st.session_state.phase8_summary = drift_summary
            st.session_state.phase8_metadata = drift_metadata
            st.session_state.phase8_interpretation = drift_interpretation
        except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as error:
            st.error(str(error))
    elif paths["phase8_csv"].is_file() and paths["phase8_json"].is_file():
        try:
            st.session_state.phase8_summary = load_csv(paths["phase8_csv"])
            saved_payload = load_json(paths["phase8_json"])
            st.session_state.phase8_metadata = saved_payload.get("metadata", {})
            st.session_state.phase8_interpretation = saved_payload.get("interpretation", {})
        except (FileNotFoundError, ValueError, KeyError) as error:
            st.error(str(error))

    if "phase8_summary" in st.session_state:
        drift_summary = st.session_state.phase8_summary
        interpretation = st.session_state.get("phase8_interpretation", {})
        st.dataframe(drift_summary, use_container_width=True, hide_index=True)
        cards = st.columns(4)
        cards[0].metric("Maximum drift score", f"{interpretation.get('maximum_drift_score', float('nan')):.3f}")
        cards[1].metric("Maximum drift level", interpretation.get("maximum_drift_level", "Unavailable"))
        cards[2].metric(
            "Meaningful feature drift",
            "Observed" if interpretation.get("meaningful_feature_drift_observed") else "Not observed",
        )
        cards[3].metric(
            "Performance degradation",
            "Observed" if interpretation.get("performance_degradation_observed") else "Not observed",
        )
        st.info(interpretation.get("threshold_note", ""))
        st.caption(interpretation.get("scope_note", ""))
        figures = (
            ("phase8_fraud_rate", "Fraud rate over time"),
            ("phase8_recall", "Recall over time"),
            ("phase8_f1", "F1 over time"),
            ("phase8_probability", "Mean fraud probability over time"),
            ("phase8_feature_drift", "Feature drift versus early reference"),
        )
        for row_start in range(0, len(figures), 2):
            for column, (key, caption) in zip(
                st.columns(2), figures[row_start:row_start + 2]
            ):
                with column:
                    show_image(paths[key], caption)
        st.success(f"Saved Phase 8 outputs under {Path(output_input).expanduser()}")

st.divider()
st.caption("CST-8415 research prototype · synthetic PaySim data · saved experimental artifacts only")
