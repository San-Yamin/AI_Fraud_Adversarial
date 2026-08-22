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
from src.deployment import prepare_deployment_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="AI Fraud Adversarial Robustness",
    page_icon="🛡️",
    layout="wide",
)

deployment_outputs = None
deployment_bootstrap_error = None
try:
    deployment_outputs = prepare_deployment_artifacts(PROJECT_ROOT)
except (OSError, RuntimeError, ValueError) as error:
    deployment_bootstrap_error = str(error)


st.markdown(
    """
    <style>
    :root {
        --canvas: #F7F8FA; --surface: #FFFFFF; --text: #1F2937;
        --muted: #6B7280; --primary: #1D4ED8; --success: #059669;
        --warning: #D97706; --danger: #DC2626; --border: #E5E7EB;
    }
    .stApp {background: var(--canvas); color: var(--text);}
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: var(--text); letter-spacing: -0.025em;}
    p, .stCaption {color: var(--muted);}
    [data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid var(--border);}
    [data-testid="stSidebar"] .block-container {padding: 1.5rem 1.1rem;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap: .25rem;}
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        padding: .5rem .65rem; border-radius: .45rem; transition: background .15s ease;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {background: #F3F4F6;}
    .brand {display:flex; align-items:center; gap:.75rem; margin:.15rem 0 1.35rem;}
    .brand-mark {width:2.25rem; height:2.25rem; border-radius:.55rem; background:#1D4ED8;
        color:white; display:flex; align-items:center; justify-content:center;
        font-weight:750; letter-spacing:-.04em;}
    .brand-name {font-size:1rem; font-weight:750; color:var(--text); line-height:1.15;}
    .brand-meta {font-size:.72rem; color:var(--muted); margin-top:.15rem;}
    .dashboard-header {padding:.1rem 0 1.15rem; border-bottom:1px solid var(--border); margin-bottom:1rem;}
    .dashboard-eyebrow {font-size:.72rem; text-transform:uppercase; letter-spacing:.11em;
        color:var(--primary); font-weight:750; margin-bottom:.35rem;}
    .dashboard-title {font-size:1.78rem; line-height:1.2; font-weight:760;
        color:var(--text); letter-spacing:-.035em; margin:0;}
    .dashboard-subtitle {font-size:.92rem; color:var(--muted); margin-top:.42rem; max-width:760px;}
    .page-header {margin:1.7rem 0 1.15rem;}
    .page-kicker {font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
        color:var(--primary); font-weight:750; margin-bottom:.3rem;}
    .page-title {font-size:1.45rem; font-weight:750; color:var(--text);
        letter-spacing:-.025em; margin:0;}
    .page-description {font-size:.91rem; color:var(--muted); margin-top:.38rem; max-width:800px;}
    .prototype-notice {padding:.75rem .95rem; border:1px solid #F3D7A1;
        border-left:4px solid var(--warning); border-radius:.5rem;
        background:#FFFBEB; color:#78350F; font-size:.86rem; font-weight:600; margin-bottom:1rem;}
    .workflow {display:flex; flex-wrap:wrap; gap:.45rem; align-items:center;
        padding:1rem; border:1px solid var(--border); border-radius:.65rem; background:white;}
    .workflow-step {padding:.46rem .7rem; border-radius:.4rem; background:#EFF6FF;
        color:#1E3A8A; font-size:.82rem; font-weight:650;}
    .workflow-arrow {color:#9CA3AF; font-size:.8rem;}
    .info-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; margin:.8rem 0 1rem;}
    .info-card {background:white; border:1px solid var(--border); border-radius:.65rem; padding:1rem;}
    .info-label {font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
        color:var(--muted); font-weight:700;}
    .info-value {font-size:.95rem; color:var(--text); font-weight:650; margin-top:.35rem;}
    [data-testid="stMetric"] {background:#FFFFFF; border:1px solid var(--border);
        border-radius:.65rem; padding:.85rem 1rem; box-shadow:0 1px 2px rgba(15,23,42,.035);}
    [data-testid="stMetricLabel"] {color:var(--muted); font-size:.78rem;}
    [data-testid="stMetricValue"] {color:var(--text); font-size:1.42rem; font-weight:730;}
    [data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF;
        border-color:var(--border) !important; border-radius:.7rem !important;
        box-shadow:0 1px 2px rgba(15,23,42,.03);}
    [data-testid="stForm"] {background:#FFFFFF; border:1px solid var(--border);
        border-radius:.7rem; padding:1.15rem;}
    [data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:.55rem; overflow:hidden;}
    .stTabs [data-baseweb="tab-list"] {gap:1.2rem; border-bottom:1px solid var(--border);}
    .stTabs [data-baseweb="tab"] {height:2.8rem; padding:0 .15rem; color:var(--muted);}
    .stTabs [aria-selected="true"] {color:var(--primary); font-weight:650;}
    .stButton > button, .stDownloadButton > button {border-radius:.48rem; font-weight:650;
        min-height:2.5rem; border-color:#D1D5DB;}
    .stButton > button[kind="primary"] {background:var(--primary); border-color:var(--primary);}
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-color:#D1D5DB; border-radius:.45rem; background:#FFFFFF;
    }
    [data-testid="stAlert"] {border-radius:.55rem; border:1px solid var(--border);}
    [data-testid="stImage"] {width:100% !important;}
    [data-testid="stImage"] img {
        display:block; width:100% !important; max-width:100% !important; height:auto !important;
    }
    hr {border-color:var(--border); margin:2rem 0 1rem;}
    @media (max-width: 800px) {
        .block-container {padding:1rem .8rem 2rem;}
        .dashboard-title {font-size:1.45rem;}
        .info-grid {grid-template-columns:1fr;}
    }
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


def page_header(title: str, description: str, kicker: str = "Research dashboard") -> None:
    st.markdown(
        f'<div class="page-header"><div class="page-kicker">{kicker}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-description">{description}</div></div>',
        unsafe_allow_html=True,
    )


def show_image(path: Path, caption: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{caption}**")
        if path.is_file():
            st.image(str(path), width="stretch")
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


default_outputs = deployment_outputs or default_output_directory(PROJECT_ROOT)
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">FS</div><div>'
        '<div class="brand-name">Fraud Security Lab</div>'
        '<div class="brand-meta">Adversarial robustness research</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
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
        label_visibility="collapsed",
    )
    with st.expander("Artifact settings"):
        output_input = st.text_input("Saved outputs directory", str(default_outputs))
    st.caption("The dashboard reads saved artifacts only. It does not retrain models.")

paths = artifact_paths(Path(output_input).expanduser())

st.markdown(
    '<div class="dashboard-header"><div class="dashboard-eyebrow">CST-8415 · Security research</div>'
    '<h1 class="dashboard-title">AI-Based Fraud Detection &amp; Adversarial Robustness</h1>'
    '<div class="dashboard-subtitle">An evidence-led view of model performance, explainability, '
    'adversarial exposure, hardening, simulation, and temporal stability.</div></div>',
    unsafe_allow_html=True,
)
show_notice()
if deployment_bootstrap_error:
    st.error(f"Hosted artifact setup failed: {deployment_bootstrap_error}")

if page == "Project Overview":
    page_header(
        "Project Overview",
        "A security-focused machine-learning prototype for detecting fraud, explaining "
        "decisions, evaluating adversarial evasion, and measuring defensive hardening.",
        "System summary",
    )
    st.markdown(
        '<div class="info-grid">'
        '<div class="info-card"><div class="info-label">Dataset</div>'
        '<div class="info-value">Synthetic PaySim transactions</div></div>'
        '<div class="info-card"><div class="info-label">Model family</div>'
        '<div class="info-value">XGBoost fraud detection</div></div>'
        '<div class="info-card"><div class="info-label">Operating mode</div>'
        '<div class="info-value">Offline research prototype</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="workflow"><span class="workflow-step">Fraud Detection</span>'
        '<span class="workflow-arrow">→</span><span class="workflow-step">SHAP</span>'
        '<span class="workflow-arrow">→</span><span class="workflow-step">Adversarial Attack</span>'
        '<span class="workflow-arrow">→</span><span class="workflow-step">Attack Comparison</span>'
        '<span class="workflow-arrow">→</span><span class="workflow-step">Hardening</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("**Completed experiment stages**")
        st.caption(
            "Baseline XGBoost detection · Tree SHAP explainability · constrained ART attacks · "
            "multi-attack comparison · leakage-safe adversarial training"
        )

elif page == "Baseline Performance":
    page_header("Baseline Model Performance", "Saved clean-test metrics from the original XGBoost detector.", "Phase 1")
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
    page_header("SHAP Explainability", "Global and local Tree SHAP evidence from the saved baseline model.", "Phase 2")
    st.caption("SHAP is not recomputed in the dashboard; all plots are saved experiment outputs.")
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
    page_header("Multiple Attack Comparison", "A consistent comparison of constrained adversarial evasion attacks.", "Phase 4")
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
    page_header("Baseline vs Hardened Model", "Clean performance and fresh-attack robustness after leakage-safe adversarial training.", "Phase 5")
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
    page_header("Single Transaction Prediction", "Evaluate one PaySim-style transaction with the saved baseline and hardened models.", "Interactive analysis")
    st.caption("The same saved Phase 1 preprocessor and encoded feature order are applied to both models.")
    try:
        baseline_model, hardened_model, preprocessor, feature_names = cached_prediction_artifacts(
            str(Path(output_input).expanduser())
        )
        transaction_types = available_transaction_types(preprocessor)
        with st.form("transaction_form"):
            st.markdown("**Transaction details**")
            st.caption("Enter valid non-negative PaySim values for the transaction snapshot.")
            left, right = st.columns(2)
            with left:
                st.markdown("##### Transaction & sender")
                transaction_type = st.selectbox("Transaction type", transaction_types)
                step = st.number_input("Step", min_value=1, value=1, step=1)
                amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
                oldbalance_org = st.number_input("Sender old balance", min_value=0.0, value=5000.0)
            with right:
                st.markdown("##### Resulting balances")
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
                    with st.container(border=True):
                        st.markdown(f"### {model_name} model")
                        st.metric("Prediction", result["prediction_label"])
                        st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
                        st.metric("Risk band", result["risk_label"])
            st.caption("Low/Medium/High is a presentation band derived from probability, not a separately trained model.")
    except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError) as error:
        st.error(str(error))
        st.info("Set the sidebar path to the outputs directory containing both models and preprocessing artifacts.")

elif page == "Real-Time Simulation":
    page_header("Real-Time Transaction Simulation", "Replay a small PaySim sequence through the hardened model and monitor running outcomes.", "Phase 7")
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
    st.markdown("#### Simulation controls")
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
    page_header("Concept Drift", "Evaluate hardened-model performance and feature stability across chronological PaySim windows.", "Phase 8")
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
    st.markdown("#### Analysis configuration")
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
