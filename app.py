from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medprogress.data import TARGET_COL  # noqa: E402
from medprogress.experiment import run_experiment  # noqa: E402

st.set_page_config(
    page_title="Multimodal Disease Progression Lab",
    page_icon="🧬",
    layout="wide",
)

st.title("Multimodal Disease Progression Lab")
st.caption("Research prototype · synthetic data only · not a diagnostic or treatment tool")
st.markdown(
    "Explore a reproducible machine-learning workflow for longitudinal health data: "
    "modality comparison, neural fusion, irregular follow-up, missing-modality stress tests, "
    "synthetic causal estimation, and subgroup evaluation."
)

with st.sidebar:
    st.header("Experiment settings")
    patient_count = st.slider("Synthetic patients", min_value=120, max_value=800, value=400, step=40)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    epochs = st.slider("Neural fusion epochs", min_value=5, max_value=40, value=20, step=5)
    st.warning("Every patient, visit, feature, outcome, and treatment in this demo is simulated.")
    run_clicked = st.button("Run full experiment", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Generating synthetic visits, fitting models, and evaluating held-out patients…"):
        try:
            st.session_state["experiment_result"] = run_experiment(
                n_patients=int(patient_count), seed=int(seed), epochs=int(epochs)
            )
            st.session_state["experiment_settings"] = {
                "synthetic_patients": int(patient_count),
                "seed": int(seed),
                "neural_epochs": int(epochs),
            }
        except Exception as exc:
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
            st.exception(exc)

result = st.session_state.get("experiment_result")
if result is None:
    st.info("Choose settings in the sidebar and select **Run full experiment** to train models and create results.")
    st.subheader("What this demo contains")
    cards = st.columns(3)
    cards[0].metric("Modalities", "3", "Image-derived, wearable, clinical")
    cards[1].metric("Stage labels", "4", "Invented labels: 0–3")
    cards[2].metric("Research themes", "5", "Progression, fusion, missingness, causal, fairness")
    st.markdown(
        "**Interpretation rule:** synthetic performance is only a software and methodology demonstration. "
        "It is not medical evidence, and results should not be presented as clinical performance."
    )
    st.stop()

splits = result["splits"]
benchmarks = result["benchmarks"]
settings = st.session_state.get("experiment_settings", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Synthetic patients", f"{len(result['cohort']['patient_id'].unique()):,}")
c2.metric("Forecast examples", f"{len(result['examples']):,}")
c3.metric("Held-out test windows", f"{len(splits.test):,}")
c4.metric("Neural fusion macro-F1", f"{result['neural_metrics']['macro_f1']:.3f}")
st.caption(
    f"Seed {settings.get('seed', seed)} · {settings.get('neural_epochs', epochs)} epochs · "
    "patient-level split (60% train / 20% validation / 20% test)."
)

progression_tab, modalities_tab, missing_tab, causal_tab, fairness_tab, data_tab = st.tabs([
    "Progression models",
    "Modality comparison",
    "Missing data",
    "Causal simulation",
    "Subgroup evaluation",
    "Data and exports",
])

with progression_tab:
    st.subheader("Next-observed-visit stage classification")
    st.write(
        "Models use current-visit measurements to predict the stage at the patient's next observed visit. "
        "Visit gaps are irregular, so this is not a fixed-horizon forecast."
    )
    fusion_rows = benchmarks[
        ((benchmarks["feature_group"] == "Multimodal fusion") &
         (benchmarks["model"].isin(["Random Forest", "Logistic Regression", "Neural multimodal fusion"])))
        | benchmarks["model"].str.contains("baseline", case=False, na=False)
    ].copy()
    st.dataframe(fusion_rows, use_container_width=True, hide_index=True)
    fig = px.bar(
        fusion_rows,
        x="model",
        y="macro_f1",
        color="feature_group",
        barmode="group",
        range_y=[0, 1],
        title="Held-out macro-F1 (higher is better; compare the same target)"
    )
    st.plotly_chart(fig, use_container_width=True)
    history = result["training_history"]
    if not history.empty:
        hfig = go.Figure()
        hfig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], mode="lines", name="Training loss"))
        hfig.add_trace(go.Scatter(x=history["epoch"], y=history["validation_macro_f1"], mode="lines", name="Validation macro-F1", yaxis="y2"))
        hfig.update_layout(
            title="Neural fusion training history",
            xaxis_title="Epoch",
            yaxis=dict(title="Training loss"),
            yaxis2=dict(title="Validation macro-F1", overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(hfig, use_container_width=True)
    st.caption("The neural model checkpoint is selected using validation macro-F1. Test labels are not used for training or checkpoint selection.")

with modalities_tab:
    st.subheader("What does each modality contribute?")
    classic = benchmarks[benchmarks["model"].isin(["Random Forest", "Logistic Regression"])].copy()
    fig = px.bar(
        classic,
        x="feature_group",
        y="macro_f1",
        color="model",
        barmode="group",
        range_y=[0, 1],
        title="Unimodal versus fused feature groups on the same held-out patients",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(classic, use_container_width=True, hide_index=True)
    st.info("This is an ablation-style comparison, not proof of clinical benefit. Random variation and synthetic data design can strongly affect the ordering.")

with missing_tab:
    st.subheader("Additional missing-modality stress test")
    miss = result["missingness"].copy()
    st.write(
        "A random forest is trained on the original training split. At evaluation time, additional whole-modality "
        "dropout is injected into test features and modality-missing flags are updated. The model is not retrained "
        "for each dropout rate."
    )
    st.dataframe(miss, use_container_width=True, hide_index=True)
    fig = px.line(
        miss,
        x="additional_test_time_modality_dropout",
        y="macro_f1",
        markers=True,
        range_y=[0, 1],
        title="Test macro-F1 under added inference-time modality dropout",
    )
    fig.update_xaxes(title="Probability of additionally masking each modality per test example")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This controlled perturbation is a stress test; it does not establish robustness to real-world missing-not-at-random data.")

with causal_tab:
    st.subheader("Can an estimator recover a known simulated treatment effect?")
    causal = result["causal"]
    st.write(
        "Treatment is assigned depending on simulated baseline risk and age, creating confounding. "
        "The outcome includes a known treatment effect. Estimates below are computed on synthetic data only."
    )
    causaltable = pd.DataFrame([
        {"method": "Known simulated effect", "estimated effect": causal["known_simulated_effect"]},
        {"method": "Unadjusted treated-minus-control", "estimated effect": causal["naive_difference"]},
        {"method": "Adjusted linear regression", "estimated effect": causal["adjusted_regression"]},
        {"method": "Inverse-probability weighting", "estimated effect": causal["inverse_probability_weighting"]},
    ])
    st.dataframe(causaltable, use_container_width=True, hide_index=True)
    fig = px.bar(causaltable, x="method", y="estimated effect", title="Synthetic causal-effect estimates (outcome units)")
    fig.add_hline(y=causal["known_simulated_effect"], line_dash="dash", annotation_text="Known simulated effect")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Simulated sample size: {causal['n']:,}. Estimated propensity-score range: "
        f"{causal['propensity_min']:.3f}–{causal['propensity_max']:.3f}. "
        "This exercise checks estimator behaviour where the data-generating assumptions are known; it does not establish causal identification in real clinical data."
    )

with fairness_tab:
    st.subheader("Descriptive subgroup evaluation")
    st.dataframe(result["fairness"], use_container_width=True, hide_index=True)
    st.warning(
        "These are simulated subgroups. Differences do not establish real-world bias or fairness. "
        "Subgroup counts, uncertainty intervals, clinically appropriate metrics, and external validation would be required for real research."
    )

with data_tab:
    st.subheader("Synthetic cohort preview")
    st.dataframe(result["cohort"].head(20), use_container_width=True, hide_index=True)
    st.subheader("Forecast examples preview")
    st.dataframe(result["examples"].head(20), use_container_width=True, hide_index=True)
    st.write("**Split sizes (forecast windows)**")
    split_sizes = pd.DataFrame([
        {"split": name, "patients": int(frame["patient_id"].nunique()), "forecast windows": len(frame)}
        for name, frame in [("Train", splits.train), ("Validation", splits.validation), ("Test", splits.test)]
    ])
    st.dataframe(split_sizes, use_container_width=True, hide_index=True)
    st.download_button(
        "Download benchmark metrics (CSV)",
        data=benchmarks.to_csv(index=False).encode("utf-8"),
        file_name="benchmark_metrics.csv",
        mime="text/csv",
    )
    prediction_export = splits.test[
        ["patient_id", "time_months", "next_interval_months", "current_stage", "target_stage"]
    ].copy()
    prediction_export["random_forest_fused_prediction"] = result["random_forest_predictions"]
    prediction_export["neural_fusion_prediction"] = result["neural_predictions"]
    st.download_button(
        "Download held-out predictions (CSV)",
        data=prediction_export.to_csv(index=False).encode("utf-8"),
        file_name="held_out_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download experiment summary (JSON)",
        data=json.dumps({
            "settings": settings,
            "software_versions": result["software_versions"],
            "split_sizes": split_sizes.to_dict(orient="records"),
            "benchmarks": benchmarks.to_dict(orient="records"),
            "neural_metrics": {
                "n_test_examples": result["neural_metrics"]["n_test_examples"],
                "accuracy": result["neural_metrics"]["accuracy"],
                "macro_f1": result["neural_metrics"]["macro_f1"],
            },
            "causal_simulation": result["causal"],
            "disclaimer": "Entirely synthetic research demonstration; not medical evidence.",
        }, indent=2, default=float).encode("utf-8"),
        file_name="experiment_summary.json",
        mime="application/json",
    )
