from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medprogress.data import (  # noqa: E402
    IMAGE_COLS,
    TARGET_COL,
    UPLOAD_TEMPLATE_COLUMNS,
    WEARABLE_COLS,
    empty_upload_template,
    prepare_uploaded_cohort,
)
from medprogress.experiment import run_experiment  # noqa: E402

st.set_page_config(
    page_title="Multimodal Disease Progression Lab",
    page_icon="🧬",
    layout="wide",
)

st.title("Multimodal Disease Progression Lab")
st.caption("Research prototype · synthetic demo or uploaded feature-level CSV · not a diagnostic or treatment tool")
st.markdown(
    "Choose the built-in synthetic demo, upload a longitudinal CSV of numerical features, or download a blank CSV template. "
    "**This is a public hosted demo:** do not upload identifiable, confidential, or unauthorised patient data. "
    "Raw medical images and raw wearable sensor files are not accepted."
)

with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "Choose how to start",
        ["Synthetic demo", "Upload CSV", "Empty CSV template"],
        index=0,
        key="data_source_mode",
    )
    uploaded_file = None
    if data_source == "Upload CSV":
        st.warning(
            "Public demo privacy warning: do not upload names, dates of birth, medical record numbers, "
            "identifiable health records, or data you are not authorised to use."
        )
        uploaded_file = st.file_uploader("Upload a completed longitudinal CSV", type=["csv"])

raw_uploaded_frame = None
uploaded_cohort = None
upload_error = None
uploaded_signature = "no-file"
if data_source == "Upload CSV" and uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    uploaded_signature = hashlib.sha256(upload_bytes).hexdigest()
    try:
        raw_uploaded_frame = pd.read_csv(io.BytesIO(upload_bytes))
        uploaded_cohort = prepare_uploaded_cohort(raw_uploaded_frame)
    except Exception as exc:
        upload_error = f"{type(exc).__name__}: {exc}"

if data_source == "Synthetic demo":
    current_input_key = "synthetic-demo"
elif data_source == "Empty CSV template":
    current_input_key = "empty-csv-template"
else:
    current_input_key = f"uploaded-csv:{uploaded_signature}"

if st.session_state.get("_experiment_input_key") != current_input_key:
    st.session_state.pop("experiment_result", None)
    st.session_state.pop("experiment_settings", None)
    st.session_state["_experiment_input_key"] = current_input_key

with st.sidebar:
    st.header("Experiment settings")
    if data_source == "Synthetic demo":
        patient_count = st.slider("Synthetic patients", min_value=120, max_value=800, value=400, step=40)
    else:
        patient_count = 400
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    epochs = st.slider("Neural fusion epochs", min_value=5, max_value=40, value=20, step=5)
    if data_source == "Synthetic demo":
        st.info("Every patient, visit, feature, outcome, and treatment in this demo is simulated.")
    elif data_source == "Upload CSV":
        st.caption("The uploaded cohort is used for the progression models. The causal-effect tab remains a separate synthetic simulation.")
    else:
        st.caption("Download the blank template, fill it with one row per visit, then switch to Upload CSV.")
    can_run = data_source == "Synthetic demo" or (
        data_source == "Upload CSV" and uploaded_cohort is not None
    )
    run_clicked = st.button(
        "Run full experiment",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

def render_blank_template_download(button_label: str = "Download blank CSV template") -> None:
    """Offer a header-only CSV so users can create a correctly named input file."""
    template = empty_upload_template()
    st.download_button(
        button_label,
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="longitudinal_upload_template.csv",
        mime="text/csv",
        key=f"template-download-{button_label}",
    )

if data_source == "Empty CSV template":
    st.subheader("Start with an empty data shell")
    st.write(
        "Download a header-only CSV, enter one row per patient visit, and then choose **Upload CSV**. "
        "The template is intentionally empty; it contains column names but no patient records."
    )
    render_blank_template_download()
    st.markdown(
        "**Required columns:** patient_id, time_months, current_stage. "
        "**Optional numeric columns:** age_years, sex_binary, lab_a, lab_b, "
        "img_0–img_7, and wear_0–wear_5."
    )
    st.markdown(
        "- patient_id: a non-identifying study code, not a person's name or medical record number.\n"
        "- time_months: numeric months since that patient's baseline visit.\n"
        "- current_stage: integer label 0, 1, 2, or 3 at each visit. Map your labels to these codes before upload.\n"
        "- Optional columns may be blank. Imaging and wearable columns are already-extracted numerical features, not raw scans or sensor streams.\n"
        "- Each patient needs at least two visits; the file needs at least 10 unique patients for the split. More data are normally needed for stable research estimates.\n"
        "- sex_binary, if supplied, must use consistent numeric codes 0 and 1; these codes are interpreted only as groups for this demo."
    )
    st.warning(
        "Privacy: this public Render app receives the uploaded file. Do not upload identifiable or confidential health data. "
        "For authorised sensitive datasets, run the project locally inside an approved secure environment."
    )
    st.stop()

if data_source == "Upload CSV":
    if uploaded_file is None:
        st.info("Upload a completed CSV, or download the blank template below to prepare your data first.")
        render_blank_template_download()
        st.stop()
    if upload_error is not None:
        st.error(f"CSV validation failed: {upload_error}")
        st.write("Check the required columns and value formats, then upload the corrected file.")
        render_blank_template_download()
        st.stop()

    n_uploaded_patients = int(uploaded_cohort["patient_id"].nunique())
    st.success(
        f"CSV format checks passed: {n_uploaded_patients} patients and {len(uploaded_cohort)} visit rows."
    )
    original_columns = {str(column).strip() for column in raw_uploaded_frame.columns}
    ignored_columns = sorted(original_columns - set(UPLOAD_TEMPLATE_COLUMNS))
    if ignored_columns:
        st.warning(
            "These extra columns are not used by this model and will be ignored: "
            + ", ".join('"' + column + '"' for column in ignored_columns)
        )
    image_values = int(uploaded_cohort[IMAGE_COLS].notna().sum().sum())
    wearable_values = int(uploaded_cohort[WEARABLE_COLS].notna().sum().sum())
    if image_values == 0:
        st.warning("No numerical imaging-related feature values were found. The imaging feature group will contain no information.")
    if wearable_values == 0:
        st.warning("No numerical wearable-style feature values were found. The wearable feature group will contain no information.")
    if n_uploaded_patients < 30:
        st.warning("This is a small cohort. Metrics may be unstable and should be treated as exploratory.")
    st.info(
        "Format validation checks the table structure, not whether a dataset is clinically appropriate or scientifically valid. "
        "Uploaded tabular data are used for progression, missingness, and subgroup analyses; the causal-effect tab uses a separate synthetic dataset."
    )

if run_clicked:
    with st.spinner(
        "Generating synthetic visits, fitting models, and evaluating held-out patients…"
        if data_source == "Synthetic demo"
        else "Preparing uploaded visits, fitting models, and evaluating held-out patients…"
    ):
        try:
            if data_source == "Synthetic demo":
                experiment_result = run_experiment(
                    n_patients=int(patient_count), seed=int(seed), epochs=int(epochs)
                )
            else:
                experiment_result = run_experiment(
                    seed=int(seed), epochs=int(epochs), cohort=uploaded_cohort
                )
            st.session_state["experiment_result"] = experiment_result
            st.session_state["experiment_settings"] = {
                "data_mode": "Synthetic demo" if data_source == "Synthetic demo" else "Uploaded CSV",
                "patient_count": int(experiment_result["cohort"]["patient_id"].nunique()),
                "seed": int(seed),
                "neural_epochs": int(epochs),
            }
        except Exception as exc:
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
            st.exception(exc)

result = st.session_state.get("experiment_result")
if result is None:
    if data_source == "Upload CSV" and uploaded_cohort is not None:
        st.subheader("Uploaded dataset preview")
        st.write(
            "The CSV passed basic schema checks. Click **Run full experiment** in the sidebar to train the models. "
            "The validator does not assess clinical suitability or approve a research analysis."
        )
        st.dataframe(uploaded_cohort.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Choose settings in the sidebar and select **Run full experiment** to train models and create results.")
        st.subheader("What this demo contains")
        cards = st.columns(3)
        cards[0].metric("Modalities", "3", "Imaging-related, wearable-style, clinical-style")
        cards[1].metric("Stage labels", "4", "User-mapped labels: 0–3")
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
c1.metric("Synthetic patients" if result.get("data_mode") == "synthetic" else "Uploaded patients", f"{len(result['cohort']['patient_id'].unique()):,}")
c2.metric("Forecast examples", f"{len(result['examples']):,}")
c3.metric("Held-out test windows", f"{len(splits.test):,}")
c4.metric("Neural fusion macro-F1", f"{result['neural_metrics']['macro_f1']:.3f}")
st.caption(
    f"Data mode: {settings.get('data_mode', 'Synthetic demo')} · Seed {settings.get('seed', seed)} · "
    f"{settings.get('neural_epochs', epochs)} neural epochs · patient-level split "
    "(60% train / 20% validation / 20% test)."
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
        "This tab uses a separate synthetic dataset of simulated individuals. Treatment is assigned depending on "
        "simulated baseline risk and age, creating confounding. The outcome includes a known treatment effect. "
        "These estimates are not calculated from the uploaded progression CSV."
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
        "Subgroup metrics are descriptive and are not, by themselves, evidence of fairness. "
        "For uploaded data, age groups use the training-set median and code groups use the supplied sex_binary values. "
        "Missing demographic fields can produce empty groups. Subgroup counts, uncertainty intervals, clinically appropriate "
        "metrics, and external validation would be required for real research."
    )

with data_tab:
    st.subheader("Synthetic cohort preview" if result.get("data_mode") == "synthetic" else "Uploaded cohort preview")
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
            "data_mode": result.get("data_mode", "synthetic"),
            "disclaimer": (
                "Progression results use synthetic data; not medical evidence."
                if result.get("data_mode") == "synthetic"
                else "Progression results use the uploaded CSV; the separate causal-effect tab uses a synthetic simulation. "
                     "Neither establishes clinical validity."
            ),
        }, indent=2, default=float).encode("utf-8"),
        file_name="experiment_summary.json",
        mime="application/json",
    )
