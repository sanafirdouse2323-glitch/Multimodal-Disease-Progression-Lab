from __future__ import annotations

import hashlib
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
    read_uploaded_csv,
)
from medprogress.experiment import run_experiment  # noqa: E402
from medprogress.simple_input import (  # noqa: E402
    PERSON_COLUMN,
    STAGE_COLUMNS,
    make_simple_stage_table,
    simple_table_to_cohort,
)

st.set_page_config(
    page_title="Multimodal Disease Progression Lab",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
      h1 { text-align: center; letter-spacing: -0.035em; }
      .intro-copy { max-width: 850px; margin: 0 auto 1rem auto; text-align: center; }
      .how-link { text-align: center; margin: 0.2rem 0 1rem 0; font-size: 1rem; }
      .how-link a { font-weight: 650; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multimodal Disease Progression Lab")
st.markdown(
    """
    <div class="intro-copy">
      <p><strong>Explore how a few simple numbers can describe changes from one visit to the next.</strong></p>
      <p>You can try the ready-made example, enter whole numbers yourself, or upload a spreadsheet.
      You do not need a research background to try the sample.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "This is an educational software demonstration, not a medical tool. "
    "The sample numbers are artificial. Do not enter names, medical record numbers, or confidential health information."
)

with st.container(border=True):
    st.markdown("<h3 style='text-align:center; margin-bottom:0.2rem;'>Choose your data source</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>Start with sample data, enter a few whole numbers, or use an existing spreadsheet.</p>",
        unsafe_allow_html=True,
    )
    source_choice = st.radio(
        "Data sources",
        ["Try sample data", "Enter simple numbers", "Upload a spreadsheet"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="data_source_mode",
    )

st.markdown(
    '<p class="how-link"><a href="#how-it-works">How it works ↓</a></p>',
    unsafe_allow_html=True,
)

source_to_mode = {
    "Try sample data": "Synthetic demo",
    "Enter simple numbers": "Simple numbers",
    "Upload a spreadsheet": "Upload CSV",
}
data_source = source_to_mode[source_choice]

with st.sidebar:
    st.header("Optional settings")
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    epochs = st.slider("Model training rounds", min_value=5, max_value=40, value=20, step=5)
    patient_count = 400
    synthetic_scenario = "progression_signal"
    if data_source == "Synthetic demo":
        patient_count = st.slider(
            "Number of sample people",
            min_value=120,
            max_value=800,
            value=400,
            step=40,
            help="This changes the size of the artificial sample, not real patient data.",
        )
        scenario_labels = {
            "Example with a pattern (recommended)": "progression_signal",
            "Example without an extra pattern": "no_signal",
            "More difficult example": "difficult",
        }
        selected_scenario_label = st.selectbox(
            "Sample data pattern",
            options=list(scenario_labels),
            index=0,
            help="These are artificial test settings, not medical facts.",
        )
        synthetic_scenario = scenario_labels[selected_scenario_label]

raw_uploaded_frame = None
uploaded_cohort = None
simple_cohort = None
uploaded_file = None
uploaded_simple = False
upload_error = None
simple_error = None
simple_table = None
uploaded_signature = "no-file"

if data_source == "Synthetic demo":
    st.subheader("Ready-to-use sample")
    st.markdown(
        "I have prepared an artificial example for you. No file or personal information is needed. "
        "Use the default settings to see how the dashboard works."
    )
    st.success("Sample data are ready. These are generated example records, not real people.")

elif data_source == "Simple numbers":
    st.subheader("Enter a few whole numbers")
    st.markdown(
        "Each row is one person. Enter the stage number recorded at month 0, month 6, and month 12. "
        "The example is already filled in, so you can run it first or change the numbers."
    )
    st.caption(
        "Use whole numbers only—no decimal points. Stage numbers must be 0, 1, 2, or 3. "
        "They are category labels for this demonstration, not diagnoses or a universal measure of health."
    )
    simple_table = st.data_editor(
        make_simple_stage_table(),
        key="simple_number_table",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            PERSON_COLUMN: st.column_config.NumberColumn(
                "Person number",
                help="Use a different positive whole number for each person.",
                min_value=1,
                max_value=9999,
                step=1,
                format="%d",
            ),
            STAGE_COLUMNS[0]: st.column_config.NumberColumn(
                "Stage at month 0",
                min_value=0,
                max_value=3,
                step=1,
                format="%d",
            ),
            STAGE_COLUMNS[1]: st.column_config.NumberColumn(
                "Stage at month 6",
                min_value=0,
                max_value=3,
                step=1,
                format="%d",
            ),
            STAGE_COLUMNS[2]: st.column_config.NumberColumn(
                "Stage at month 12",
                min_value=0,
                max_value=3,
                step=1,
                format="%d",
            ),
        },
    )
    try:
        simple_cohort = simple_table_to_cohort(simple_table)
        st.success(
            f"Your table is ready: {simple_cohort['patient_id'].nunique()} people and "
            f"{len(simple_cohort)} visit records."
        )
    except ValueError as exc:
        simple_error = str(exc)
        st.warning(simple_error)

    st.info(
        "This quick-entry option uses only visit times and stage numbers. It does not include scan measurements "
        "or wearable-device measurements. For a full multimodal comparison, use the spreadsheet option with "
        "numerical features that have already been prepared."
    )
    blank_number_sheet = pd.DataFrame({
        PERSON_COLUMN: list(range(1, 11)),
        STAGE_COLUMNS[0]: [pd.NA] * 10,
        STAGE_COLUMNS[1]: [pd.NA] * 10,
        STAGE_COLUMNS[2]: [pd.NA] * 10,
    })
    st.download_button(
        "Download a blank number sheet",
        data=blank_number_sheet.to_csv(index=False).encode("utf-8"),
        file_name="simple_stage_numbers_template.csv",
        mime="text/csv",
        help="A simple four-column sheet with one row per person.",
    )

elif data_source == "Upload CSV":
    st.subheader("Upload an existing spreadsheet")
    st.markdown(
        "You can upload the simple four-column number sheet from the previous option, "
        "or a detailed visit-by-visit CSV. The detailed format uses patient_id, time_months, "
        "and current_stage, with optional numerical feature columns."
    )
    st.warning(
        "This public website receives the file you upload. Do not upload names, dates of birth, "
        "medical record numbers, identifiable health records, confidential information, or data "
        "you are not authorised to use."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        max_upload_size=20,
        help="Maximum file size: 20 MB.",
    )
    st.download_button(
        "Download the detailed CSV template",
        data=empty_upload_template().to_csv(index=False).encode("utf-8"),
        file_name="longitudinal_upload_template.csv",
        mime="text/csv",
        help="This template is for users who already have numerical feature columns.",
    )
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        uploaded_signature = hashlib.sha256(upload_bytes).hexdigest()
        try:
            raw_uploaded_frame = read_uploaded_csv(upload_bytes)
            is_simple_number_sheet = all(
                column in raw_uploaded_frame.columns
                for column in (PERSON_COLUMN, *STAGE_COLUMNS)
            )
            if is_simple_number_sheet:
                uploaded_cohort = simple_table_to_cohort(raw_uploaded_frame)
                uploaded_simple = True
            else:
                uploaded_cohort = prepare_uploaded_cohort(raw_uploaded_frame)
            n_uploaded_patients = int(uploaded_cohort["patient_id"].nunique())
            st.success(
                f"File checks passed: {n_uploaded_patients} people and {len(uploaded_cohort)} visit records."
            )
            if uploaded_simple:
                st.info(
                    "I recognised the simple number sheet. This run uses visit times and stage numbers only; "
                    "it does not include scan or wearable-device measurements."
                )
            original_columns = {str(column).strip() for column in raw_uploaded_frame.columns}
            recognised_columns = (
                {PERSON_COLUMN, *STAGE_COLUMNS}
                if uploaded_simple
                else set(UPLOAD_TEMPLATE_COLUMNS)
            )
            ignored_columns = sorted(original_columns - recognised_columns)
            if ignored_columns:
                st.warning(
                    "These extra columns are not used and will be ignored: "
                    + ", ".join('"' + column + '"' for column in ignored_columns)
                )
            image_values = int(uploaded_cohort[IMAGE_COLS].notna().sum().sum())
            wearable_values = int(uploaded_cohort[WEARABLE_COLS].notna().sum().sum())
            if not uploaded_simple and image_values == 0:
                st.info("No imaging-related numbers were supplied. The image feature group will be empty.")
            if not uploaded_simple and wearable_values == 0:
                st.info("No wearable-related numbers were supplied. The wearable feature group will be empty.")
            if n_uploaded_patients < 30:
                st.warning("This is a small sample. Results may be unstable and should be treated as exploratory.")
            st.caption(
                "The file checks confirm its structure only. They do not confirm medical suitability or scientific validity."
            )
        except Exception as exc:
            upload_error = f"{type(exc).__name__}: {exc}"
            st.error(f"I could not read this spreadsheet: {upload_error}")
            st.caption("Check the required column names and number formats, then try again.")

# A stable input signature prevents results from an older source or edited table being shown.
if data_source == "Synthetic demo":
    current_input_key = f"synthetic-demo:{synthetic_scenario}"
elif data_source == "Simple numbers":
    table_text = simple_table.to_csv(index=False) if simple_table is not None else "no-table"
    simple_signature = hashlib.sha256(table_text.encode("utf-8")).hexdigest()
    current_input_key = f"simple-numbers:{simple_signature}"
else:
    current_input_key = f"uploaded-csv:{uploaded_signature}"

if st.session_state.get("_experiment_input_key") != current_input_key:
    st.session_state.pop("experiment_result", None)
    st.session_state.pop("experiment_settings", None)
    st.session_state["_experiment_input_key"] = current_input_key

st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
with st.expander("How it works", expanded=False):
    st.markdown(
        """
        1. **Choose a data source.** The sample is ready to run. You can also enter whole numbers or upload a CSV.
        2. **Add visit information.** In the simple table, each person has three stage numbers: month 0, month 6, and month 12.
        3. **Run the example.** The program uses one visit to estimate the stage number at that person's next recorded visit.
        4. **Read the results.** The tables and charts compare model outputs. They are learning examples, not medical advice.
        """
    )
    st.markdown(
        "[Project overview](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab) · "
        "[Methods and limits](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab/blob/main/METHODS_AND_LIMITATIONS.md) · "
        "[Detailed CSV guide](https://github.com/sanafirdouse2323-glitch/Multimodal-Disease-Progression-Lab/blob/main/UPLOAD_DATA_GUIDE.md)"
    )

if data_source == "Upload CSV" and uploaded_file is None:
    st.info("Choose a CSV file above, or start with Enter simple numbers if you do not have a spreadsheet.")

run_cohort = simple_cohort if data_source == "Simple numbers" else uploaded_cohort
can_run = data_source == "Synthetic demo" or run_cohort is not None
button_labels = {
    "Synthetic demo": "Run sample experiment",
    "Simple numbers": "Run with my numbers",
    "Upload CSV": "Run on my spreadsheet",
}
run_clicked = st.button(
    button_labels[data_source],
    type="primary",
    use_container_width=True,
    disabled=not can_run,
)

if run_clicked:
    with st.spinner(
        "Preparing the sample and running the models…"
        if data_source == "Synthetic demo"
        else "Checking the numbers and running the models…"
    ):
        try:
            if data_source == "Synthetic demo":
                experiment_result = run_experiment(
                    n_patients=int(patient_count),
                    seed=int(seed),
                    epochs=int(epochs),
                    scenario=synthetic_scenario,
                )
            else:
                experiment_result = run_experiment(
                    seed=int(seed),
                    epochs=int(epochs),
                    cohort=run_cohort,
                )
                if data_source == "Simple numbers" or uploaded_simple:
                    experiment_result["data_mode"] = "simple_numbers"
            st.session_state["experiment_result"] = experiment_result
            st.session_state["experiment_settings"] = {
                "data_mode": (
                    "Sample data"
                    if data_source == "Synthetic demo"
                    else "Numbers in the simple sheet"
                    if data_source == "Simple numbers" or uploaded_simple
                    else "Uploaded spreadsheet"
                ),
                "patient_count": int(experiment_result["cohort"]["patient_id"].nunique()),
                "seed": int(seed),
                "neural_epochs": int(epochs),
                "synthetic_scenario": experiment_result.get("synthetic_scenario"),
            }
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The experiment could not finish: {type(exc).__name__}: {exc}")
            st.exception(exc)

result = st.session_state.get("experiment_result")
if result is None:
    if data_source == "Simple numbers" and simple_table is not None:
        st.subheader("Your number table")
        st.caption("The table above is the input. The program turns each person into three visit records.")
    elif data_source == "Upload CSV" and uploaded_cohort is not None:
        st.subheader("Spreadsheet preview")
        st.dataframe(uploaded_cohort.head(10), use_container_width=True, hide_index=True)
    else:
        st.markdown("### Ready when you are")
        st.write("Choose a data source above, then use the button to run the example.")
    st.stop()

splits = result["splits"]


benchmarks = result["benchmarks"]
settings = st.session_state.get("experiment_settings", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("People in this example", f"{len(result['cohort']['patient_id'].unique()):,}")
c2.metric("Visit-to-visit examples", f"{len(result['examples']):,}")
c3.metric("Examples checked", f"{len(splits.test):,}")
c4.metric("Neural model score", f"{result['neural_metrics']['macro_f1'] * 100:.0f}%")
scenario_caption = (
    f" · Sample pattern: {settings.get('synthetic_scenario')}"
    if result.get("data_mode") == "synthetic" else ""
)
st.caption(
    f"Data source: {settings.get('data_mode', 'Sample data')}{scenario_caption}. "
    "The results are educational examples, not medical predictions."
)

progression_tab, modalities_tab, missing_tab, causal_tab, fairness_tab, data_tab = st.tabs([
    "Prediction results",
    "Data groups",
    "Missing information",
    "Practice simulation",
    "Group summaries",
    "My data and downloads",
])

with progression_tab:
    st.subheader("What did the model predict for the next visit?")
    if result.get("data_mode") == "simple_numbers":
        st.write(
            "I use the stage number at one visit to estimate the number recorded at that person's next visit. "
            "This quick example uses only the numbers entered in the table; it does not contain scan or wearable measurements."
        )
    else:
        st.write(
            "I use information from one visit to estimate the stage number recorded at the next observed visit. "
            "The time between visits can vary."
        )
    fusion_rows = benchmarks[
        ((benchmarks["feature_group"] == "Multimodal fusion") &
         (benchmarks["model"].isin(["Random Forest", "Logistic Regression", "Neural multimodal fusion"])))
        | benchmarks["model"].str.contains("baseline", case=False, na=False)
    ].copy()
    display_rows = fusion_rows.copy()
    if result.get("data_mode") == "simple_numbers":
        display_rows["model"] = display_rows["model"].replace({
            "Neural multimodal fusion": "Neural model",
        })
        display_rows["feature_group"] = display_rows["feature_group"].replace({
            "Multimodal fusion": "Visit and stage numbers only",
        })
    display_rows = display_rows.rename(columns={
        "model": "Method",
        "feature_group": "Data used",
        "accuracy": "Correct predictions",
        "macro_f1": "Balanced score",
        "n_test_examples": "Examples checked",
    })
    display_rows["Correct predictions"] = (display_rows["Correct predictions"] * 100).round(0).astype(int).astype(str) + "%"
    display_rows["Balanced score"] = (display_rows["Balanced score"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    st.caption("Scores are shown as whole percentages, rounded to the nearest percent.")
    chart_rows = fusion_rows.copy()
    chart_rows["balanced_score_percent"] = chart_rows["macro_f1"] * 100
    if result.get("data_mode") == "simple_numbers":
        chart_rows["model"] = chart_rows["model"].replace({
            "Neural multimodal fusion": "Neural model",
        })
        chart_rows["feature_group"] = chart_rows["feature_group"].replace({
            "Multimodal fusion": "Visit and stage numbers only",
        })
    fig = px.bar(
        chart_rows,
        x="model",
        y="balanced_score_percent",
        color="feature_group",
        barmode="group",
        range_y=[0, 100],
        title="Balanced model score (higher is better)",
    )
    fig.update_yaxes(title="Balanced score", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    actual_stage = splits.test["target_stage"].astype(int).to_numpy()
    current_stage = splits.test["current_stage"].astype(int).to_numpy()
    actual_change = actual_stage != current_stage
    n_actual_changes = int(actual_change.sum())
    if result.get("data_mode") == "simple_numbers":
        transition_predictions = {
            "Repeat current stage": current_stage,
            "Random Forest": result["random_forest_predictions"],
            "Logistic Regression": result["logistic_regression_predictions"],
            "Neural model": result["neural_predictions"],
        }
    else:
        transition_predictions = {
            "Persistence baseline": current_stage,
            "Random Forest (combined data)": result["random_forest_predictions"],
            "Logistic Regression (combined data)": result["logistic_regression_predictions"],
            "Neural multimodal fusion": result["neural_predictions"],
        }
    transition_rows = []
    for model_name, raw_prediction in transition_predictions.items():
        prediction = pd.Series(raw_prediction).to_numpy(dtype=int)
        predicted_change = prediction != current_stage
        true_positive = int((actual_change & predicted_change).sum())
        false_positive = int((~actual_change & predicted_change).sum())
        false_negative = int((actual_change & ~predicted_change).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        change_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact_changed_accuracy = (
            float((prediction[actual_change] == actual_stage[actual_change]).mean())
            if n_actual_changes else float("nan")
        )
        transition_rows.append({
            "model": model_name,
            "actual stage-change windows": n_actual_changes,
            "exact next-stage accuracy on changed windows": exact_changed_accuracy,
            "change-detection precision": precision,
            "change-detection recall": recall,
            "change-detection F1": change_f1,
        })
    st.subheader("Can the model notice when a stage number changes?")
    st.caption(
        "This table focuses on visits where the stage number changed, rather than relying only on overall correct predictions. "
        "The results are examples from the selected data, not clinical validation."
    )
    transition_display = pd.DataFrame(transition_rows).rename(columns={
        "model": "Method",
        "actual stage-change windows": "Visits where stage changed",
        "exact next-stage accuracy on changed windows": "Exact next-stage accuracy",
        "change-detection precision": "Change detection precision",
        "change-detection recall": "Change detection recall",
        "change-detection F1": "Change detection score",
    })
    for column in (
        "Exact next-stage accuracy",
        "Change detection precision",
        "Change detection recall",
        "Change detection score",
    ):
        transition_display[column] = (transition_display[column] * 100).round(0).astype("Int64").astype(str) + "%"
    st.dataframe(transition_display, use_container_width=True, hide_index=True)
    history = result["training_history"]
    if not history.empty and result.get("data_mode") != "simple_numbers":
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
    if result.get("data_mode") != "simple_numbers":
        st.caption("I select the neural model using validation data. I do not use test labels for training or selection.")

with modalities_tab:
    st.subheader("Which kinds of data were included?")
    if result.get("data_mode") == "simple_numbers":
        st.info(
            "The simple-number option contains visit times and stage numbers only. "
            "It does not include scan measurements or wearable-device measurements, so a comparison between those data groups is not available here. "
            "Use the spreadsheet option with already-prepared numerical features for a multimodal comparison."
        )
    else:
        classic = benchmarks[benchmarks["model"].isin(["Random Forest", "Logistic Regression"])].copy()
        fig = px.bar(
            classic,
            x="feature_group",
            y="macro_f1",
            color="model",
            barmode="group",
            range_y=[0, 1],
            title="Comparison of individual and combined data groups",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(classic, use_container_width=True, hide_index=True)
        st.info("This comparison shows how the selected feature groups behaved in this run. It is not proof of clinical benefit.")

with missing_tab:
    st.subheader("What happens when some information is missing?")
    if result.get("data_mode") == "simple_numbers":
        st.info(
            "The simple-number example does not include scan or wearable measurements, so this extra missing-information test does not apply. "
            "It is available when a dataset includes the corresponding numerical feature groups."
        )
    else:
        miss = result["missingness"].copy()
        st.write(
            "I train a Random Forest on the original training data. I then hide additional image or wearable feature values during testing "
            "to see how the fixed model responds. I do not retrain it for each setting."
        )
        st.dataframe(miss, use_container_width=True, hide_index=True)
        fig = px.line(
            miss,
            x="additional_test_time_modality_dropout",
            y="macro_f1",
            markers=True,
            range_y=[0, 1],
            title="Model score as more feature information is hidden",
        )
        fig.update_xaxes(title="Chance of additionally hiding one data group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This is a controlled software test, not proof that a model is robust to missing real-world information.")

with causal_tab:
    st.subheader("Separate practice example: estimating a known effect")
    if result.get("data_mode") == "simple_numbers":
        st.info(
            "This is an optional research-methods exercise using a separate artificial dataset. "
            "It is not based on the simple visit numbers you entered, so I have kept its technical calculations out of this beginner-friendly view."
        )
    else:
        causal = result["causal"]
        st.write(
            "I use a separate artificial dataset in this tab. The example includes a known effect so I can compare different estimates. "
            "I do not calculate these estimates from the visit numbers or spreadsheet you supplied."
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
            "I use this exercise to check estimator behaviour when the data-generating assumptions are known. It does not establish causal identification in real clinical data."
        )

with fairness_tab:
    st.subheader("How did results differ between groups?")
    if result.get("data_mode") == "simple_numbers":
        st.info(
            "The simple number sheet does not include age or group information, so this comparison is not available for this example."
        )
    else:
        st.dataframe(result["fairness"], use_container_width=True, hide_index=True)
        st.warning(
            "I treat these group summaries as descriptive, not as evidence of fairness. "
            "They require suitable group definitions, enough examples, uncertainty estimates, and external validation before research conclusions."
        )

with data_tab:
    st.subheader("Your visit information")
    if result.get("data_mode") == "simple_numbers":
        st.dataframe(
            result["cohort"][["patient_id", "time_months", "current_stage"]].head(30),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(result["cohort"].head(20), use_container_width=True, hide_index=True)
    st.subheader("Visit-to-visit examples")
    if result.get("data_mode") == "simple_numbers":
        example_columns = ["patient_id", "time_months", "next_interval_months", "current_stage", "target_stage"]
        examples_display = result["examples"][example_columns].rename(columns={
            "patient_id": "Person number",
            "time_months": "Current visit month",
            "next_interval_months": "Months to next visit",
            "current_stage": "Current stage number",
            "target_stage": "Next stage number",
        })
        st.dataframe(examples_display.head(20), use_container_width=True, hide_index=True)
    else:
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
                "Progression results use artificial sample data; not medical evidence."
                if result.get("data_mode") == "synthetic"
                else (
                    "Progression results use simple stage numbers entered in the dashboard; no scan or wearable measurements were supplied. "
                    "The separate practice simulation uses its own synthetic data. Neither establishes clinical validity."
                    if result.get("data_mode") == "simple_numbers"
                    else "Progression results use the uploaded CSV; the separate practice simulation uses synthetic data. "
                         "Neither establishes clinical validity."
                )
            ),
        }, indent=2, default=float).encode("utf-8"),
        file_name="experiment_summary.json",
        mime="application/json",
    )
