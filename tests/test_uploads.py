import numpy as np
import pandas as pd
import pytest

from medprogress.data import (
    IMAGE_COLS,
    UPLOAD_TEMPLATE_COLUMNS,
    WEARABLE_COLS,
    create_forecast_examples,
    empty_upload_template,
    prepare_uploaded_cohort,
    read_uploaded_csv,
)
from medprogress.experiment import run_experiment


def _valid_upload_frame(n_patients: int = 30) -> pd.DataFrame:
    rows = []
    for patient_number in range(n_patients):
        patient_id = f"SUBJECT_{patient_number:03d}"
        initial_stage = patient_number % 4
        for visit_index, month in enumerate((0, 6, 12)):
            stage = min(3, initial_stage + visit_index)
            row = {
                "patient_id": patient_id,
                "time_months": month,
                "current_stage": stage,
                "age_years": 40 + (patient_number % 40),
                "sex_binary": patient_number % 2,
                "lab_a": 0.4 + 0.1 * stage,
                "lab_b": 0.5 + 0.12 * stage,
            }
            row.update({column: stage + index * 0.01 for index, column in enumerate(IMAGE_COLS)})
            row.update({column: 1 - stage * 0.1 + index * 0.01 for index, column in enumerate(WEARABLE_COLS)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_empty_template_has_headers_but_no_records():
    template = empty_upload_template()
    assert template.empty
    assert list(template.columns) == UPLOAD_TEMPLATE_COLUMNS
    assert {"patient_id", "time_months", "current_stage"}.issubset(template.columns)


def test_uploaded_csv_is_normalized_into_model_schema():
    raw = _valid_upload_frame()
    # One visit intentionally has a fully missing imaging feature group.
    raw.loc[(raw["patient_id"] == "SUBJECT_000") & (raw["time_months"] == 6), IMAGE_COLS] = np.nan
    cohort = prepare_uploaded_cohort(raw)

    assert cohort["patient_id"].nunique() == 30
    assert len(cohort) == 90
    assert {"age_scaled", "current_stage_scaled", "visit_gap_scaled", "image_missing", "wearable_missing"}.issubset(cohort.columns)
    selected = cohort[(cohort["patient_id"] == "SUBJECT_000") & (cohort["time_months"] == 6)].iloc[0]
    assert selected["image_missing"] == 1
    assert selected["visit_gap_scaled"] == pytest.approx(0.5)
    assert selected["current_stage_scaled"] == pytest.approx(selected["current_stage"] / 3.0)
    assert cohort.attrs["synthetic_only"] is False
    examples = create_forecast_examples(cohort)
    assert examples.attrs["synthetic_only"] is False
    assert examples.attrs["data_source"] == "uploaded_csv"


def test_csv_reader_preserves_numeric_like_patient_ids():
    patient_ids = ["001", "1", *[f"{number:03d}" for number in range(2, 10)]]
    # Header whitespace is normalized by the validator; IDs must stay text meanwhile.
    lines = [" patient_id ,time_months,current_stage"]
    for patient_id in patient_ids:
        lines.append(f"{patient_id},0,0")
        lines.append(f"{patient_id},6,1")

    raw = read_uploaded_csv(("\n".join(lines) + "\n").encode("utf-8"))
    cohort = prepare_uploaded_cohort(raw)

    assert raw["patient_id"].iloc[0] == "001"
    assert cohort["patient_id"].nunique() == 10
    assert {"001", "1"}.issubset(set(cohort["patient_id"]))


def test_optional_feature_columns_can_be_omitted():
    raw = _valid_upload_frame()[["patient_id", "time_months", "current_stage"]]
    cohort = prepare_uploaded_cohort(raw)
    assert cohort["img_0"].isna().all()
    assert cohort["wear_0"].isna().all()
    assert cohort["age_scaled"].isna().all()
    assert cohort["sex_binary"].isna().all()


def test_upload_requires_stage_codes_zero_through_three():
    raw = _valid_upload_frame()
    raw.loc[0, "current_stage"] = 4
    with pytest.raises(ValueError, match="current_stage"):
        prepare_uploaded_cohort(raw)


def test_upload_rejects_duplicate_patient_visit_times():
    raw = _valid_upload_frame()
    raw.loc[1, "time_months"] = 0
    with pytest.raises(ValueError, match="duplicate time_months"):
        prepare_uploaded_cohort(raw)


def test_upload_rejects_patients_without_a_follow_up_visit():
    raw = _valid_upload_frame()
    raw = raw[~((raw["patient_id"] == "SUBJECT_000") & (raw["time_months"] != 0))]
    with pytest.raises(ValueError, match="at least two visits"):
        prepare_uploaded_cohort(raw)


def test_experiment_rejects_a_single_target_class_with_a_clear_message():
    raw = _valid_upload_frame()
    raw["current_stage"] = 0
    cohort = prepare_uploaded_cohort(raw)

    with pytest.raises(ValueError, match="training partition contains only one next-visit stage class"):
        run_experiment(seed=7, epochs=1, cohort=cohort)


def test_uploaded_cohort_runs_through_experiment_pipeline():
    cohort = prepare_uploaded_cohort(_valid_upload_frame())
    result = run_experiment(seed=7, epochs=1, cohort=cohort)
    assert result["data_mode"] == "uploaded_csv"
    assert result["cohort"]["patient_id"].nunique() == 30
    assert len(result["benchmarks"]) > 0
    assert result["neural_metrics"]["n_test_examples"] > 0
    # The causal demo remains a separate simulation by design.
    assert result["causal"]["known_simulated_effect"] == -0.8
