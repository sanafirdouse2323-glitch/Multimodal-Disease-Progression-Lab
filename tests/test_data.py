import numpy as np

from medprogress.data import (
    FEATURE_COLS,
    IMAGE_COLS,
    TARGET_COL,
    WEARABLE_COLS,
    create_forecast_examples,
    make_synthetic_cohort,
    split_by_patient,
)


def test_synthetic_cohort_has_irregular_visits_and_modalities():
    cohort = make_synthetic_cohort(60, seed=7)
    assert set(IMAGE_COLS).issubset(cohort.columns)
    assert set(WEARABLE_COLS).issubset(cohort.columns)
    assert cohort.groupby("patient_id").size().min() >= 3
    assert cohort.groupby("patient_id")["time_months"].apply(lambda x: x.is_monotonic_increasing).all()
    assert cohort["current_stage"].between(0, 3).all()
    assert cohort[IMAGE_COLS].isna().any().any()
    assert cohort[WEARABLE_COLS].isna().any().any()


def test_forecast_target_is_next_visit_and_schema_has_22_features():
    cohort = make_synthetic_cohort(40, seed=8)
    examples = create_forecast_examples(cohort)
    assert examples.attrs["synthetic_only"] is True
    assert len(FEATURE_COLS) == 22
    row = examples.iloc[0]
    patient_visits = cohort[cohort.patient_id == row.patient_id].sort_values("time_months")
    current_idx = patient_visits.index[patient_visits.time_months == row.time_months][0]
    current_pos = patient_visits.index.get_loc(current_idx)
    assert int(row[TARGET_COL]) == int(patient_visits.iloc[current_pos + 1].current_stage)
    assert row.next_interval_months > 0
    assert TARGET_COL not in FEATURE_COLS
    assert "patient_id" not in FEATURE_COLS
    assert not {"time_months", "next_interval_months", "synthetic_age", "synthetic_sex"}.intersection(FEATURE_COLS)


def test_patient_ids_are_disjoint_across_splits():
    examples = create_forecast_examples(make_synthetic_cohort(80, seed=9))
    splits = split_by_patient(examples, seed=11)
    ids = [set(part.patient_id) for part in (splits.train, splits.validation, splits.test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert len(set.union(*ids)) == examples.patient_id.nunique()


def test_invalid_generator_settings_are_rejected():
    try:
        make_synthetic_cohort(20)
    except ValueError as exc:
        assert "at least 30" in str(exc)
    else:
        raise AssertionError("Expected too-small cohort to be rejected")

    try:
        make_synthetic_cohort(40, image_missing_probability=1.1)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected invalid missingness probability to be rejected")
