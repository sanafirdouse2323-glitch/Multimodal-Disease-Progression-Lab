from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medprogress.data import (
    IMAGE_COLS,
    SYNTHETIC_SCENARIOS,
    TARGET_COL,
    WEARABLE_COLS,
    create_forecast_examples,
    make_synthetic_cohort,
    split_by_patient,
)


def _transition_auc(examples: pd.DataFrame, use_measurements: bool) -> float:
    """Score whether current-visit measurements indicate an upward next-stage transition."""
    def matrix(frame: pd.DataFrame, add_measurements: bool) -> np.ndarray:
        stage = frame["current_stage"].astype(int).to_numpy()
        columns = [np.eye(4, dtype=float)[stage]]
        if add_measurements:
            # Aggregate the generated modalities so the test checks for signal in
            # measurements, not a special hand-coded latent-risk feature.
            measurements = np.column_stack([
                frame[IMAGE_COLS].mean(axis=1).to_numpy(),
                frame[WEARABLE_COLS].mean(axis=1).to_numpy(),
                frame["lab_a"].to_numpy(dtype=float),
                frame["lab_b"].to_numpy(dtype=float),
            ])
            columns.append(measurements)
        return np.column_stack(columns)

    # The helper receives only a single split and evaluates on patients held out
    # by split_by_patient; no participant appears in both training and test data.
    train_mask = examples.attrs.get("_unused", None)
    del train_mask  # Avoid attaching transient split metadata to the forecast frame.
    raise RuntimeError("_transition_auc must be called with train/test frames")


def _auc_for_split(train: pd.DataFrame, test: pd.DataFrame, use_measurements: bool) -> float:
    def matrix(frame: pd.DataFrame) -> np.ndarray:
        stage = frame["current_stage"].astype(int).to_numpy()
        columns = [np.eye(4, dtype=float)[stage]]
        if use_measurements:
            columns.append(np.column_stack([
                frame[IMAGE_COLS].mean(axis=1).to_numpy(),
                frame[WEARABLE_COLS].mean(axis=1).to_numpy(),
                frame["lab_a"].to_numpy(dtype=float),
                frame["lab_b"].to_numpy(dtype=float),
            ]))
        return np.column_stack(columns)

    y_train = (train[TARGET_COL].to_numpy() > train["current_stage"].to_numpy()).astype(int)
    y_test = (test[TARGET_COL].to_numpy() > test["current_stage"].to_numpy()).astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=13),
    )
    model.fit(matrix(train), y_train)
    scores = model.predict_proba(matrix(test))[:, 1]
    return float(roc_auc_score(y_test, scores))


def test_synthetic_scenarios_are_named_reproducible_and_valid():
    assert SYNTHETIC_SCENARIOS == ("no_signal", "progression_signal", "difficult")
    for scenario in SYNTHETIC_SCENARIOS:
        first = make_synthetic_cohort(n_patients=40, seed=29, scenario=scenario)
        second = make_synthetic_cohort(n_patients=40, seed=29, scenario=scenario)
        pd.testing.assert_frame_equal(first, second)
        assert first.attrs["synthetic_only"] is True
        assert first.attrs["synthetic_scenario"] == scenario
        assert first["current_stage"].between(0, 3).all()
        examples = create_forecast_examples(first)
        assert examples.attrs["synthetic_scenario"] == scenario
        assert set(examples[TARGET_COL].unique()).issubset({0, 1, 2, 3})


def test_progression_signal_scenario_places_recoverable_signal_in_measurements():
    # This is a generator sanity check, not a claim about real medical data.
    # Using several thousand synthetic patients makes the test less sensitive to
    # chance variation while keeping patients disjoint across the evaluation split.
    cohort = make_synthetic_cohort(
        n_patients=1800, seed=731, scenario="progression_signal"
    )
    examples = create_forecast_examples(cohort)
    splits = split_by_patient(examples, seed=731)

    stage_only_auc = _auc_for_split(splits.train, splits.test, use_measurements=False)
    multimodal_auc = _auc_for_split(splits.train, splits.test, use_measurements=True)

    assert np.isfinite(stage_only_auc)
    assert np.isfinite(multimodal_auc)
    assert multimodal_auc > stage_only_auc + 0.03


def test_unknown_synthetic_scenario_is_rejected():
    try:
        make_synthetic_cohort(n_patients=40, seed=1, scenario="invented")
    except ValueError as exc:
        assert "scenario must be one of" in str(exc)
    else:
        raise AssertionError("Unknown scenarios must be rejected")
