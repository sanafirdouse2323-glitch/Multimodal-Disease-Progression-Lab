"""Synthetic longitudinal cohort generation and patient-level splitting.

All records generated here are artificial and must not be interpreted as clinical data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_COLS = [f"img_{i}" for i in range(8)]
WEARABLE_COLS = [f"wear_{i}" for i in range(6)]
CLINICAL_COLS = [
    "age_scaled",
    "sex_binary",
    "lab_a",
    "lab_b",
    "current_stage_scaled",
    "visit_gap_scaled",
]
MISSING_COLS = ["image_missing", "wearable_missing"]
FEATURE_COLS = IMAGE_COLS + WEARABLE_COLS + CLINICAL_COLS + MISSING_COLS
MODALITY_FEATURES = {
    "Imaging": IMAGE_COLS + ["image_missing"],
    "Wearables": WEARABLE_COLS + ["wearable_missing"],
    "Clinical + temporal/current stage": CLINICAL_COLS,
    "Multimodal fusion": FEATURE_COLS,
}
TARGET_COL = "target_stage"
PATIENT_COL = "patient_id"
STAGE_LABELS = [0, 1, 2, 3]


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_synthetic_cohort(
    n_patients: int = 400,
    seed: int = 42,
    image_missing_probability: float = 0.12,
    wearable_missing_probability: float = 0.18,
) -> pd.DataFrame:
    """Create artificial patients with irregular visits and three data modalities.

    Stage 0-3 is an invented ordinal label. Modalities are noisy signals driven by
    stage and latent patient risk; they are not actual scans or clinical measurements.
    """
    if n_patients < 30:
        raise ValueError("n_patients must be at least 30 for meaningful demo splits")
    for name, value in (
        ("image_missing_probability", image_missing_probability),
        ("wearable_missing_probability", wearable_missing_probability),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for patient_idx in range(n_patients):
        patient_id = f"SYNTH_{patient_idx:06d}"
        latent_risk = float(rng.normal())
        age = float(np.clip(rng.normal(62, 10), 30, 88))
        sex = int(rng.integers(0, 2))
        latent_risk += 0.12 * (age - 62) / 10.0
        visit_count = int(rng.integers(3, 7))
        month = 0.0
        stage = int(np.digitize(latent_risk, [-0.8, 0.15, 1.05]))
        patient_signature = rng.normal(0, 0.10, size=8)
        activity_signature = float(rng.normal(0, 0.08))

        for visit_idx in range(visit_count):
            gap_months = 0.0
            if visit_idx > 0:
                gap_months = float(rng.choice([3, 6, 9, 12]))
                month += gap_months
                up_probability = 0.10 + 0.07 * (stage >= 1)
                if rng.random() < up_probability:
                    stage = min(3, stage + 1)
                elif rng.random() < 0.035:
                    stage = max(0, stage - 1)

            severity = stage / 3.0
            image_values = rng.normal(
                loc=severity + 0.12 * latent_risk + patient_signature,
                scale=0.32,
                size=8,
            )
            wearable_values = rng.normal(
                loc=1.0 - 0.55 * severity + activity_signature,
                scale=0.24,
                size=6,
            )
            lab_a = float(rng.normal(0.45 + 0.38 * severity + 0.12 * latent_risk, 0.16))
            lab_b = float(rng.normal(0.35 + 0.42 * severity + 0.08 * latent_risk, 0.18))
            image_missing = bool(rng.random() < image_missing_probability)
            wearable_missing = bool(rng.random() < wearable_missing_probability)
            if image_missing:
                image_values[:] = np.nan
            if wearable_missing:
                wearable_values[:] = np.nan

            row = {
                PATIENT_COL: patient_id,
                "visit_idx": visit_idx,
                "time_months": float(month),
                "current_stage": int(stage),
                "age_scaled": age / 100.0,
                "sex_binary": float(sex),
                "lab_a": lab_a if rng.random() >= 0.07 else np.nan,
                "lab_b": lab_b if rng.random() >= 0.07 else np.nan,
                "current_stage_scaled": stage / 3.0,
                "visit_gap_scaled": gap_months / 12.0,
                "image_missing": int(image_missing),
                "wearable_missing": int(wearable_missing),
                # Synthetic demographic/causal variables are for controlled demos only.
                "synthetic_age": age,
                "synthetic_sex": sex,
            }
            row.update({col: float(value) for col, value in zip(IMAGE_COLS, image_values)})
            row.update({col: float(value) for col, value in zip(WEARABLE_COLS, wearable_values)})
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values([PATIENT_COL, "time_months"]).reset_index(drop=True)
    frame.attrs["synthetic_only"] = True
    return frame


def create_forecast_examples(cohort: pd.DataFrame) -> pd.DataFrame:
    """Make next-observed-visit examples using current visit features only."""
    required = {PATIENT_COL, "time_months", "current_stage", *FEATURE_COLS}
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"Cohort missing required columns: {sorted(missing)}")

    output: list[dict] = []
    for patient_id, visits in cohort.groupby(PATIENT_COL, sort=False):
        visits = visits.sort_values("time_months")
        if visits["time_months"].duplicated().any():
            raise ValueError(f"Patient {patient_id} has duplicate visit times")
        records = visits.to_dict("records")
        for current, future in zip(records[:-1], records[1:]):
            example = {
                PATIENT_COL: patient_id,
                "time_months": float(current["time_months"]),
                "next_interval_months": float(future["time_months"] - current["time_months"]),
                "current_stage": int(current["current_stage"]),
                TARGET_COL: int(future["current_stage"]),
                "synthetic_age": float(current["synthetic_age"]),
                "synthetic_sex": int(current["synthetic_sex"]),
            }
            example.update({col: current[col] for col in FEATURE_COLS})
            output.append(example)

    if not output:
        raise ValueError("No forecast examples could be created")
    examples = pd.DataFrame(output)
    examples.attrs["synthetic_only"] = True
    return examples


def split_by_patient(examples: pd.DataFrame, seed: int = 42) -> DataSplits:
    """Create 60/20/20 train/validation/test partitions with disjoint patients."""
    unique_patients = np.array(sorted(examples[PATIENT_COL].unique()))
    if len(unique_patients) < 10:
        raise ValueError("At least 10 unique patients are required for splitting")
    train_val_ids, test_ids = train_test_split(
        unique_patients, test_size=0.20, random_state=seed
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.25, random_state=seed
    )
    train_set, val_set, test_set = map(set, (train_ids, val_ids, test_ids))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Patient-level split overlap detected")

    def select(ids: set[str]) -> pd.DataFrame:
        return examples[examples[PATIENT_COL].isin(ids)].reset_index(drop=True).copy()

    return DataSplits(train=select(train_set), validation=select(val_set), test=select(test_set))
