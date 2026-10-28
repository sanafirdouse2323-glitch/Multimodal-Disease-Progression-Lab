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

# CSV schema for user-supplied, feature-level longitudinal data. Raw medical
# images and raw wearable streams are deliberately outside this interface.
UPLOAD_REQUIRED_COLUMNS = ["patient_id", "time_months", "current_stage"]
UPLOAD_OPTIONAL_COLUMNS = [
    "age_years",
    "sex_binary",
    "lab_a",
    "lab_b",
    *IMAGE_COLS,
    *WEARABLE_COLS,
]
UPLOAD_TEMPLATE_COLUMNS = UPLOAD_REQUIRED_COLUMNS + UPLOAD_OPTIONAL_COLUMNS


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



def empty_upload_template() -> pd.DataFrame:
    """Return a header-only CSV template for user-supplied visit-level features."""
    return pd.DataFrame(columns=UPLOAD_TEMPLATE_COLUMNS)


def prepare_uploaded_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded CSV to the model's feature schema.

    The CSV must contain one row per patient visit and the required columns
    patient_id, time_months, current_stage. Optional numeric columns are allowed
    to be blank or omitted. Stage labels must be integer codes 0, 1, 2, or 3.
    This function does not ingest raw images or raw wearable sensor recordings.
    """
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("Upload a CSV table with column headers.")
    if raw.empty:
        raise ValueError("The CSV has headers but no data rows. Fill in the template first.")
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("The CSV contains duplicate column names. Make every header unique.")

    missing_required = sorted(set(UPLOAD_REQUIRED_COLUMNS) - set(frame.columns))
    if missing_required:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_required)
            + ". Download the blank template to see the expected format."
        )

    if frame["patient_id"].isna().any():
        raise ValueError("patient_id cannot be blank. Use a non-identifying study code for each person.")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    if frame["patient_id"].eq("").any():
        raise ValueError("patient_id cannot be blank. Use a non-identifying study code for each person.")

    for column in UPLOAD_OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    numeric_columns = ["time_months", "current_stage", *UPLOAD_OPTIONAL_COLUMNS]
    for column in numeric_columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        nonempty = original.notna() & original.astype(str).str.strip().ne("")
        invalid = nonempty & numeric.isna()
        if invalid.any():
            examples = original[invalid].astype(str).head(3).tolist()
            raise ValueError(
                f"Column '{column}' must contain numeric values or blanks. "
                f"Invalid example(s): {examples}"
            )
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Column '{column}' contains an infinite value. Replace it with a number or blank.")
        frame[column] = numeric

    if frame["time_months"].isna().any():
        raise ValueError("time_months is required for every visit and must be numeric.")
    if (frame["time_months"] < 0).any():
        raise ValueError("time_months must be zero or positive months since the patient's baseline visit.")
    if frame["current_stage"].isna().any():
        raise ValueError("current_stage is required for every visit.")
    stage_values = frame["current_stage"].to_numpy(dtype=float)
    if not np.equal(stage_values, np.floor(stage_values)).all() or not np.isin(stage_values, STAGE_LABELS).all():
        raise ValueError("current_stage must use integer labels 0, 1, 2, or 3. Recode your target labels before uploading.")

    age_values = frame["age_years"].dropna()
    if ((age_values < 0) | (age_values > 120)).any():
        raise ValueError("Nonblank age_years values must be between 0 and 120.")
    sex_values = frame["sex_binary"].dropna()
    if not sex_values.isin([0, 1]).all():
        raise ValueError("Nonblank sex_binary values must be coded consistently as 0 or 1, or left blank.")

    patient_count = frame["patient_id"].nunique()
    if patient_count < 10:
        raise ValueError(
            f"At least 10 unique patient IDs are required for train/validation/test splitting; found {patient_count}."
        )
    visit_counts = frame.groupby("patient_id").size()
    patients_with_one_visit = visit_counts[visit_counts < 2]
    if not patients_with_one_visit.empty:
        sample_ids = patients_with_one_visit.index.astype(str).tolist()[:3]
        raise ValueError(
            "Each patient must have at least two visits to create a next-visit prediction example. "
            f"Examples with only one row: {sample_ids}"
        )
    duplicate_times = frame.duplicated(subset=["patient_id", "time_months"], keep=False)
    if duplicate_times.any():
        raise ValueError("A patient has duplicate time_months values. Give every visit a distinct time for that patient.")

    frame = frame.sort_values(["patient_id", "time_months"]).reset_index(drop=True)
    cohort = pd.DataFrame({
        "patient_id": frame["patient_id"],
        "time_months": frame["time_months"].astype(float),
        "current_stage": frame["current_stage"].astype(int),
        "age_scaled": frame["age_years"].astype(float) / 100.0,
        "sex_binary": frame["sex_binary"].astype(float),
        "lab_a": frame["lab_a"].astype(float),
        "lab_b": frame["lab_b"].astype(float),
    })
    for column in IMAGE_COLS + WEARABLE_COLS:
        cohort[column] = frame[column].astype(float)

    cohort["visit_idx"] = cohort.groupby("patient_id").cumcount().astype(int)
    cohort["current_stage_scaled"] = cohort["current_stage"].astype(float) / 3.0
    cohort["visit_gap_scaled"] = (
        cohort.groupby("patient_id")["time_months"].diff().fillna(0.0).astype(float) / 12.0
    )
    cohort["image_missing"] = cohort[IMAGE_COLS].isna().all(axis=1).astype(int)
    cohort["wearable_missing"] = cohort[WEARABLE_COLS].isna().all(axis=1).astype(int)
    cohort.attrs["synthetic_only"] = False
    cohort.attrs["data_source"] = "uploaded_csv"
    return cohort
