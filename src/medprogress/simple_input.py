"""Beginner-friendly integer-only inputs for the dashboard.

This path intentionally uses only visit month and generic stage labels. It does
not invent imaging or wearable measurements, and it is not a clinical dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import prepare_uploaded_cohort

PERSON_COLUMN = "Person number"
STAGE_COLUMNS = ("Month 0 stage", "Month 6 stage", "Month 12 stage")
SIMPLE_TABLE_COLUMNS = (PERSON_COLUMN, *STAGE_COLUMNS)

# Generic, artificial examples. Codes 0-3 are categories, not medical diagnoses.
_DEFAULT_STAGE_PATTERNS = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 3),
    (3, 3, 3),
)


def make_simple_stage_table(n_people: int = 30) -> pd.DataFrame:
    """Return an editable table with integer IDs and three generic stage labels."""
    if n_people < 10:
        raise ValueError("n_people must be at least 10.")
    rows = []
    for person_number in range(1, n_people + 1):
        pattern = _DEFAULT_STAGE_PATTERNS[(person_number - 1) % len(_DEFAULT_STAGE_PATTERNS)]
        rows.append({
            PERSON_COLUMN: person_number,
            STAGE_COLUMNS[0]: pattern[0],
            STAGE_COLUMNS[1]: pattern[1],
            STAGE_COLUMNS[2]: pattern[2],
        })
    return pd.DataFrame(rows, columns=SIMPLE_TABLE_COLUMNS, dtype="int64")


def simple_table_to_cohort(table: pd.DataFrame) -> pd.DataFrame:
    """Convert the simple wide table into the validated visit-level model input."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError("Enter the values in the number table.")
    missing = [column for column in SIMPLE_TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError("The number table is missing one or more columns. Reset the example table and try again.")

    frame = table.loc[:, list(SIMPLE_TABLE_COLUMNS)].copy()
    # Ignore completely blank rows that may be added by the editable table.
    frame = frame.dropna(how="all").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Enter at least 10 people in the number table.")

    numeric = pd.DataFrame(index=frame.index)
    for column in SIMPLE_TABLE_COLUMNS:
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
        if numeric[column].isna().any():
            raise ValueError(f"Fill in every cell in '{column}' using a whole number.")
        values = numeric[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Use finite whole numbers in '{column}'.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Use whole numbers only in '{column}' (no decimal points).")

    people = numeric[PERSON_COLUMN].astype(int)
    if (people < 1).any():
        raise ValueError("Person numbers must be positive whole numbers, starting at 1.")
    if people.duplicated().any():
        raise ValueError("Each person number must appear only once in the table.")

    for column in STAGE_COLUMNS:
        if not numeric[column].isin([0, 1, 2, 3]).all():
            raise ValueError(f"'{column}' must contain stage numbers 0, 1, 2, or 3.")

    if len(numeric) < 10:
        raise ValueError("At least 10 different people are needed. I recommend starting with the 30 example rows.")

    rows: list[dict[str, int | str]] = []
    months = (0, 6, 12)
    for row in numeric.itertuples(index=False, name=None):
        person_number = int(row[0])
        for month, stage in zip(months, row[1:]):
            rows.append({
                "patient_id": f"PERSON_{person_number:04d}",
                "time_months": month,
                "current_stage": int(stage),
            })

    cohort = prepare_uploaded_cohort(pd.DataFrame(rows))
    cohort.attrs["data_source"] = "simple_numbers"
    cohort.attrs["synthetic_only"] = False
    return cohort
