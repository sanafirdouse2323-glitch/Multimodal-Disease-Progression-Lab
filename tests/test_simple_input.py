import pandas as pd
import pytest

from medprogress.simple_input import (
    PERSON_COLUMN,
    SIMPLE_TABLE_COLUMNS,
    STAGE_COLUMNS,
    make_simple_stage_table,
    simple_table_to_cohort,
)


def test_default_simple_table_is_integer_only_and_converts_to_visits():
    table = make_simple_stage_table()
    assert list(table.columns) == list(SIMPLE_TABLE_COLUMNS)
    assert len(table) == 30
    assert all(str(dtype).startswith("int") for dtype in table.dtypes)

    cohort = simple_table_to_cohort(table)
    assert cohort["patient_id"].nunique() == 30
    assert len(cohort) == 90
    assert set(cohort["time_months"].unique()) == {0.0, 6.0, 12.0}
    assert cohort["current_stage"].between(0, 3).all()
    assert cohort.attrs["data_source"] == "simple_numbers"
    assert cohort.attrs["synthetic_only"] is False


def test_simple_table_rejects_decimal_values():
    table = make_simple_stage_table()
    # Pandas 3.x does not allow assigning a float into an integer column.
    table[STAGE_COLUMNS[0]] = table[STAGE_COLUMNS[0]].astype(float)
    table.loc[0, STAGE_COLUMNS[0]] = 1.5
    with pytest.raises(ValueError, match="whole numbers only"):
        simple_table_to_cohort(table)


def test_simple_table_rejects_duplicate_person_numbers():
    table = make_simple_stage_table()
    table.loc[1, PERSON_COLUMN] = table.loc[0, PERSON_COLUMN]
    with pytest.raises(ValueError, match="appear only once"):
        simple_table_to_cohort(table)


def test_simple_table_rejects_stage_codes_outside_zero_to_three():
    table = make_simple_stage_table()
    table.loc[0, STAGE_COLUMNS[0]] = 4
    with pytest.raises(ValueError, match="0, 1, 2, or 3"):
        simple_table_to_cohort(table)


def test_simple_table_requires_at_least_ten_people():
    table = make_simple_stage_table().iloc[:9]
    with pytest.raises(ValueError, match="At least 10"):
        simple_table_to_cohort(table)


def test_simple_table_rejects_missing_values_but_ignores_blank_added_rows():
    table = make_simple_stage_table()
    blank = pd.DataFrame([{column: None for column in SIMPLE_TABLE_COLUMNS}])
    cohort = simple_table_to_cohort(pd.concat([table, blank], ignore_index=True))
    assert cohort["patient_id"].nunique() == 30

    table.loc[0, STAGE_COLUMNS[1]] = None
    with pytest.raises(ValueError, match="Fill in every cell"):
        simple_table_to_cohort(table)


def test_simple_sheet_can_be_uploaded_and_preserves_person_ids():
    from medprogress.data import prepare_uploaded_cohort

    table = make_simple_stage_table()
    raw = table.rename(columns={
        PERSON_COLUMN: "Person number",
        STAGE_COLUMNS[0]: "Month 0 stage",
        STAGE_COLUMNS[1]: "Month 6 stage",
        STAGE_COLUMNS[2]: "Month 12 stage",
    })
    cohort = simple_table_to_cohort(raw)
    assert cohort["patient_id"].nunique() == 30
    assert set(cohort["time_months"]) == {0.0, 6.0, 12.0}
    assert cohort.attrs["data_source"] == "simple_numbers"


def test_simple_table_rejects_nonpositive_and_duplicate_person_identifiers():
    table = make_simple_stage_table()
    table.loc[0, PERSON_COLUMN] = 0
    with pytest.raises(ValueError, match="positive whole numbers"):
        simple_table_to_cohort(table)

    table = make_simple_stage_table()
    table.loc[1, PERSON_COLUMN] = table.loc[0, PERSON_COLUMN]
    with pytest.raises(ValueError, match="appear only once"):
        simple_table_to_cohort(table)
