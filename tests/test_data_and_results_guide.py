from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_includes_full_data_and_results_guide():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    guide_path = ROOT / "DATA_AND_RESULTS_GUIDE.md"

    assert guide_path.is_file()
    assert "DATA_AND_RESULTS_GUIDE.md" in app_source
    assert "Full guide: data fields, processing, models, and result meanings" in app_source
    assert "synthetic_preview_rows" in app_source
    assert "make_synthetic_cohort(n_patients=30, seed=42, scenario=scenario)" in app_source


def test_data_guide_documents_model_inputs_and_upload_fields():
    guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")

    for field in (
        "patient_id",
        "time_months",
        "current_stage",
        "age_years",
        "sex_binary",
        "lab_a",
        "lab_b",
        "img_0 to img_7",
        "wear_0 to wear_5",
        "age_scaled",
        "current_stage_scaled",
        "visit_gap_scaled",
        "image_missing",
        "wearable_missing",
        "target_stage",
        "next_interval_months",
    ):
        assert field in guide

    assert "22 numerical model features" in guide
    assert "pre-computed numerical imaging-related features" in guide
    assert "raw wearable sensor streams" in guide


def test_data_guide_explains_each_dashboard_result_and_limitations():
    guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")

    for section in (
        "What the summary cards mean",
        "What the prediction scores mean",
        "What the changed-stage table means",
        "What the data-group comparison means",
        "What the missing-information experiment means",
        "What the practice simulation means",
        "What the group summaries mean",
        "What appears in My data and downloads",
        "What the application does not currently provide",
        "Privacy and responsible use",
    ):
        assert section in guide

    assert "The causal tab" not in guide or "separate synthetic" in guide
    assert "not a diagnostic, prognostic, or treatment tool" in guide
    assert "not a formal fairness audit" in guide
