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

    assert "It does not use the progression CSV or the simple numbers entered on the homepage." in guide
    assert "not a diagnostic, prognostic, or treatment tool" in guide
    assert "not a formal fairness audit" in guide



def test_documentation_matches_dashboard_defaults_and_metric_definitions():
    guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")
    setup = (ROOT / "GITHUB_SETUP.md").read_text(encoding="utf-8")
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "### Balanced accuracy" in guide
    assert "### Macro precision, macro recall, and macro-F1" in guide
    assert "Macro-F1 is not the same as balanced accuracy." in guide
    assert "(the balanced score)" not in guide.lower()
    assert "Validation macro-F1" in guide
    assert "Enter simple numbers" in setup
    assert "Synthetic demo" in setup
    assert "default data source is the synthetic demo" not in setup.lower()
    assert '["Enter simple numbers", "Synthetic demo", "Upload CSV"]' in app_source


def test_ci_workflow_preserves_repeated_seed_outputs_and_uses_node24_actions():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    assert "actions/checkout@v7.0.1" in workflow
    assert "actions/setup-python@v7.1.0" in workflow
    assert "actions/upload-artifact@v7.0.2" in workflow
    assert "/tmp/scenario-runs.csv" in workflow
    assert "/tmp/scenario-summary.csv" in workflow
    assert "retention-days: 30" in workflow


def test_upload_guide_matches_the_supported_csv_inputs_and_privacy_rules():
    guide = (ROOT / "UPLOAD_DATA_GUIDE.md").read_text(encoding="utf-8")
    data_source = (ROOT / "src" / "medprogress" / "data.py").read_text(encoding="utf-8")
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    for field in (
        "patient_id", "time_months", "current_stage", "age_years",
        "sex_binary", "lab_a", "lab_b", "img_0", "img_7", "wear_0", "wear_5",
        "Person number", "Month 0 stage", "Month 6 stage", "Month 12 stage",
    ):
        assert field in guide

    assert "20 MB" in guide
    assert "at least two visits" in guide
    assert "not stratified" in guide
    assert "separate synthetic dataset" in guide
    assert "Do not upload" in guide
    assert "max_upload_size=20" in app_source
    assert "UPLOAD_REQUIRED_COLUMNS = [\"patient_id\", \"time_months\", \"current_stage\"]" in data_source


def test_guides_explain_follow_up_interval_and_window_weighted_subgroups():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    methods = (ROOT / "METHODS_AND_LIMITATIONS.md").read_text(encoding="utf-8")
    guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")

    assert "not conditioned on a specified future follow-up interval" in readme
    assert "evaluation combines examples with different next-visit intervals" in methods
    assert "median age among training prediction windows" in guide
    assert "not aggregated once per participant" in methods
    assert "intersectional groups are not calculated" in guide


def test_sex_binary_upload_field_is_documented_as_both_predictor_and_subgroup():
    guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")
    upload = (ROOT / "UPLOAD_DATA_GUIDE.md").read_text(encoding="utf-8")
    methods = (ROOT / "METHODS_AND_LIMITATIONS.md").read_text(encoding="utf-8")
    data_source = (ROOT / "src" / "medprogress" / "data.py").read_text(encoding="utf-8")

    assert '"sex_binary",' in data_source
    assert "Included as a clinical model input" in guide
    assert "not merely a reporting label" in guide
    assert "model input as well as a subgrouping field" in upload
    assert "can influence model predictions" in upload
    assert "`sex_binary`" in methods


def test_upload_documentation_states_sex_code_consistency_rule():
    guide = (ROOT / "UPLOAD_DATA_GUIDE.md").read_text(encoding="utf-8")
    data_guide = (ROOT / "DATA_AND_RESULTS_GUIDE.md").read_text(encoding="utf-8")
    data_source = (ROOT / "src" / "medprogress" / "data.py").read_text(encoding="utf-8")

    assert "consistent across visits for the same participant" in guide
    assert "consistent across one participant's visits" in data_guide
    assert "patients_with_conflicting_sex_codes" in data_source
