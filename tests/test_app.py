from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_dashboard_renders_and_sample_experiment_completes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    assert any("Multimodal Disease Progression Lab" in title.value for title in app.title)
    assert any(button.label == "Run sample experiment" for button in app.button)
    assert app.radio[0].options == [
        "Try sample data",
        "Enter simple numbers",
        "Upload a spreadsheet",
    ]

    # Keep the integration test small while exercising the complete sample path.
    app.slider[0].set_value(5)    # model training rounds
    app.slider[1].set_value(120)  # number of sample people
    app.button[0].click()
    app.run(timeout=120)

    assert not app.exception, [exc.message for exc in app.exception]
    metric_labels = {metric.label for metric in app.metric}
    assert "People in this example" in metric_labels
    assert "Visit-to-visit examples" in metric_labels
    assert "Balanced model score" in metric_labels
    assert len(app.tabs) == 6
    assert any("What did the model predict for the next visit?" in item.value for item in app.subheader)


def test_beginner_friendly_homepage_explains_data_sources_and_whole_numbers():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    rendered_markdown = "\\n".join(item.value for item in app.markdown)
    assert "How it works" in rendered_markdown
    assert "simple numbers" in rendered_markdown.lower()
    assert "no decimal points" in rendered_markdown.lower()
    assert app.radio[0].options == [
        "Try sample data",
        "Enter simple numbers",
        "Upload a spreadsheet",
    ]
    assert any("not a medical tool" in item.value.lower() for item in app.info)


def test_simple_number_entry_accepts_default_integer_table_and_runs():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    app.radio[0].set_value("Enter simple numbers")
    app.run(timeout=30)

    assert not app.exception, [exc.message for exc in app.exception]
    assert any("30 people" in item.value for item in app.success)
    assert any(button.label == "Run with my numbers" for button in app.button)

    app.slider[0].set_value(5)
    [button for button in app.button if button.label == "Run with my numbers"][0].click()
    app.run(timeout=120)

    assert not app.exception, [exc.message for exc in app.exception]
    assert len(app.tabs) == 6
    assert any("scan measurements or wearable-device measurements" in item.value for item in app.info)


def test_spreadsheet_upload_source_is_available():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    app.radio[0].set_value("Upload a spreadsheet")
    app.run(timeout=30)

    assert not app.exception, [exc.message for exc in app.exception]
    assert len(app.file_uploader) == 1
    assert any(button.label == "Run on my spreadsheet" for button in app.button)
    assert any("Download the detailed CSV template" == button.label for button in app.button)


def test_synthetic_scenario_selector_accepts_all_sample_patterns():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    selectors = [box for box in app.selectbox if box.label == "Sample data pattern"]
    assert len(selectors) == 1
    assert set(selectors[0].options) == {
        "Example with a pattern (recommended)",
        "Example without an extra pattern",
        "More difficult example",
    }

    selectors[0].set_value("Example without an extra pattern")
    app.run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
