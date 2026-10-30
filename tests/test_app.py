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
        "Synthetic demo",
        "Enter simple numbers",
        "Upload CSV",
    ]
    assert any(download.label == "Download empty CSV template" for download in app.download_button)
    assert any(slider.label == "Model training rounds" for slider in app.slider)
    assert any(slider.label == "Number of sample people" for slider in app.slider)

    # Keep the integration test small while exercising the complete sample path.
    app.slider[0].set_value(5)    # model training rounds
    app.slider[1].set_value(120)  # number of sample people
    [button for button in app.button if button.label == "Run sample experiment"][0].click()
    app.run(timeout=120)

    assert not app.exception, [exc.message for exc in app.exception]
    metric_labels = {metric.label for metric in app.metric}
    assert "People in this example" in metric_labels
    assert "Visit-to-visit examples" in metric_labels
    assert "Neural model score" in metric_labels
    assert len(app.tabs) == 6
    assert any("What did the model predict for the next visit?" in item.value for item in app.subheader)
    assert any("Start here: how this demo works" in item.value for item in app.subheader)


def test_beginner_friendly_homepage_explains_data_sources_and_whole_numbers():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    rendered_text = "\\n".join(
        [item.value for item in app.markdown]
        + [item.label for item in app.radio]
        + [item.label for item in app.caption]
    )
    assert "Experiment settings" in rendered_text or any(
        "Experiment settings" in item.value for item in app.subheader
    )
    assert "whole numbers" in rendered_text.lower()
    assert app.radio[0].options == [
        "Synthetic demo",
        "Enter simple numbers",
        "Upload CSV",
    ]
    assert any(download.label == "Download empty CSV template" for download in app.download_button)
    assert any(number_input.label == "Random seed" for number_input in app.number_input)
    # Streamlit's AppTest exposes these element collections separately; do not
    # concatenate ElementList instances (that is unsupported).
    safety_notices = [item.value for item in app.info] + [item.value for item in app.warning]
    assert any("not a medical tool" in value.lower() for value in safety_notices)

    app.radio[0].set_value("Enter simple numbers")
    app.run(timeout=30)
    assert any("no decimal points" in item.value.lower() for item in app.caption)


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
    app.radio[0].set_value("Upload CSV")
    app.run(timeout=30)

    assert not app.exception, [exc.message for exc in app.exception]
    assert len(app.file_uploader) == 1
    assert any(button.label == "Run on my spreadsheet" for button in app.button)
    assert any(uploader.label == "Choose a CSV file" for uploader in app.file_uploader)


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
