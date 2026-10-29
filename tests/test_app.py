from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_and_full_experiment_action_completes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    assert any("Multimodal Disease Progression Lab" in title.value for title in app.title)
    assert any(button.label == "Run full experiment" for button in app.button)

    # Use the smallest settings supported by the dashboard to exercise its UI path.
    app.slider[0].set_value(120)
    app.slider[1].set_value(5)
    run_buttons = [button for button in app.button if button.label == "Run full experiment"]
    run_buttons[0].click()
    app.run(timeout=120)

    assert not app.exception, [exc.message for exc in app.exception]
    metric_labels = {metric.label for metric in app.metric}
    assert "Synthetic patients" in metric_labels
    assert "Forecast examples" in metric_labels
    assert "Neural fusion macro-F1" in metric_labels
    assert len(app.tabs) == 6



def test_plain_language_guide_is_visible_on_the_homepage():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    rendered_markdown = "\\n".join(item.value for item in app.markdown)
    assert "What does “next visit” mean?" in rendered_markdown
    assert "What do the six result tabs show?" in rendered_markdown
    assert "How to understand the scores" in rendered_markdown
    assert "Can I use my own data?" in rendered_markdown
    assert "This prototype is not a diagnostic tool" in rendered_markdown


def test_empty_csv_template_mode_is_available():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]

    app.radio[0].set_value("Empty CSV template")
    app.run(timeout=30)

    assert not app.exception, [exc.message for exc in app.exception]
    assert any("Start with an empty data shell" in item.value for item in app.subheader)
    assert any("Required columns" in item.value for item in app.markdown)
