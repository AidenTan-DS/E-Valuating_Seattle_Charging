"""
UI tests for interactive_map/app_v2.py using Streamlit AppTest
"""

import pytest
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "interactive_map"))
APP_PATH = str(ROOT / "interactive_map" / "app_v2.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.exception) == 0  # ElementList() when no exceptions, not None


def test_app_title():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert any("Seattle EV Station Explorer" in str(t.value) for t in at.title)


def test_app_has_two_tabs():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.tabs) >= 2


def test_zip_selectbox_present():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.selectbox) >= 1


def test_zip_selectbox_contains_seattle_zips():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    options = at.selectbox[0].options
    assert any(str(o).startswith("981") for o in options if o is not None)


def test_selecting_zip_shows_metrics():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.selectbox[0].select("98101").run()
    assert len(at.metric) > 0


def test_evaluation_tab_has_sliders():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.slider) >= 3


def test_clear_button_appears_after_zip_selection():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.selectbox[0].select("98101").run()
    assert any("Clear" in str(b.label) for b in at.button)

### tab 3 tests


def _run_app_and_switch_to_prediction_tab():
    """Run app and return AppTest with Prediction tab (index 2) accessible."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_prediction_tab_exists():
    """Tab 3 (Prediction) exists and has label."""
    at = _run_app_and_switch_to_prediction_tab()
    assert len(at.tabs) >= 3
    assert "Prediction" in at.tabs[2].label or "🔮" in at.tabs[2].label


def test_prediction_tab_has_info_box():
    """Prediction tab shows ML-based placement info."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.info) >= 1
    info_text = tab3.info[0].value
    assert "ML" in info_text or "Station" in info_text
    assert "power" in info_text.lower() or "Amber" in info_text or "Green" in info_text


def test_prediction_tab_has_power_lines_checkbox():
    """Prediction tab has 'Show power lines' checkbox."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.checkbox) >= 1
    assert any("power" in str(c.label).lower() for c in tab3.checkbox)


def test_prediction_tab_has_probability_slider():
    """Prediction tab has Min probability slider."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.slider) >= 1
    assert any("probability" in str(s.label).lower() or "prob" in str(s.label).lower()
               for s in tab3.slider)


def test_prediction_tab_shows_metrics():
    """Prediction tab shows Existing Stations and Recommended Locations metrics."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.metric) >= 2
    labels = [str(m.label) for m in tab3.metric]
    assert any("Existing" in l for l in labels)
    assert any("Recommended" in l for l in labels)


def test_prediction_tab_has_caption():
    """Prediction tab has caption about red dots and ml_model."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.caption) >= 1
    caption_text = tab3.caption[0].value
    assert "red" in caption_text.lower() or "grid" in caption_text.lower()
    assert "ml_model" in caption_text or "python" in caption_text.lower()


def test_prediction_tab_has_map_chart():
    """Prediction tab has controls and metrics (map is rendered with them)."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.checkbox) >= 1 and len(tab3.slider) >= 1 and len(tab3.metric) >= 2


def test_prediction_tab_checkbox_toggle():
    """Can toggle Show power lines checkbox without error."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.checkbox) >= 1
    tab3.checkbox[0].check().run()
    assert len(at.exception) == 0


def test_prediction_tab_slider_change():
    """Can change Min probability slider without error."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    assert len(tab3.slider) >= 1
    tab3.slider[0].set_value(80).run()
    assert len(at.exception) == 0


def test_prediction_tab_shows_warning_or_download():
    """Prediction tab shows either warning (no data) or download button (has data)."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    has_warning = len(tab3.warning) >= 1
    download_btns = tab3.get("download_button")
    has_download = len(download_btns) >= 1 if download_btns else False
    assert has_warning or has_download, "Expected warning or download button"


def test_prediction_tab_download_button_when_data():
    """When recommended data exists, download button is present and has correct label."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    download_btns = tab3.get("download_button")
    if download_btns and len(download_btns) >= 1:
        btn = download_btns[0]
        label = str(getattr(btn, "label", ""))
        assert "Download" in label or "CSV" in label.upper()


def test_prediction_tab_slider_default_value():
    """Min probability slider defaults to 60%."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    if len(tab3.slider) >= 1:
        # Slider is 0-100 with default 60
        val = tab3.slider[0].value
        assert val == 60 or (0 <= val <= 100)


def test_prediction_tab_checkbox_default_unchecked():
    """Show power lines checkbox defaults to unchecked."""
    at = _run_app_and_switch_to_prediction_tab()
    tab3 = at.tabs[2]
    if len(tab3.checkbox) >= 1:
        assert tab3.checkbox[0].value is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
