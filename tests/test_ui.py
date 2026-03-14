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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
