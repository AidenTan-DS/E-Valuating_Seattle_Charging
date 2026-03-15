"""
Unit tests for Prediction tab logic (app_v3 Tab 3) and ml_model helpers.

These tests use simple, pure functions that are easy to verify.
Run from project root: pytest tests/test_prediction.py -v
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add interactive_map to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))

from ml_model import (
    filter_recommended_by_probability,
    load_recommended_from_csv,
    geoms_to_plotly,
    EMPTY_RECOMMENDED,
)


# ── filter_recommended_by_probability ─────────────────────────────────────────

def test_filter_recommended_empty_dataframe():
    """Empty input returns empty output."""
    result = filter_recommended_by_probability(EMPTY_RECOMMENDED, 0.5)
    assert result.empty


def test_filter_recommended_no_predicted_prob_column():
    """If predicted_prob column is missing, return input unchanged."""
    df = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "ZIP": ["Cell 0"],
    })
    result = filter_recommended_by_probability(df, 0.5)
    assert len(result) == 1
    assert "predicted_prob" not in result.columns


def test_filter_recommended_keeps_above_threshold():
    """Locations with prob >= threshold are kept."""
    df = pd.DataFrame({
        "Latitude": [47.6, 47.61, 47.62],
        "Longitude": [-122.3, -122.31, -122.32],
        "predicted_prob": [0.4, 0.6, 0.8],
        "ZIP": ["Cell 0", "Cell 1", "Cell 2"],
    })
    result = filter_recommended_by_probability(df, 0.6)
    assert len(result) == 2
    assert result["predicted_prob"].min() >= 0.6


def test_filter_recommended_includes_exact_threshold():
    """Locations with prob exactly equal to threshold are kept."""
    df = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "predicted_prob": [0.5],
        "ZIP": ["Cell 0"],
    })
    result = filter_recommended_by_probability(df, 0.5)
    assert len(result) == 1


def test_filter_recommended_all_below_returns_empty():
    """When all probs are below threshold, returns empty DataFrame."""
    df = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "predicted_prob": [0.3],
        "ZIP": ["Cell 0"],
    })
    result = filter_recommended_by_probability(df, 0.5)
    assert result.empty


# ── load_recommended_from_csv ────────────────────────────────────────────────

def test_load_recommended_missing_file():
    """Missing file returns empty DataFrame with correct columns."""
    result = load_recommended_from_csv("/nonexistent/path/to/file.csv")
    assert result.empty
    assert list(result.columns) == ["Latitude", "Longitude", "ZIP", "predicted_prob", "Location"]


def test_load_recommended_valid_csv():
    """Valid CSV with all columns loads correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,Location,cell_id\n")
        f.write("47.6,-122.3,0.75,Grid cell 0 (prob: 0.75),0\n")
        f.write("47.61,-122.31,0.9,Grid cell 1 (prob: 0.90),1\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 2
        assert result["ZIP"].tolist() == ["Cell 0", "Cell 1"]
        assert result["predicted_prob"].tolist() == [0.75, 0.9]
        assert result["Latitude"].tolist() == [47.6, 47.61]
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_csv_missing_optional_columns():
    """CSV with only Latitude, Longitude still loads (fills defaults)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude\n")
        f.write("47.6,-122.3\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 1
        assert "predicted_prob" in result.columns
        assert "Location" in result.columns
        assert result["ZIP"].iloc[0] == "Cell 0"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_empty_csv():
    """Empty CSV returns empty DataFrame."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,Location\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert result.empty
    finally:
        Path(path).unlink(missing_ok=True)


# ── geoms_to_plotly ──────────────────────────────────────────────────────────

def test_geoms_to_plotly_empty():
    """Empty geometry series returns (None, None)."""
    from shapely.geometry import LineString
    result = geoms_to_plotly([])
    assert result == (None, None)


def test_geoms_to_plotly_single_linestring():
    """Single LineString converts to lons/lats with None separator."""
    from shapely.geometry import LineString
    geom = LineString([(0, 1), (2, 3)])
    lons, lats = geoms_to_plotly([geom])
    assert lons == [0, 2, None]
    assert lats == [1, 3, None]


def test_geoms_to_plotly_multilinestring():
    """MultiLineString converts each part with None separators."""
    from shapely.geometry import LineString, MultiLineString
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(2, 2), (3, 3)])
    multi = MultiLineString([line1, line2])
    lons, lats = geoms_to_plotly([multi])
    assert lons == [0, 1, None, 2, 3, None]
    assert lats == [0, 1, None, 2, 3, None]  # Wait, (2,2) and (3,3) so lats = [0,1,None,2,3,None]
    # Actually: line1 coords (0,0),(1,1) -> lons [0,1] lats [0,1] then None,None
    # line2 coords (2,2),(3,3) -> lons [2,3] lats [2,3] then None,None
    # So lons = [0,1,None,2,3,None], lats = [0,1,None,2,3,None]
    assert lats == [0, 1, None, 2, 3, None]


def test_geoms_to_plotly_skips_none_and_empty():
    """None and empty geometries are skipped."""
    from shapely.geometry import LineString
    geom = LineString([(0, 0), (1, 1)])
    lons, lats = geoms_to_plotly([None, geom, LineString()])
    assert lons == [0, 1, None]
    assert lats == [0, 1, None]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
