"""
Unit tests for Prediction tab logic (app_v2 Tab 3) and ml_model helpers.

Includes edge tests, one-shot tests, and exception tests.
Run from project root: pytest tests/test_ml_model.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

# Add interactive_map to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))
from ml_model import (
    filter_recommended_by_probability,
    load_recommended_from_csv,
    geoms_to_plotly,
    get_recommended_stations,
    get_electric_lines_for_map,
    get_zip_predictions,
    generate_electric_lines_cache,
    generate_grid_recommendations,
    EMPTY_RECOMMENDED,
    _zcta_zip_series,
    _prepare_features,
    _get_zip_centroids,
)


# ── _zcta_zip_series ────────────────────────────────────────────────────────

def test_zcta_zip_series_uses_zip_zcta_when_present():
    """When ZIP_zcta column exists, use it with zfill(5)."""
    gdf = gpd.GeoDataFrame({"ZIP_zcta": ["123", "4567"], "geometry": [Point(0, 0), Point(1, 1)]})
    result = _zcta_zip_series(gdf)
    assert list(result) == ["00123", "04567"]


def test_zcta_zip_series_uses_zcta5ce10_when_no_zip_zcta():
    """When ZIP_zcta missing but ZCTA5CE10 exists, use ZCTA5CE10."""
    gdf = gpd.GeoDataFrame({
        "ZCTA5CE10": ["98105", "98122"],
        "geometry": [Point(0, 0), Point(1, 1)],
    })
    result = _zcta_zip_series(gdf)
    assert list(result) == ["98105", "98122"]


def test_zcta_zip_series_uses_geoid10_when_only_geoid10():
    """When only GEOID10 exists, use it."""
    gdf = gpd.GeoDataFrame({
        "GEOID10": ["98101", "98102"],
        "geometry": [Point(0, 0), Point(1, 1)],
    })
    result = _zcta_zip_series(gdf)
    assert list(result) == ["98101", "98102"]


def test_zcta_zip_series_geoid10_zfill_short_values():
    """GEOID10 with short values gets zfill(5)."""
    gdf = gpd.GeoDataFrame({
        "GEOID10": ["1", "99"],
        "geometry": [Point(0, 0), Point(1, 1)],
    })
    result = _zcta_zip_series(gdf)
    assert list(result) == ["00001", "00099"]


def test_zcta_zip_series_prefers_zip_zcta_over_zcta5ce10():
    """When both ZIP_zcta and ZCTA5CE10 exist, ZIP_zcta takes precedence."""
    gdf = gpd.GeoDataFrame({
        "ZIP_zcta": ["12345"],
        "ZCTA5CE10": ["99999"],
        "geometry": [Point(0, 0)],
    })
    result = _zcta_zip_series(gdf)
    assert list(result) == ["12345"]


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


def test_filter_recommended_handles_nan_prob_gracefully():
    """NaN in predicted_prob is excluded by the filter."""
    df = pd.DataFrame({
        "Latitude": [47.6, 47.61],
        "Longitude": [-122.3, -122.31],
        "predicted_prob": [float("nan"), 0.8],
        "ZIP": ["Cell 0", "Cell 1"],
    })
    result = filter_recommended_by_probability(df, 0.5)
    assert len(result) == 1
    assert result["predicted_prob"].iloc[0] == 0.8


def test_filter_recommended_edge_threshold_zero_keeps_all():
    """Edge: prob_min=0 keeps all rows including prob=0."""
    df = pd.DataFrame({
        "Latitude": [47.6, 47.61],
        "Longitude": [-122.3, -122.31],
        "predicted_prob": [0.0, 0.5],
        "ZIP": ["Cell 0", "Cell 1"],
    })
    result = filter_recommended_by_probability(df, 0.0)
    assert len(result) == 2
    assert 0.0 in result["predicted_prob"].values


def test_filter_recommended_edge_threshold_one():
    """Edge: prob_min=1.0 keeps only rows with prob exactly 1.0."""
    df = pd.DataFrame({
        "Latitude": [47.6, 47.61, 47.62],
        "Longitude": [-122.3, -122.31, -122.32],
        "predicted_prob": [0.9, 1.0, 0.99],
        "ZIP": ["Cell 0", "Cell 1", "Cell 2"],
    })
    result = filter_recommended_by_probability(df, 1.0)
    assert len(result) == 1
    assert result["predicted_prob"].iloc[0] == 1.0


def test_filter_recommended_preserves_all_columns():
    """Edge: filter keeps all input columns, does not drop any."""
    df = pd.DataFrame({
        "Latitude": [47.6, 47.61],
        "Longitude": [-122.3, -122.31],
        "predicted_prob": [0.4, 0.7],
        "ZIP": ["Cell 0", "Cell 1"],
        "Location": ["A", "B"],
    })
    result = filter_recommended_by_probability(df, 0.5)
    assert list(result.columns) == list(df.columns)
    assert result["Location"].iloc[0] == "B"


def test_filter_recommended_does_not_mutate_input():
    """Edge: filter returns a copy; input DataFrame is unchanged."""
    df = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "predicted_prob": [0.8],
        "ZIP": ["Cell 0"],
    })
    original_len = len(df)
    result = filter_recommended_by_probability(df, 0.5)
    assert len(df) == original_len
    assert result is not df


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
        assert sorted(result["ZIP"].tolist()) == ["Cell 0", "Cell 1"]
        assert result["predicted_prob"].tolist() == [0.9, 0.75]  # sorted desc by prob
        assert sorted(result["Latitude"].tolist()) == [47.6, 47.61]
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


def test_load_recommended_raises_on_malformed_csv():
    """Malformed CSV (invalid UTF-8) causes pd.read_csv to raise."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        f.write(b"Latitude,Longitude\n")
        f.write(b"\xff\xfe\xfd")  # Invalid UTF-8
        path = f.name
    try:
        with pytest.raises((UnicodeDecodeError, pd.errors.ParserError)):
            load_recommended_from_csv(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_csv_missing_required_columns_returns_empty():
    """CSV missing Latitude or Longitude returns empty DataFrame (required columns guard)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("foo,bar\n")
        f.write("1,2\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert result.empty
        assert list(result.columns) == ["Latitude", "Longitude", "ZIP", "predicted_prob", "Location"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_default_location_when_missing():
    """When Location column is missing, uses 'Recommended' as default."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob\n")
        f.write("47.6,-122.3,0.7\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 1
        assert result["Location"].iloc[0] == "Recommended"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_default_predicted_prob_when_missing():
    """When predicted_prob column is missing, uses 0.5 as default."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude\n")
        f.write("47.6,-122.3\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 1
        assert result["predicted_prob"].iloc[0] == 0.5
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_sorts_by_prob_descending():
    """Output is sorted by predicted_prob descending."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,cell_id\n")
        f.write("47.6,-122.3,0.3,0\n")
        f.write("47.61,-122.31,0.9,1\n")
        f.write("47.62,-122.32,0.5,2\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert result["predicted_prob"].tolist() == [0.9, 0.5, 0.3]
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_recommended_one_row_with_cell_id():
    """One-shot: single row with cell_id maps to correct ZIP label."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,Location,cell_id\n")
        f.write("47.6062,-122.3321,0.85,Sample location,42\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 1
        assert result["ZIP"].iloc[0] == "Cell 42"
        assert result["predicted_prob"].iloc[0] == 0.85
    finally:
        Path(path).unlink(missing_ok=True)


def test_integration_load_csv_then_filter_by_probability():
    """One-shot: load CSV → filter by probability → correct shape and columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,Location,cell_id\n")
        f.write("47.60,-122.33,0.4,Low,0\n")
        f.write("47.61,-122.32,0.6,Mid,1\n")
        f.write("47.62,-122.31,0.9,High,2\n")
        path = f.name
    try:
        loaded = load_recommended_from_csv(path)
        assert len(loaded) == 3
        filtered = filter_recommended_by_probability(loaded, 0.6)
        assert len(filtered) == 2
        assert set(filtered["Location"]) == {"Mid", "High"}
        assert list(filtered.columns) == list(loaded.columns)
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
    assert lats == [0, 1, None, 2, 3, None]


def test_geoms_to_plotly_skips_none_and_empty():
    """None and empty geometries are skipped."""
    from shapely.geometry import LineString
    geom = LineString([(0, 0), (1, 1)])
    lons, lats = geoms_to_plotly([None, geom, LineString()])
    assert lons == [0, 1, None]
    assert lats == [0, 1, None]


def test_geoms_to_plotly_edge_all_invalid_returns_none():
    """Edge: iterable of only None/empty geometries returns (None, None)."""
    from shapely.geometry import LineString
    result = geoms_to_plotly([None, LineString(), None])
    assert result == (None, None)


def test_geoms_to_plotly_raises_on_invalid_geometry_type():
    """Non-geometry (e.g. int) raises AttributeError."""
    with pytest.raises(AttributeError):
        geoms_to_plotly([123])


def test_geoms_to_plotly_raises_on_string_instead_of_geom():
    """String instead of geometry raises AttributeError."""
    with pytest.raises(AttributeError):
        geoms_to_plotly(["not a geometry"])


def test_geoms_to_plotly_multiple_geometries_in_list():
    """Edge: multiple LineStrings in one list are concatenated with None separators."""
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(2, 2), (3, 3)])
    line3 = LineString([(4, 4), (5, 5)])
    lons, lats = geoms_to_plotly([line1, line2, line3])
    assert lons == [0, 1, None, 2, 3, None, 4, 5, None]
    assert lats == [0, 1, None, 2, 3, None, 4, 5, None]


# ── get_recommended_stations ──────────────────────────────────────────────────

@patch("ml_model.load_recommended_from_csv")
def test_get_recommended_stations_returns_loaded_data(mock_load):
    """get_recommended_stations returns whatever load_recommended_from_csv returns."""
    expected = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "ZIP": ["Cell 0"],
        "predicted_prob": [0.9],
        "Location": ["Grid cell 0"],
    })
    mock_load.return_value = expected
    result = get_recommended_stations()
    assert len(result) == 1
    assert result["predicted_prob"].iloc[0] == 0.9


@patch("ml_model.load_recommended_from_csv")
def test_get_recommended_stations_empty_when_no_data(mock_load):
    """One-shot: when CSV missing or empty, get_recommended_stations returns empty."""
    mock_load.return_value = EMPTY_RECOMMENDED.copy()
    result = get_recommended_stations()
    assert result.empty
    assert list(result.columns) == ["Latitude", "Longitude", "ZIP", "predicted_prob", "Location"]


# ── get_electric_lines_for_map ────────────────────────────────────────────────

@patch("ml_model.geoms_to_plotly")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_get_electric_lines_missing_cache_returns_none(mock_isfile, mock_read, mock_geoms):
    """When cache file does not exist, returns (None, None)."""
    mock_isfile.return_value = False
    result = get_electric_lines_for_map()
    assert result == (None, None)
    mock_read.assert_not_called()
    mock_geoms.assert_not_called()


@patch("ml_model.geoms_to_plotly")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_get_electric_lines_valid_cache_returns_oh_ug_tuples(mock_isfile, mock_read, mock_geoms):
    """When cache has data, returns (oh_lons/lats, ug_lons/lats) from geoms_to_plotly."""
    import geopandas as gpd
    mock_isfile.return_value = True
    mock_read.return_value = gpd.GeoDataFrame({
        "geometry": [None],
        "ConductorType1": ["OH"],
    })
    mock_geoms.side_effect = [([0, 1, None], [2, 3, None]), (None, None)]
    oh, ug = get_electric_lines_for_map()
    assert oh == ([0, 1, None], [2, 3, None])
    assert ug == (None, None)


@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_get_electric_lines_empty_cache_returns_none(mock_isfile, mock_read):
    """Edge: cache file exists but is empty (e.g. corrupted or just created)."""
    import geopandas as gpd
    mock_isfile.return_value = True
    mock_read.return_value = gpd.GeoDataFrame()
    result = get_electric_lines_for_map()
    assert result == (None, None)


@patch("ml_model.geoms_to_plotly")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_get_electric_lines_missing_conductor_type_uses_all_geoms(
    mock_isfile, mock_read, mock_geoms
):
    """Edge: when ConductorType1 column is missing, all geometries go to OH; UG is empty."""
    mock_isfile.return_value = True
    mock_read.return_value = gpd.GeoDataFrame({
        "geometry": [LineString([(0, 0), (1, 1)])],
    })
    mock_geoms.side_effect = [([0, 1, None], [0, 1, None]), (None, None)]
    oh, ug = get_electric_lines_for_map()
    assert oh == ([0, 1, None], [0, 1, None])
    assert ug == (None, None)
    assert mock_geoms.call_count == 2


@patch("ml_model.geoms_to_plotly")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_get_electric_lines_both_oh_and_ug_present(mock_isfile, mock_read, mock_geoms):
    """When cache has both OH and UG lines, returns both tuples."""
    mock_isfile.return_value = True
    mock_read.return_value = gpd.GeoDataFrame({
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
        ],
        "ConductorType1": ["OH", "UG"],
    })
    mock_geoms.side_effect = [
        ([0, 1, None], [0, 1, None]),
        ([2, 3, None], [2, 3, None]),
    ]
    oh, ug = get_electric_lines_for_map()
    assert oh == ([0, 1, None], [0, 1, None])
    assert ug == ([2, 3, None], [2, 3, None])


# ── generate_electric_lines_cache ────────────────────────────────────────────

@patch("ml_model.os.path.isfile")
def test_generate_electric_lines_cache_skips_when_cache_exists(mock_isfile):
    """When cache file exists, returns early without building."""
    mock_isfile.return_value = True
    with patch("ml_model.gpd.read_file") as mock_read:
        generate_electric_lines_cache()
        mock_read.assert_not_called()


@patch("ml_model.os.makedirs")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_generate_electric_lines_cache_builds_when_missing(
    mock_isfile, mock_read, mock_makedirs
):
    """One-shot: when cache missing, builds cache (mocked geo data)."""
    from shapely.geometry import Polygon
    mock_isfile.return_value = False
    seattle = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    line = LineString([(0.5, 0.5), (0.6, 0.6)])
    mock_read.side_effect = [
        gpd.GeoDataFrame({"geometry": [seattle]}, crs="EPSG:2285"),
        gpd.GeoDataFrame({"geometry": [line], "ConductorType1": ["OH"]}, crs="EPSG:2285"),
    ]
    with patch("ml_model.gpd.clip") as mock_clip:
        mock_clip.return_value = gpd.GeoDataFrame({"geometry": [line]}, crs="EPSG:4326")
        with patch.object(gpd.GeoDataFrame, "to_file"):
            generate_electric_lines_cache()
    mock_makedirs.assert_called_once()
    assert mock_read.call_count >= 2


@patch("ml_model.os.makedirs")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_generate_electric_lines_cache_returns_early_when_lines_empty(
    mock_isfile, mock_read, mock_makedirs
):
    """Edge: when no lines intersect Seattle, returns early without writing cache."""
    from shapely.geometry import Polygon
    mock_isfile.return_value = False
    seattle = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    line_far = LineString([(100, 100), (101, 101)])
    mock_read.side_effect = [
        gpd.GeoDataFrame({"geometry": [seattle]}, crs="EPSG:2285"),
        gpd.GeoDataFrame({"geometry": [line_far], "ConductorType1": ["OH"]}, crs="EPSG:2285"),
    ]
    with patch.object(gpd.GeoDataFrame, "to_file") as mock_to_file:
        generate_electric_lines_cache()
        mock_to_file.assert_not_called()


@patch("ml_model.os.makedirs")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_generate_electric_lines_cache_handles_exception(mock_isfile, mock_read, mock_makedirs):
    """Exception: when geo processing fails, catches exception and does not raise."""
    mock_isfile.return_value = False
    mock_read.side_effect = OSError("Geo file not found")
    generate_electric_lines_cache()


# ── _prepare_features ────────────────────────────────────────────────────────

@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_prepare_features_empty_merge_returns_zeros(mock_power, mock_roads, mock_zoning):
    """When all feature loaders return empty, _prepare_features fills with zeros."""
    import ml_model as m
    m._feature_cache.clear()
    mock_power.return_value = pd.DataFrame(
        columns=["ZIP", "total_power_line_length", "pct_underground_power"]
    )
    mock_roads.return_value = pd.DataFrame(columns=["ZIP", "dist_to_major_road"])
    mock_zoning.return_value = pd.DataFrame(columns=["ZIP", "pct_multifamily"])
    scored = pd.DataFrame({"ZIP": ["98105"], "station_count": [1]})
    result = _prepare_features(scored)
    assert len(result) == 1
    assert result["total_power_line_length"].iloc[0] == 0
    assert result["pct_underground_power"].iloc[0] == 0
    assert result["dist_to_major_road"].iloc[0] == 0
    assert result["pct_multifamily"].iloc[0] == 0


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_prepare_features_handles_nan_with_fillna(mock_power, mock_roads, mock_zoning):
    """NaN in merged features are filled with 0."""
    import ml_model as m
    m._feature_cache.clear()
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105"],
        "total_power_line_length": [100],
        "pct_underground_power": [0.5],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105"],
        "dist_to_major_road": [float("nan")],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105"],
        "pct_multifamily": [0.2],
    })
    scored = pd.DataFrame({"ZIP": ["98105"], "station_count": [1]})
    result = _prepare_features(scored)
    assert len(result) == 1
    assert result["dist_to_major_road"].iloc[0] == 0


# ── _get_zip_centroids ──────────────────────────────────────────────────────

@patch("ml_model.gpd.read_file")
def test_get_zip_centroids_returns_lat_lon(mock_read):
    """_get_zip_centroids returns ZIP, Latitude, Longitude from ZCTA centroids."""
    import ml_model as m
    m._feature_cache.clear()
    zcta = gpd.GeoDataFrame({
        "ZCTA5CE10": ["98105", "98122"],
        "geometry": [Point(-122.3, 47.6), Point(-122.32, 47.61)],
    }, crs="EPSG:4326")
    zcta["centroid"] = zcta.geometry
    zcta["Longitude"] = zcta["centroid"].x
    zcta["Latitude"] = zcta["centroid"].y
    mock_read.return_value = zcta
    result = _get_zip_centroids()
    assert list(result.columns) == ["ZIP", "Latitude", "Longitude"]
    assert list(result["ZIP"]) == ["98105", "98122"]
    assert len(result) == 2


@patch("ml_model.gpd.read_file")
def test_get_zip_centroids_uses_cache(mock_read):
    """_get_zip_centroids uses cache on second call."""
    import ml_model as m
    m._feature_cache.clear()
    zcta = gpd.GeoDataFrame({
        "ZCTA5CE10": ["98105"],
        "geometry": [Point(-122.3, 47.6)],
    }, crs="EPSG:4326")
    zcta["centroid"] = zcta.geometry
    zcta["Longitude"] = zcta["centroid"].x
    zcta["Latitude"] = zcta["centroid"].y
    mock_read.return_value = zcta
    _get_zip_centroids()
    _get_zip_centroids()
    assert mock_read.call_count == 1


# ── get_zip_predictions ──────────────────────────────────────────────────────

def test_get_zip_predictions_empty_dataframe():
    """Empty scored_df returns empty DataFrame with ZIP and predicted_prob columns."""
    result = get_zip_predictions(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == ["ZIP", "predicted_prob"]


def test_get_zip_predictions_missing_station_count_column():
    """scored_df without station_count returns empty DataFrame."""
    scored = pd.DataFrame({"ZIP": ["98105"], "mean_ADT": [1000]})
    result = get_zip_predictions(scored)
    assert result.empty
    assert list(result.columns) == ["ZIP", "predicted_prob"]


@patch("ml_model._prepare_features")
@patch("ml_model._get_or_train_model")
def test_get_zip_predictions_single_class_model_returns_zeros(mock_get_model, mock_prepare):
    """Edge: when model predicts only class 0 (prob_arr 1 col, 1 not in classes_), probs are zeros."""
    import numpy as np
    scored = pd.DataFrame({"ZIP": ["98105", "98122"], "station_count": [0, 0]})
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.9], [0.8]])
    mock_model.classes_ = np.array([0])
    mock_get_model.return_value = (mock_model, MagicMock())
    mock_prepare.return_value = pd.DataFrame({
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
        "dist_to_major_road": [50, 100],
        "pct_multifamily": [0.2, 0.4],
    })
    result = get_zip_predictions(scored)
    assert len(result) == 2
    assert list(result["predicted_prob"]) == [0.0, 0.0]


@patch("ml_model.gpd.read_file")
def test_load_power_line_features_exception_returns_empty(mock_read):
    """Exception: when geo files fail to load, _load_power_line_features returns empty DataFrame."""
    import ml_model as m
    m._feature_cache.clear()
    mock_read.side_effect = OSError("File not found")
    result = m._load_power_line_features_by_zip()
    assert result.empty
    assert list(result.columns) == ["ZIP", "total_power_line_length", "pct_underground_power"]


@patch("ml_model._prepare_features")
@patch("ml_model._get_or_train_model")
def test_get_zip_predictions_returns_probs_for_each_zip(mock_get_model, mock_prepare):
    """get_zip_predictions returns predicted_prob for each ZIP when model predicts."""
    import numpy as np
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "station_count": [2, 0],
    })
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
    mock_model.classes_ = np.array([0, 1])
    mock_get_model.return_value = (mock_model, MagicMock())
    mock_prepare.return_value = pd.DataFrame({
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
        "dist_to_major_road": [50, 100],
        "pct_multifamily": [0.2, 0.4],
    })
    result = get_zip_predictions(scored)
    assert len(result) == 2
    assert list(result["ZIP"]) == ["98105", "98122"]
    assert result["predicted_prob"].tolist() == [0.8, 0.3]
    mock_get_model.assert_called_once()
    mock_prepare.assert_called()


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_trains_and_returns_valid_probs(
    mock_power, mock_roads, mock_zoning
):
    """
    One-shot: get_zip_predictions with mocked feature loaders trains model and
    returns valid probabilities. Exercises real _train_model and _prepare_features.
    """
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "station_count": [1, 0],
    })
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "dist_to_major_road": [50.0, 100.0],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "pct_multifamily": [0.2, 0.4],
    })
    result = get_zip_predictions(scored)
    assert len(result) == 2
    assert list(result["ZIP"]) == ["98105", "98122"]
    assert "predicted_prob" in result.columns
    assert all(0 <= p <= 1 for p in result["predicted_prob"])


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_handles_missing_features_gracefully(
    mock_power, mock_roads, mock_zoning
):
    """
    Exception: when feature loaders return empty (e.g. geo files missing),
    get_zip_predictions still returns valid output with zeros for features.
    """
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105"],
        "station_count": [1],
    })
    mock_power.return_value = pd.DataFrame(
        columns=["ZIP", "total_power_line_length", "pct_underground_power"]
    )
    mock_roads.return_value = pd.DataFrame(columns=["ZIP", "dist_to_major_road"])
    mock_zoning.return_value = pd.DataFrame(columns=["ZIP", "pct_multifamily"])
    result = get_zip_predictions(scored)
    assert len(result) == 1
    assert result["ZIP"].iloc[0] == "98105"
    assert 0 <= result["predicted_prob"].iloc[0] <= 1


# ── ML model: determinism, probability bounds, model selection ─────────────────

@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_deterministic(mock_power, mock_roads, mock_zoning):
    """Same input yields same output (random_state=42)."""
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "station_count": [1, 0],
    })
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "dist_to_major_road": [50.0, 100.0],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "pct_multifamily": [0.2, 0.4],
    })
    result1 = get_zip_predictions(scored)
    result2 = get_zip_predictions(scored)
    pd.testing.assert_frame_equal(result1, result2)


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_all_probs_in_valid_range(mock_power, mock_roads, mock_zoning):
    """All predicted_prob values are in [0, 1]."""
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122", "98101"],
        "station_count": [1, 0, 1],
    })
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122", "98101"],
        "total_power_line_length": [100, 200, 50],
        "pct_underground_power": [0.5, 0.3, 0.8],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122", "98101"],
        "dist_to_major_road": [50.0, 100.0, 25.0],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122", "98101"],
        "pct_multifamily": [0.2, 0.4, 0.6],
    })
    result = get_zip_predictions(scored)
    assert all(0 <= p <= 1 for p in result["predicted_prob"])


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_model_selection_all_stations(mock_power, mock_roads, mock_zoning):
    """When all ZIPs have stations, DummyClassifier path: all probs = 1."""
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "station_count": [2, 1],
    })
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "dist_to_major_road": [50.0, 100.0],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "pct_multifamily": [0.2, 0.4],
    })
    result = get_zip_predictions(scored)
    assert list(result["predicted_prob"]) == [1.0, 1.0]


@patch("ml_model._load_zoning_features_by_zip")
@patch("ml_model._load_road_features_by_zip")
@patch("ml_model._load_power_line_features_by_zip")
def test_get_zip_predictions_model_selection_no_stations(mock_power, mock_roads, mock_zoning):
    """When no ZIPs have stations, DummyClassifier path: all probs = 0."""
    import ml_model as m
    m._model_cache.clear()
    m._feature_cache.clear()
    scored = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "station_count": [0, 0],
    })
    mock_power.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
    })
    mock_roads.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "dist_to_major_road": [50.0, 100.0],
    })
    mock_zoning.return_value = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "pct_multifamily": [0.2, 0.4],
    })
    result = get_zip_predictions(scored)
    assert list(result["predicted_prob"]) == [0.0, 0.0]


# ── Geospatial: coordinate order, bounds ──────────────────────────────────────

def test_geoms_to_plotly_coordinate_order_lon_lat():
    """Geoms use (lon, lat) order; geoms_to_plotly returns lons, lats correctly."""
    from shapely.geometry import LineString
    geom = LineString([(-122.33, 47.61), (-122.32, 47.62)])
    lons, lats = geoms_to_plotly([geom])
    assert lons[0] == -122.33
    assert lats[0] == 47.61
    assert lons[1] == -122.32
    assert lats[1] == 47.62


def test_load_recommended_coordinates_in_seattle_bounds():
    """Loaded recommended locations have lat/lon within Seattle extent."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Latitude,Longitude,predicted_prob,Location,cell_id\n")
        f.write("47.6062,-122.3321,0.85,Test,0\n")
        path = f.name
    try:
        result = load_recommended_from_csv(path)
        assert len(result) == 1
        lat, lon = result["Latitude"].iloc[0], result["Longitude"].iloc[0]
        assert 47.4 <= lat <= 47.8
        assert -122.6 <= lon <= -122.1
    finally:
        Path(path).unlink(missing_ok=True)


# ── generate_grid_recommendations ────────────────────────────────────────────

@patch("ml_model.generate_electric_lines_cache")
@patch("ml_model.pd.read_csv")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_generate_grid_recommendations_uses_cache_when_exists(
    mock_isfile, mock_read, mock_csv, mock_elec
):
    """When grid cache exists, loads from cache and skips grid building."""
    import ml_model as m
    mock_isfile.side_effect = lambda p: "grid" in str(p).lower()
    grid = gpd.GeoDataFrame({
        "geometry": [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        "cell_id": [0, 1],
        "total_power_line_length": [100, 200],
        "pct_underground_power": [0.5, 0.3],
        "dist_to_major_road": [50, 100],
        "pct_multifamily": [0.2, 0.4],
    }, crs="EPSG:2285")
    mock_read.return_value = grid
    mock_csv.return_value = pd.DataFrame({
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "City": ["Seattle"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(m, "GRID_CACHE_PATH", Path(tmp) / "grid.geojson"):
            with patch.object(m, "GRID_RECOMMENDED_PATH", Path(tmp) / "rec.csv"):
                with patch.object(m, "EV_STATIONS_PATH", "/dev/null"):
                    generate_grid_recommendations()
        out_path = Path(tmp) / "rec.csv"
        assert out_path.exists()
        result = pd.read_csv(out_path)
        assert "Latitude" in result.columns
        assert "Longitude" in result.columns


@patch("ml_model.generate_electric_lines_cache")
@patch("ml_model.pd.read_csv")
@patch("ml_model.gpd.read_file")
@patch("ml_model.os.path.isfile")
def test_generate_grid_recommendations_empty_recommended_when_all_have_stations(
    mock_isfile, mock_read, mock_csv, mock_elec
):
    """When all high-prob cells have stations, outputs empty recommended CSV."""
    import ml_model as m
    mock_isfile.side_effect = lambda p: "grid_with_features" in p or "recommended" in p
    grid = gpd.GeoDataFrame({
        "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        "cell_id": [0],
        "total_power_line_length": [100],
        "pct_underground_power": [0.5],
        "dist_to_major_road": [50],
        "pct_multifamily": [0.2],
    }, crs="EPSG:2285")
    mock_read.return_value = grid
    mock_csv.return_value = pd.DataFrame(columns=["Latitude", "Longitude"])
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(m, "GRID_CACHE_PATH", Path(tmp) / "grid.geojson"):
            with patch.object(m, "GRID_RECOMMENDED_PATH", Path(tmp) / "rec.csv"):
                with patch.object(m, "EV_STATIONS_PATH", "/dev/null"):
                    generate_grid_recommendations()
        result = pd.read_csv(Path(tmp) / "rec.csv")
        assert result.empty or len(result) == 0


# ── geoms_to_plotly additional edge cases ─────────────────────────────────────

def test_geoms_to_plotly_point_geometry():
    """Point geometry is supported (single coord, then None separator)."""
    from shapely.geometry import Point as ShapelyPoint
    geom = ShapelyPoint(-122.3, 47.6)
    lons, lats = geoms_to_plotly([geom])
    assert lons == [-122.3, None]
    assert lats == [47.6, None]


def test_geoms_to_plotly_polygon_raises_or_works():
    """Polygon has coords - geoms_to_plotly iterates boundary coords."""
    from shapely.geometry import Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    lons, lats = geoms_to_plotly([poly])
    assert lons is not None
    assert len(lons) > 0
    assert lons[-1] is None


# ── get_recommended_stations path ────────────────────────────────────────────

@patch("ml_model.load_recommended_from_csv")
def test_get_recommended_stations_passes_correct_path(mock_load):
    """get_recommended_stations calls load_recommended_from_csv with GRID_RECOMMENDED_PATH."""
    import ml_model as m
    mock_load.return_value = EMPTY_RECOMMENDED.copy()
    get_recommended_stations()
    mock_load.assert_called_once_with(m.GRID_RECOMMENDED_PATH)


# ── EMPTY_RECOMMENDED constant ───────────────────────────────────────────────

def test_empty_recommended_has_expected_columns():
    """EMPTY_RECOMMENDED has correct column names."""
    assert list(EMPTY_RECOMMENDED.columns) == [
        "Latitude", "Longitude", "ZIP", "predicted_prob", "Location"
    ]
    assert EMPTY_RECOMMENDED.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
