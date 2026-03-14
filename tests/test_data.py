
"""
Basic tests for Seattle EV Explorer app
Coupled with basic tests for eval and map-building.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from shapely.geometry import Point, Polygon

# Add the interactive_map directory to path so we can import app_v2
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))

from app_v2 import (
    ADT_BINS, ADT_LABELS, ADT_COLORS, ADT_WIDTHS,
    aggregate_ev_by_zip, build_master, build_geojson,
    single_zip_geojson, zip_centroid,
)


def test_imports():
    """Test that all required packages can be imported"""
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go
    import streamlit as st
    assert True


def test_adt_bins_configuration():
    """Test that ADT bins are properly configured"""
    # ADT_BINS has one more element than labels (bin edges vs bin labels)
    assert len(ADT_BINS) == len(ADT_LABELS) + 1
    assert len(ADT_LABELS) == 5
    assert len(ADT_COLORS) == 5
    assert len(ADT_WIDTHS) == 5


def test_aggregate_ev_by_zip():
    """Test EV aggregation function"""
    df = pd.DataFrame({
        'ZIP': ['98101', '98101', '98102'],
        'Station Name': ['Station A', 'Station B', 'Station C'],
        'EV Level2 EVSE Num': [5, 10, 3],
        'EV DC Fast Count': [2, 1, 0]
    })

    result = aggregate_ev_by_zip(df)

    assert len(result) == 2  # Two unique ZIPs
    assert 'station_count' in result.columns
    assert 'level2_spots' in result.columns
    assert 'dcfast_count' in result.columns

    # Check ZIP 98101 has 2 stations
    zip_98101 = result[result['ZIP'] == '98101']
    assert zip_98101['station_count'].values[0] == 2
    assert zip_98101['level2_spots'].values[0] == 15  # 5 + 10


def test_build_master():
    """Test master dataframe building"""
    ev_data = pd.DataFrame({
        'ZIP': ['98101', '98102'],
        'station_count': [5, 3],
        'level2_spots': [20, 10],
        'dcfast_count': [2, 1]
    })

    traffic_data = pd.DataFrame({
        'ZIP': ['98101', '98102', '98103'],
        'mean_ADT': [5000, 3000, 2000],
        'population': [10000, 8000, 6000]
    })

    result = build_master(ev_data, traffic_data)

    assert len(result) == 3  # Should have all ZIPs from traffic_data
    assert result[result['ZIP'] == '98103']['station_count'].values[0] == 0  # Filled with 0


def test_build_geojson():
    """Test GeoJSON building function"""
    sample_gdf = gpd.GeoDataFrame({
        'ZIP_zcta': ['98101', '98102', '98103'],
        'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        ]
    })

    zip_set = {'98101', '98102'}
    result = build_geojson(sample_gdf, zip_set)

    assert result['type'] == 'FeatureCollection'
    assert len(result['features']) == 2  # Only requested ZIPs


# zip_centroid
def test_zip_centroid_valid_search():
    """Checks if the centroid dictionary is correctly extracted."""
    test_zip_code = "98105"
    mock_gdf = gpd.GeoDataFrame({
        "ZIP_zcta": [test_zip_code],
        "geometry": [Point(-122.3, 47.6)]
    }, crs="EPSG:4326")

    ret_val = zip_centroid(mock_gdf, test_zip_code)

    assert ret_val == {"lat": 47.6, "lon": -122.3}


def test_zip_centroid_fallback_case():
    """Checks if the Seattle default is returned when the ZIP is missing."""
    mock_gdf = gpd.GeoDataFrame({"ZIP_zcta": [], "geometry": []}, crs="EPSG:4326")
    ret_val = zip_centroid(mock_gdf, "00000")
    assert ret_val == {"lat": 47.61, "lon": -122.33}


# single_zip_geojson
def test_single_zip_geojson_none_case():
    """Checks if None is returned if the dataframe is entirely empty."""
    test_zip_code = "98105"
    mock_gdf = gpd.GeoDataFrame({"ZIP_zcta": [], "geometry": []}, crs="EPSG:4326")
    ret_val = single_zip_geojson(mock_gdf, test_zip_code)
    assert ret_val is None


def test_single_zip_geojson_no_match_case():
    """Checks if None is returned if the ZIP is not in the dataframe."""
    test_zip_code = "98105"
    mock_gdf = gpd.GeoDataFrame({
        "ZIP_zcta": ["99999"],
        "geometry": [Point(0, 0)]
    }, crs="EPSG:4326")
    ret_val = single_zip_geojson(mock_gdf, test_zip_code)
    assert ret_val is None


def test_single_zip_geojson_valid_case():
    """Checks if a proper GeoJSON dictionary is returned for a valid match."""
    import streamlit as st
    st.cache_data.clear()  # Prevent cache pollution from earlier tests with same zip code
    test_zip_code = "98105"
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mock_gdf = gpd.GeoDataFrame({
        "ZIP_zcta": [test_zip_code],
        "geometry": [square]
    }, crs="EPSG:4326")
    ret_val = single_zip_geojson(mock_gdf, test_zip_code)
    assert ret_val is not None
    assert ret_val["type"] == "FeatureCollection"
    assert ret_val["features"][0]["id"] == test_zip_code
    assert "geometry" in ret_val["features"][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
