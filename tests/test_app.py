
"""
Basic tests for Seattle EV Explorer app
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the interactive_map directory to path so we can import app_v2
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))


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
    from app_v2 import ADT_BINS, ADT_LABELS, ADT_COLORS, ADT_WIDTHS
    
    assert len(ADT_BINS) == 5
    assert len(ADT_LABELS) == 5
    assert len(ADT_COLORS) == 5
    assert len(ADT_WIDTHS) == 5


def test_aggregate_ev_by_zip():
    """Test EV aggregation function"""
    from app_v2 import aggregate_ev_by_zip
    
    # Create sample data
    sample_data = pd.DataFrame({
        'ZIP': ['98101', '98101', '98102'],
        'Station Name': ['Station A', 'Station B', 'Station C'],
        'EV Level2 EVSE Num': [5, 10, 3],
        'EV DC Fast Count': [2, 1, 0]
    })
    
    result = aggregate_ev_by_zip(sample_data)
    
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
    from app_v2 import build_master
    
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
    from app_v2 import build_geojson
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Create sample geodataframe
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])