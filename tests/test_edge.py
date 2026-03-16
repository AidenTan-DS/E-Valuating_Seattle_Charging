"""
Critical edge case tests for Seattle EV Explorer app.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from unittest.mock import patch
 
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))
 
from app_v2 import (
    load_ev_stations, fix_missing_zips, aggregate_ev_by_zip,
    aggregate_traffic_from_csv, build_master, get_score_color,
    build_main_map, build_road_map, _line_coords, _midpoint,
    ADT_BINS
)
 
def test_fix_missing_zips_all_outside():
    """CRITICAL: Stations outside all polygons should keep original ZIP"""
    zcta_data = {
        "ZIP_zcta": ["98105"],
        "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    }
    zcta_gdf = gpd.GeoDataFrame(zcta_data, crs="EPSG:4326")
    ev_data = pd.DataFrame({
        "Station Name": ["Remote1", "Remote2"],
        "ZIP": ["00000", "99999"],
        "Latitude": [10.0, 20.0],  # Far outside polygon
        "Longitude": [10.0, 20.0]
    })
    result = fix_missing_zips(ev_data, zcta_gdf)
    # Should remain unchanged
    assert result.iloc[0]["ZIP"] == "00000"
    assert result.iloc[1]["ZIP"] == "99999"
 
 
def test_fix_missing_zips_multipolygon():
    """CRITICAL: MultiPolygon geometries should work correctly"""
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multi = MultiPolygon([poly1, poly2])
    
    zcta_gdf = gpd.GeoDataFrame({
        "ZIP_zcta": ["98105"],
        "geometry": [multi]
    }, crs="EPSG:4326")
    
    ev_data = pd.DataFrame({
        "Station Name": ["InPoly1", "InPoly2"],
        "ZIP": ["00000", "00000"],
        "Latitude": [0.5, 2.5],
        "Longitude": [0.5, 2.5]
    })
    result = fix_missing_zips(ev_data, zcta_gdf)
    # Both should be corrected to 98105
    assert all(result["ZIP"] == "98105")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATION 
# ═════════════════════════════════════════════════════════════════════════════
 
def test_aggregate_ev_all_zero_counts():
    """Zero charging ports should count stations but show 0 ports"""
    df = pd.DataFrame({
        'ZIP': ['98101', '98101', '98102'],
        'Station Name': ['A', 'B', 'C'],
        'EV Level2 EVSE Num': [0, 0, 0],
        'EV DC Fast Count': [0, 0, 0]
    })
    result = aggregate_ev_by_zip(df)
    assert result['level2_spots'].sum() == 0
    assert result['dcfast_count'].sum() == 0
    assert result['station_count'].sum() == 3  # Still count stations
 
 
def test_build_master_completely_disjoint():
    """No overlapping ZIPs should fill with zeros, not crash"""
    ev_data = pd.DataFrame({
        'ZIP': ['98001', '98002'],
        'station_count': [5, 3],
        'level2_spots': [20, 10],
        'dcfast_count': [2, 1]
    })
    traffic_data = pd.DataFrame({
        'ZIP': ['98101', '98102'],
        'mean_ADT': [5000, 3000],
        'population': [10000, 8000]
    })
    result = build_master(ev_data, traffic_data)
    # Should have all traffic ZIPs with 0 stations
    assert len(result) == 2
    assert all(result['station_count'] == 0)
    assert all(result['level2_spots'] == 0)
    assert all(result['dcfast_count'] == 0)
 
 
def test_build_master_empty_traffic():
    """Empty traffic data should return empty result"""
    ev_data = pd.DataFrame({
        'ZIP': ['98101'],
        'station_count': [5],
        'level2_spots': [20],
        'dcfast_count': [2]
    })
    traffic_data = pd.DataFrame(columns=['ZIP', 'mean_ADT', 'population'])
    result = build_master(ev_data, traffic_data)
    assert len(result) == 0  # Left join on empty = empty
 
 
# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS 
# ═════════════════════════════════════════════════════════════════════════════
 
def test_line_coords_minimum_points():
    """Minimum valid LineString (2 points) should not crash"""
    line = LineString([(0, 0), (10, 10)])
    lons, lats = _line_coords(line)
    assert len(lons) == 3  # 2 points + None
    assert lons == [0.0, 10.0, None]
    assert lats == [0.0, 10.0, None]
 
 
def test_midpoint_two_points():
    """Two-point line should pick second point (index 1)"""
    line = LineString([(0, 0), (10, 10)])
    lon, lat = _midpoint(line)
    # With 2 points (indices 0, 1), 2//2 = 1
    assert lon == 10
    assert lat == 10
 
 
# ═════════════════════════════════════════════════════════════════════════════
# SCORING & BINNING 
# ═════════════════════════════════════════════════════════════════════════════
 
def test_score_color_exact_boundaries():
    """Exact boundary values must map to correct color"""
    assert get_score_color(20) == "#ef4444"   # At red/yellow boundary
    assert get_score_color(21) == "#facc15"   # Just above
    assert get_score_color(40) == "#facc15"   # At yellow/green boundary
    assert get_score_color(41) == "#22c55e"   # Just above
 
 
def test_score_color_extreme_values():
    """Extreme values should not crash"""
    assert get_score_color(float('inf')) == "#22c55e"
    assert get_score_color(float('-inf')) == "#ef4444"
    # NaN will fall through comparisons
    result = get_score_color(float('nan'))
    assert result in ["#ef4444", "#facc15", "#22c55e"]
 
 
def test_adt_bins_boundaries():
    """CRITICAL: ADT bin boundaries must be correct for visualization"""
    # Test exact boundary values
    test_cases = [
        (0, 0),          # Zero ADT
        (500, 1),        # First boundary
        (3800, 2),       # Second boundary
        (6700, 3),       # Third boundary
        (12000, 4),      # Fourth boundary
        (999999, 4)      # Very high
    ]
    
    for value, expected_bin in test_cases:
        for i in range(len(ADT_BINS) - 1):
            if ADT_BINS[i] <= value < ADT_BINS[i + 1]:
                assert i == expected_bin, \
                    f"ADT {value} should be in bin {expected_bin}, got {i}"
                break
 
 
# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 
# ═════════════════════════════════════════════════════════════════════════════
 
def test_build_main_map_empty_scored():
    """Empty scored data should not crash"""
    scored_df = pd.DataFrame(columns=["ZIP", "mean_ADT"])
    ev_df = pd.DataFrame({
        "Station Name": ["S1"],
        "ZIP": ["98105"],
        "Latitude": [47.6],
        "Longitude": [-122.3],
        "EV Level2 EVSE Num": [2],
        "EV DC Fast Count": [1],
        "EV Network": ["N"]
    })
    fig = build_main_map(ev_df, scored_df, {}, None)
    # Should create valid figure even with no data
    assert fig is not None
    assert len(fig.data) >= 1
 
 
def test_build_main_map_all_nan_adt():
    """All NaN ADT values should not crash visualization"""
    scored_df = pd.DataFrame({
        "ZIP": ["98105", "98122"],
        "mean_ADT": [np.nan, np.nan]
    })
    ev_df = pd.DataFrame(columns=[
        "ZIP", "Latitude", "Longitude", "Station Name",
        "EV Level2 EVSE Num", "EV DC Fast Count", "EV Network"
    ])
    fig = build_main_map(ev_df, scored_df, {}, None)
    assert fig is not None
 
 
def test_build_main_map_invalid_selected_zip():
    """Selected ZIP not in data should not crash"""
    scored_df = pd.DataFrame({
        "ZIP": ["98105"],
        "mean_ADT": [1000]
    })
    ev_df = pd.DataFrame(columns=[
        "ZIP", "Latitude", "Longitude", "Station Name",
        "EV Level2 EVSE Num", "EV DC Fast Count", "EV Network"
    ])
    fig = build_main_map(ev_df, scored_df, {}, "99999")  # Invalid ZIP
    assert fig.data[0].selectedpoints is None
 
 
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
 