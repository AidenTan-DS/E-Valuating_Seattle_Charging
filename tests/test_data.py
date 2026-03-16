
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
from shapely.geometry import Point, Polygon, LineString
from unittest.mock import patch, MagicMock, PropertyMock

# Add the interactive_map directory to path so we can import app_v2
sys.path.insert(0, str(Path(__file__).parent.parent / "interactive_map"))

from app_v2 import (
    ADT_BINS, ADT_LABELS, ADT_COLORS, ADT_WIDTHS, load_ev_stations, fix_missing_zips,
    aggregate_ev_by_zip, aggregate_traffic_from_csv,  load_zcta, build_master, build_geojson, 
    load_demand_gap, load_all, load_streets_with_adt, cached_geojson, cached_road_fig, 
    single_zip_geojson, zip_centroid,get_eval_map_base, _line_coords, _midpoint, 
    get_score_color, build_main_map, build_road_map, map_fragment
)

# ─────────────────────────────────────────────────────────────────────────────
# Tests for Data functions
# ─────────────────────────────────────────────────────────────────────────────
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

@patch("pandas.read_csv")
def test_load_ev_stations_cleaning_logic(mock_read_csv):
    """
    Verifies that function filters for state=WA and cleans data correctly.
    Creates initial messy mock dataframe with rows to keep and discard, 
    Mock returns dataframe when pd.read_csv() is called. 
    Check if non-matching entries are gone, implement formatting as necessary & fill missing values. 
    """
    mock_data = pd.DataFrame({
        "Station Name": ["Station A", "Station B", "Station C", "Station D"],
        "State": ["WA", "CA", "WA", "WA"],
        "Latitude": [47.6, 34.0, None, 47.1],
        "Longitude": [-122.3, -118.2, -122.5, -122.1],
        "ZIP": [98105, 90210, 98122, "981"], 
        "EV Level2 EVSE Num": [2, 5, 0, None], 
        "EV DC Fast Count": [None, 1, 1, 1],  
        "EV Network": ["Non-Networked", "Tesla", "ChargePoint", "Blink"]
    })
    mock_read_csv.return_value = mock_data
    df = load_ev_stations()

    assert len(df) == 2
    assert all(df["State"] == "WA")
    # '981' should become '00981'
    assert "00981" in df["ZIP"].values
    assert df.loc[df["Station Name"] == "Station D", "EV Level2 EVSE Num"].iloc[0] == 0
    assert df.loc[df["Station Name"] == "Station A", "EV DC Fast Count"].iloc[0] == 0

def test_fix_missing_zips_spatial_correction():
    """
    Tests if invalid ZIPs are corrected based on their spatial location. 
    Sets up mock ZCTA geodataframe & EV stations dataframe. Checks if station is correct and gets corrected for input 
    """
    zcta_data = {
        "ZIP_zcta": ["98105", "98122"],
        "geometry": [
            Polygon([(0,0), (1,0), (1,1), (0,1)]),
            Polygon([(1,0), (2,0), (2,1), (1,1)])
        ]
    }
    zcta_gdf = gpd.GeoDataFrame(zcta_data, crs="EPSG:4326")
    ev_data = pd.DataFrame({
        "Station Name": ["Correct Station", "Lost Station"],
        "ZIP": ["98105", "00000"], # 00000 is invalid
        "Latitude": [0.5, 0.5],     # Both at vertical midpoint 0.5
        "Longitude": [0.5, 1.5]     # 0.5 is in 98105, 1.5 is in 98122
    })
    result_df = fix_missing_zips(ev_data, zcta_gdf)
    # Correct Station should still be 98105
    correct_zip = result_df.loc[result_df["Station Name"] == "Correct Station", "ZIP"].iloc[0]
    # Lost Station should have been corrected to 98122
    corrected_zip = result_df.loc[result_df["Station Name"] == "Lost Station", "ZIP"].iloc[0]
    assert correct_zip == "98105"
    assert corrected_zip == "98122"
    # Ensure the original dataframe wasn't modified (it should return a copy)
    assert ev_data.loc[1, "ZIP"] == "00000"

@patch("pandas.read_csv")
def test_aggregate_ev_by_zip(mock_read):
    """Test EV aggregation function to ensure the number of stations within a zipcode are counted correctly"""
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

@patch("pandas.read_csv")
def test_aggregate_traffic_logic(mock_read_csv):
    """
    Tests if traffic data is correctly cleaned and aggregated by ZIP.
    Creates mock data with duplicates ZIPs, messy strings. 
    Aggregation reduces number of data values
    """
    mock_data = pd.DataFrame({
        "zip_code": [98105, 98105, 98122], # Two entries for 98105
        "avg_daily_flow": [100, 200, 500],  # Mean for 98105 should be 150
        "Population Estimate": [5000, 5000, 2000],
        "city": ["Seattle", "Seattle", "Seattle"],
        "Median Household Income": ["100000", "100000", "**"], # ** is messy
        "Total EV Registrations": [50, 50, "-"]                # - is messy
    })
    mock_read_csv.return_value = mock_data
    result_df = aggregate_traffic_from_csv()
    #  3 initial rows across 2 unique ZIP values. Result should have 2 rows.
    assert len(result_df) == 2
    # Check aggregation for 98105
    row_98105 = result_df[result_df["ZIP"] == "98105"].iloc[0]
    assert row_98105["mean_ADT"] == 150.0
    assert row_98105["population"] == 5000
    # Check cleaning/numeric conversion for 98122
    row_98122 = result_df[result_df["ZIP"] == "98122"].iloc[0]
    # "pd.to_numeric(..., errors='coerce')" turns "**" and "-" into NaN
    assert pd.isna(row_98122["med_income"])
    assert pd.isna(row_98122["ev_registrations"])
    assert all(result_df["ZIP"].str.len() == 5)

@patch("geopandas.read_file")
def test_load_zcta_logic(mock_read_file):
    """
    Tests if ZCTA data is loaded, projected, and area is calculated.
    Creates mock GeoDataFrame in degrees, starting w/ 1-deg x 1-deg square near Seattle
    """
    poly = Polygon([(-122, 47), (-121, 47), (-121, 48), (-122, 48)])
    mock_raw_gdf = gpd.GeoDataFrame({
        "ZCTA5CE10": ["98105"],
        "geometry": [poly]
    }, crs="EPSG:4326")
    mock_read_file.return_value = mock_raw_gdf
    result_gdf = load_zcta("fake/path/to/shapefile.shp")
    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert not result_gdf.empty
    assert "ZIP_zcta" in result_gdf.columns
    assert result_gdf["ZIP_zcta"].iloc[0] == "98105"
    # Check area calculation after projecting to EPSG:2285(feet) and dividing by 27,878,400, 
    # the value should be a positive number representing sq miles.
    assert "area_sq_mi" in result_gdf.columns
    assert result_gdf["area_sq_mi"].iloc[0] > 0
    # Check final CRS
    assert result_gdf.crs == "EPSG:4326"

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
    # Should have all ZIPs from traffic_data
    assert len(result) == 3  
    assert result[result['ZIP'] == '98103']['station_count'].values[0] == 0 


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
    assert len(result['features']) == 2 

# ─────────────────────────────────────────────────────────────────────────────
# Tests for Cached Loaders
# ───────────────────────────────────────────────────────────────────────────── 

@patch("pandas.read_csv")
def test_load_demand_gap(mock_read) -> pd.DataFrame:
    """
    Tests if df containing zip codes and demand gap data gets loaded properly. 
    Create fake data for pd.read_csv to return. See if extra cols get filtered. 
    """
    fake_input = pd.DataFrame({
        "zip_code": [98105, 98122],
        "demand_gap": [10.5, 20.1],
        "extra_col": ["ignore_me", "ignore_me"] 
    })
    mock_read.return_value = fake_input
    result_df = load_demand_gap()

    assert len(result_df.columns) == 2
    assert 'ZIP' in result_df.columns
    assert 'demand_gap' in result_df.columns
    assert 'zip_code' not in result_df.columns
    # Check that ZFILL worked (integers became 5-character strings)
    assert result_df['ZIP'].iloc[0] == "98105"
    assert isinstance(result_df['ZIP'].iloc[0], str)

# Comprehensive amount of patched functions needed for testing:
@patch("app_v2.load_zcta")
@patch("app_v2.load_ev_stations")
@patch("app_v2.fix_missing_zips")
@patch("app_v2.aggregate_ev_by_zip")
@patch("app_v2.aggregate_traffic_from_csv")
@patch("app_v2.build_master")
@patch("app_v2.load_demand_gap")
@patch("streamlit.error")
@patch("streamlit.stop")
def test_load_all_integration(mock_stop, mock_error, mock_load_demand, mock_build_master, 
    mock_agg_traffic, mock_agg_ev, mock_fix_zips, mock_load_ev, mock_load_zcta):
    """
    Tests full orchestration and merging logic of load_all to ensure dataset w/ traffic, demographics, EV stations, 
    pop density & demand gap get returned. Start by making mock dfs for zcta, fix_zips, build_master,load_demand. 
    Check calculation
    """
    # Mock ZCTA: 10 sq miles
    mock_load_zcta.return_value = gpd.GeoDataFrame({
        "ZIP_zcta": ["98105"],
        "area_sq_mi": [10.0],
        "geometry": [None]
    })
    mock_fix_zips.return_value = pd.DataFrame({"ZIP": ["98105"], "Station Name": ["S1"]})
    mock_build_master.return_value = pd.DataFrame({
        "ZIP": ["98105"],
        "population": [1000]
    })
    # Mock Demand Gap: Gap of 5.5
    mock_load_demand.return_value = pd.DataFrame({
        "ZIP": ["98105"],
        "demand_gap": [5.5]
    })
    zcta_gdf, ev_df, scored = load_all()
    # Check that merges happened correctly
    assert "area_sq_mi" in scored.columns
    assert "demand_gap" in scored.columns
    # Check Calculation: pop_density = 1000 / 10 = 100
    assert scored.loc[0, "pop_density"] == 100.0
    # Check Demand Gap merge
    assert scored.loc[0, "demand_gap"] == 5.5
    # Ensure Streamlit error wasn't called
    assert not mock_error.called

@patch("geopandas.read_file")
@patch("pandas.read_csv")
def test_load_streets_with_adt_logic(mock_read_csv, mock_read_file):
    """
    Tests if traffic data is correctly joined to both sides of street segments.
    Creates mock street geometry & mock traffic data (same street name but different ZIPs)
    """
    # Mock street geometry: NE 45TH ST between two ZIPs
    mock_streets = gpd.GeoDataFrame({
        "ORD_STNAME_CONCAT": ["NE 45TH ST"],
        "L_ZIP": [98105],
        "R_ZIP": [98195],
        "ARTERIAL_CODE": [1],
        "geometry": [LineString([(0, 0), (1, 1)])]
    }, crs="EPSG:4326")
    mock_read_file.return_value = mock_streets
    # Mock traffic data (same street name, different ZIPs)
    mock_traffic = pd.DataFrame({
        "zip_code": [98105, 98195],
        "STDY_TITLE_PART": ["NE 45TH ST, east of I-5", "NE 45TH ST, west of 15th"],
        "avg_daily_flow": [1000, 2000]
    })
    mock_read_csv.return_value = mock_traffic
    result_gdf = load_streets_with_adt()
    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert len(result_gdf) == 1
    # Check left side join (98105 -> 1000)
    assert result_gdf.iloc[0]["adt_l"] == 1000
    # Check right side join (98195 -> 2000)
    assert result_gdf.iloc[0]["adt_r"] == 2000
    # Verify ZFILL worked on the street ZIPs
    assert result_gdf.iloc[0]["L_ZIP"] == "98105"
    assert result_gdf.iloc[0]["R_ZIP"] == "98195"

@patch("app_v2.build_geojson")
def test_cached_geojson_conversion(mock_build):
    """
    Verifies that zip_tuple is converted to a set for the inner function.
    Set up mock inputs that use tuple since Streamlit caches need hashable inputs,
    then define inner function's return values. 
    Ensure build_geojson() fn called w/ SET instead of TUPLE
    """
    
    test_zips = ("98105", "98122", "98105")
    mock_gdf = gpd.GeoDataFrame({"geometry": []})
    expected_output = {"type": "FeatureCollection", "features": []}
    mock_build.return_value = expected_output
    result = cached_geojson(mock_gdf, test_zips)
    args, _ = mock_build.call_args
    assert isinstance(args[1], set)
    assert args[1] == {"98105", "98122"}
    assert result == expected_output

def test_cached_geojson_empty_input():
    """
    Checks behavior when an empty tuple is passed.
    """
    mock_gdf = gpd.GeoDataFrame({"geometry": []})
    with patch("app_v2.build_geojson") as mock_build:
        mock_build.return_value = {}
        result = cached_geojson(mock_gdf, ())
        # Verify it passed an empty set
        assert mock_build.call_args[0][1] == set()
        assert result == {}

@patch("app_v2.build_road_map")
def test_cached_road_fig_parameters(mock_build_map):
    """
    Verifies that cached wrapper passes all data to map builder.
    Set up mock input & set dummy return value for map builder, then execute. 
    """
    test_zip = "98105"
    mock_streets = gpd.GeoDataFrame({"geometry": []})
    mock_zcta = gpd.GeoDataFrame({"geometry": []})
    mock_ev = pd.DataFrame({"ZIP": []})
    mock_build_map.return_value = "Mock Plotly Figure"
    result = cached_road_fig(test_zip, mock_streets, mock_zcta, mock_ev)
    # Check if the inner function was called exactly once
    assert mock_build_map.called
    # Verify it was called with correct arguments in order
    args, kwargs = mock_build_map.call_args
    assert args[0] == test_zip
    assert isinstance(args[1], gpd.GeoDataFrame) 
    assert isinstance(args[3], pd.DataFrame)      
    # Verify the return value matches what the builder produced
    assert result == "Mock Plotly Figure"

def test_cached_road_fig_handles_none():
    """
    Ensures function handles empty data gracefully.
    """
    # Wrapper should still allow empty objects to pass through. 
    with patch("app_v2.build_road_map") as mock_build:
        mock_build.return_value = None
        result = cached_road_fig("00000", None, None, None)
        assert result is None

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

@patch("app_v2.build_geojson")
def test_get_eval_map_base_structure(mock_build_geojson):
    """
    Checks if static background map is built with correct ZIPs and GeoJSON.
    Set up mock input & define fake geojson w/ known coordinates
    """
    
    mock_zcta = gpd.GeoDataFrame({"geometry": []})
    test_zips = ("98105", "98122")
    fake_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "98105",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        }]
    }
    mock_build_geojson.return_value = fake_geojson
    fig = get_eval_map_base(mock_zcta, test_zips)

    assert isinstance(fig, go.Figure)
    # The figure should have exactly one trace (the Choroplethmapbox)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert isinstance(trace, go.Choroplethmapbox)
    # Verify the locations (ZIPs) were passed correctly
    assert list(trace.locations) == ["98105", "98122"]                                                                                     
    # Verify the GeoJSON coordinates were absorbed
    # This is how we verify "lat/lon" data is present in the base map
    assert trace.geojson == fake_geojson
    coords = trace.geojson["features"][0]["geometry"]["coordinates"][0]
    assert coords[0] == [0, 0] # Check a specific coordinate

def test_get_eval_map_base_empty():
    """Ensures the function doesn't crash with an empty ZIP tuple."""
    mock_zcta = gpd.GeoDataFrame({"geometry": []})
    with patch("app_v2.build_geojson") as mock_geo:
        mock_geo.return_value = {"type": "FeatureCollection", "features": []}
        fig = get_eval_map_base(mock_zcta, ())
        assert len(fig.data[0].locations) == 0

# ─────────────────────────────────────────────────────────────────────────────
# Testing Helper Fns
# ─────────────────────────────────────────────────────────────────────────────
def test_line_coords_logic():
    """Checks if coordinates are extracted correctly with the None separator."""
    line = LineString([(0, 0), (1, 1)])
    lons, lats = _line_coords(line)
    # Expect [x1, x2, None]
    assert lons == [0.0, 1.0, None]
    assert lats == [0.0, 1.0, None]
    assert len(lons) == 3

def test_midpoint_odd_vertices():
    """Checks if the middle vertex is picked for a line with odd number of points."""
    # Line with 3 points: Midpoint is clearly the second point (5, 5)
    line = LineString([(0, 0), (5, 5), (10, 10)])
    lon, lat = _midpoint(line)

    assert lon == 5.0
    assert lat == 5.0

def test_midpoint_even_vertices():
    """Checks which vertex is picked for a line with an even number of points."""
    # E.g. line with 4 points: Index 4 // 2 = 2. Points are 0, 1, 2, 3.
    # 3rd point should be picked
    line = LineString([(0, 0), (2, 2), (4, 4), (6, 6)])
    lon, lat = _midpoint(line)

    assert lon == 4.0
    assert lat == 4.0

def test_score_color_values():
    """Checks if correct color value is being return based on score"""
    r_test_1 = 20
    r_test_2 = -10
    y_test_1 = 21
    y_test_2 = 40
    g_test_1 = 41
    g_test_2 = 70
    assert get_score_color(r_test_1) == "#ef4444" 
    assert get_score_color(r_test_2) == "#ef4444" 
    assert get_score_color(y_test_1) == "#facc15"
    assert get_score_color(y_test_2) == "#facc15"
    assert get_score_color(g_test_1) == "#22c55e"
    assert get_score_color(g_test_2) == "#22c55e"

# ─────────────────────────────────────────────────────────────────────────────
# Tests for Main map (ZIP choropleth + EV station dots)
# ─────────────────────────────────────────────────────────────────────────────

def test_build_main_map_structure():
    """
    Tests the layers and data mapping of the main evaluation map.
    Sets up mock data, ev_df, geojson
    """
    scored_df = pd.DataFrame({
        "ZIP": ["98105", "98122", "98101"],
        "mean_ADT": [1000, 2000, 3000] 
    })
    ev_df = pd.DataFrame({
        "Station Name": ["Station A", "Station B", "Outside Station"],
        "ZIP": ["98105", "98122", "99999"], 
        "Latitude": [47.6, 47.7, 48.0],
        "Longitude": [-122.3, -122.4, -122.5],
        "EV Level2 EVSE Num": [2, 4, 0],
        "EV DC Fast Count": [1, 1, 0],
        "EV Network": ["Blink", "ChargePoint", "None"]
    })
    mock_geojson = {"type": "FeatureCollection", "features": []}
    fig = build_main_map(ev_df, scored_df, mock_geojson, "98122")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 
    choropleth = fig.data[0]
    assert isinstance(choropleth, go.Choroplethmapbox)
    assert choropleth.zmax == scored_df["mean_ADT"].quantile(0.95)
    # assert choropleth.selectedpoints == [1]
    assert list(choropleth.selectedpoints) == [1]
    # Trace 1: Scatter (EV Stations)
    scatter = fig.data[1]
    assert isinstance(scatter, go.Scattermapbox)
    # Ensure "Outside Station" was filtered out (only 2 stations in valid ZIPs)
    assert len(scatter.lat) == 2
    assert "Station A" in scatter.text
    assert "Outside Station" not in scatter.text
    # Check Customdata - customdata[0] should be the ZIP
    assert scatter.customdata[0][0] == "98105"

def test_build_main_map_no_selection():
    """Verifies map behavior when no ZIP is selected."""
    scored_df = pd.DataFrame({"ZIP": ["98105"], "mean_ADT": [100]})
    ev_df = pd.DataFrame({"ZIP": ["98105"], "Latitude": [47], "Longitude": [-122], 
                          "Station Name": ["S1"], "EV Level2 EVSE Num": [1], 
                          "EV DC Fast Count": [1], "EV Network": ["N"]})
    
    fig = build_main_map(ev_df, scored_df, {}, None)
    
    # selectedpoints should be None if selected_zip is not in list
    assert fig.data[0].selectedpoints is None
# ─────────────────────────────────────────────────────────────────────────────
# Test detail map (real road lines coloured by avg_daily_flow)
# ─────────────────────────────────────────────────────────────────────────────

@patch("app_v2.zip_centroid")
@patch("app_v2.single_zip_geojson")
def test_build_road_map_full_logic(mock_geojson, mock_centroid):
    """
    Tests if the road map builds the boundary trace and filters streets correctly.
    Set up mock inputs, create left & right street outputs
    """
    test_zip = "98105"
    mock_centroid.return_value = {"lat": 47.6, "lon": -122.3}
    mock_geojson.return_value = {"type": "FeatureCollection", "features": []}
    zcta_gdf = gpd.GeoDataFrame({
        "ZIP_zcta": [test_zip],
        "geometry": [None]
    })
    streets_gdf = gpd.GeoDataFrame({
        "L_ZIP": [test_zip, "99999"],
        "R_ZIP": ["98122", "99999"],
        "adt_l": [1000, 0], 
        "adt_r": [0, 0],
        "ORD_STNAME_CONCAT": ["NE 45TH ST", "MAIN ST"],
        "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])]
    }, crs="EPSG:4326")

    ev_df = pd.DataFrame(columns=["ZIP"]) 
    fig = build_road_map(test_zip, streets_gdf, zcta_gdf, ev_df)

    assert isinstance(fig, go.Figure)
    # Verify ZIP boundary trace was added (1st trace added)
    assert len(fig.data) > 0
    assert fig.data[0].name == "ZIP boundary"
    assert list(fig.data[0].locations) == [test_zip]
    # Verify mapbox layout was set
    assert fig.layout.mapbox.center.lat == 47.6
    assert fig.layout.height == 400

@patch("app_v2.zip_centroid")
@patch("app_v2.single_zip_geojson")
@patch("streamlit.caption")
def test_build_road_map_empty_streets(mock_caption, mock_geojson, mock_centroid):
    """
    Tests fallback behavior when no streets match the ZIP.
    """
    test_zip = "98105"
    mock_centroid.return_value = {"lat": 47.6, "lon": -122.3}
    mock_geojson.return_value = None # No boundary
    zcta_gdf = gpd.GeoDataFrame({"ZIP_zcta": [test_zip], "geometry": [None]})
    # No streets match 98105
    streets_gdf = gpd.GeoDataFrame({"L_ZIP": ["00000"], "R_ZIP": ["00000"], "geometry": [None]})
    fig = build_road_map(test_zip, streets_gdf, zcta_gdf, pd.DataFrame())

    # Should trigger the st.caption and return a figure with 0 traces
    assert len(fig.data) == 0
    assert mock_caption.called
    assert mock_caption.call_args[0][0] == "No road data for this ZIP."

@patch("app_v2.load_all")
@patch("app_v2.cached_geojson")
@patch("app_v2.build_main_map")
@patch("app_v2.st.plotly_chart")
@patch("app_v2.st.rerun")

def test_map_fragment_click_event(mock_rerun, mock_chart, mock_build, mock_geojson, mock_load):
    """
    Tests if clicking a ZIP on the map updates session state and triggers a rerun.
    Set up mock data & session state, simulate currently being on map_version 0 w/ no ZIP selected
    After, simulate plotly click event for a given ZIP
    """
    # Setup local "database" for the mock
    test_zip = "98105"
    initial_zip = "98122"
    mock_load.return_value = (
        gpd.GeoDataFrame({"geometry": []}),
        pd.DataFrame(columns=["ZIP"]),            
        pd.DataFrame({"ZIP": [test_zip, initial_zip]})
    )
    mock_geojson.return_value = {}
    mock_build.return_value = MagicMock()

    # Build a clone of Streamlit's Session State
    class FakeSessionState(dict):
        """Dictionary that allows attribute access, exactly like st.session_state"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    # Initialize fake state with the starting data
    fake_state = FakeSessionState({
        "map_version": 0,
        "selected_zip": initial_zip,
        "last_event_key": "initial"
    })
    mock_event = MagicMock()
    mock_event.selection.points = [{"location": test_zip}]
    fake_state["main_map_0"] = mock_event

    with patch("app_v2.st.session_state", fake_state):
        """ 
        Use __wrapped__ to bypass Streamlit engine & execute raw Python logic.
        Accessing map_fragment()function directly involves calling Streamlit's internal wrapper engine.
        When executing test, Streamlit's wrapper realizes no web server is being run, 
        throws warning & aborts execution. 
        """
        map_fragment.__wrapped__()
    
    assert fake_state["selected_zip"] == test_zip
    assert fake_state["last_event_key"] != "initial"
    assert mock_rerun.called

@patch("app_v2.load_all")
@patch("streamlit.session_state", spec=dict)
@patch("streamlit.rerun")
def test_map_fragment_ignores_duplicate_click(mock_rerun, mock_state, mock_load):
    """
    Ensures clicking the same ZIP twice doesn't trigger unnecessary reruns.
    If no changes, no reruns should be called.
    """
    mock_load.return_value = (None, None, pd.DataFrame({"ZIP": ["98105"]}))
    # State already says 98105 is selected and last event was this specific point
    mock_event = MagicMock()
    mock_event.selection.points = [{"location": "98105"}]
    ek = repr(mock_event.selection.points[0])
    state_store = {
        "map_version": 0,
        "selected_zip": "98105",
        "last_event_key": ek,
        "main_map_0": mock_event
    }
    mock_state.get.side_effect = state_store.get
    map_fragment()

    assert not mock_rerun.called

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
