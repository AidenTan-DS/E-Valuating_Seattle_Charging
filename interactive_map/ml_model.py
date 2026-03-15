"""
ml_model.py – EV station placement prediction (matches station_prediction.ipynb).

Creates a 500m grid over Seattle, trains logistic regression, and saves
recommended locations (red dots) for the map in app_v3.

Run from project root:
  python -m interactive_map.ml_model
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Data paths (run from project root so these paths work)
ELECTRIC_PATH = "data/geo/electric_lines.geojson"
STREETS_PATH = "data/geo/seattle_streets.geojson"
ZCTA_PATH = "data/geo/zcta.json"
ZONING_PATH = (
    "data/geo/zoning_data-shp/"
    "9569d188-064e-4b50-8ec3-4a2ae969360d2020328-1-149bzvr.n3i9.shp"
)
NEIGHBORHOODS_PATH = (
    "data/geo/Neighborhood_Map_Atlas_Neighborhoods/"
    "Neighborhood_Map_Atlas_Neighborhoods.shp"
)
EV_STATIONS_PATH = "data/cleaned/ev_station.csv"
GRID_RECOMMENDED_PATH = "data/processed/recommended_grid_locations.csv"
GRID_CACHE_PATH = "data/processed/grid_with_features.geojson"
ELECTRIC_SEATTLE_CACHE = "data/processed/electric_lines_seattle.geojson"

# Coordinate system for grid (Washington State Plane, feet) – same as notebook
GRID_CRS = "EPSG:2285"

_model_cache = {}
_feature_cache = {}

FEATURE_COLS = [
    "total_power_line_length",
    "pct_underground_power",
    "dist_to_major_road",
    "pct_multifamily",
]


def _zcta_zip_series(zcta: gpd.GeoDataFrame):
    """Get ZIP series from ZCTA (ZIP_zcta, ZCTA5CE10, or GEOID10)."""
    if "ZIP_zcta" in zcta.columns:
        return zcta["ZIP_zcta"].astype(str).str.zfill(5)
    zcol = "ZCTA5CE10" if "ZCTA5CE10" in zcta.columns else "GEOID10"
    return zcta[zcol].astype(str).str.zfill(5)


def _load_power_line_features_by_zip() -> pd.DataFrame:
    """
    Load power line features aggregated by ZIP.

    Uses power line centroids spatially joined to ZCTA. Per ZIP: total_power_line_length
    = sum of line lengths, pct_underground_power = mean of is_underground (1 if
    ConductorType1=='UG', else 0). Returns empty DataFrame on error.
    """
    if "power_lines" in _feature_cache:
        return _feature_cache["power_lines"]
    try:
        lines = gpd.read_file(ELECTRIC_PATH)
        lines = lines.to_crs(GRID_CRS)
        lines["line_length"] = lines.geometry.length
        lines["is_underground"] = (
            lines["ConductorType1"].eq("UG").astype(int)
            if "ConductorType1" in lines.columns
            else np.zeros(len(lines), dtype=int)
        )
        zcta = gpd.read_file(ZCTA_PATH)
        zcta = zcta.to_crs("EPSG:4326")
        zcta["ZIP"] = _zcta_zip_series(zcta)
        lines["centroid"] = lines.geometry.centroid
        lines_pts = lines.set_geometry("centroid").to_crs("EPSG:4326")
        joined = gpd.sjoin(
            lines_pts[["OBJECTID", "line_length", "is_underground", "centroid"]],
            zcta[["ZIP", "geometry"]],
            how="left",
            predicate="within",
        )
        zip_agg = joined.groupby("ZIP").agg(
            total_power_line_length=("line_length", "sum"),
            pct_underground_power=("is_underground", "mean"),
        ).reset_index()
        zip_agg["total_power_line_length"] = zip_agg["total_power_line_length"].fillna(0)
        zip_agg["pct_underground_power"] = zip_agg["pct_underground_power"].fillna(0)
        result = zip_agg[["ZIP", "total_power_line_length", "pct_underground_power"]]
        _feature_cache["power_lines"] = result
        return result
    except Exception as e:
        print(f"Warning: Could not load power line features: {e}")
        return pd.DataFrame(
            columns=["ZIP", "total_power_line_length", "pct_underground_power"]
        )


def _load_road_features_by_zip() -> pd.DataFrame:
    """
    Load dist_to_major_road per ZIP.

    Min distance from ZCTA centroid to arterial roads (ARTERIAL_CODE > 0).
    Uses all streets if ARTERIAL_CODE column is missing. Returns 0.0 if no arterials.
    """
    if "roads" in _feature_cache:
        return _feature_cache["roads"]
    try:
        zcta = gpd.read_file(ZCTA_PATH)
        zcta = zcta.to_crs(GRID_CRS)
        zcta["ZIP"] = _zcta_zip_series(zcta)
        zcta["centroid"] = zcta.geometry.centroid
        streets = gpd.read_file(STREETS_PATH)
        streets = streets.to_crs(GRID_CRS)
        if "ARTERIAL_CODE" in streets.columns:
            arterials = streets[streets["ARTERIAL_CODE"] > 0].copy()
        else:
            arterials = streets.copy()
        if arterials.empty:
            result = pd.DataFrame({"ZIP": zcta["ZIP"], "dist_to_major_road": 0.0})
        else:
            dists = []
            for _, row in zcta.iterrows():
                d = arterials.distance(row["centroid"]).min()
                dists.append(float(d) if pd.notna(d) else 0.0)
            result = pd.DataFrame({"ZIP": zcta["ZIP"], "dist_to_major_road": dists})
        _feature_cache["roads"] = result
        return result
    except Exception as e:
        print(f"Warning: Could not load road features: {e}")
        return pd.DataFrame(columns=["ZIP", "dist_to_major_road"])


def _load_zoning_features_by_zip() -> pd.DataFrame:
    """
    Load pct_multifamily per ZIP.

    Zoning parcels joined to ZCTA by centroid. Per ZIP: pct_multifamily =
    multifamily_area / total_zone_area. Multifamily = ZONE_TYPE in ['MF'] or
    ZONELUT/ZONE containing 'MF'.
    """
    if "zoning" in _feature_cache:
        return _feature_cache["zoning"]
    try:
        zoning = gpd.read_file(ZONING_PATH)
        zoning = zoning.to_crs("EPSG:4326")
        zcta = gpd.read_file(ZCTA_PATH)
        zcta = zcta.to_crs("EPSG:4326")
        zcta["ZIP"] = _zcta_zip_series(zcta)
        zoning_proj = zoning.to_crs(GRID_CRS)
        zcta_proj = zcta.to_crs(GRID_CRS)
        zoning_proj["zone_area"] = zoning_proj.geometry.area
        if "ZONE_TYPE" in zoning_proj.columns:
            zoning_proj["is_multifamily"] = zoning_proj["ZONE_TYPE"].isin(["MF"]).astype(int)
        else:
            zone_col = "ZONELUT" if "ZONELUT" in zoning_proj.columns else "ZONE"
            zoning_proj["is_multifamily"] = (
                zoning_proj[zone_col]
                .str.upper()
                .str.contains(r"MF", na=False, regex=True)
                .astype(int)
            )
        zoning_pts = zoning_proj.copy()
        zoning_pts["centroid"] = zoning_pts.geometry.centroid
        zoning_pts = zoning_pts.set_geometry("centroid")
        joined = gpd.sjoin(
            zoning_pts[["zone_area", "is_multifamily", "centroid"]],
            zcta_proj[["ZIP", "geometry"]],
            how="left",
            predicate="within",
        )
        joined["multifamily_area"] = joined["zone_area"] * joined["is_multifamily"]
        zip_agg = joined.groupby("ZIP").agg(
            total_zone_area=("zone_area", "sum"),
            multifamily_area=("multifamily_area", "sum"),
        ).reset_index()
        zip_agg["pct_multifamily"] = (
            zip_agg["multifamily_area"] / zip_agg["total_zone_area"].replace(0, np.nan)
        ).fillna(0)
        result = zip_agg[["ZIP", "pct_multifamily"]]
        _feature_cache["zoning"] = result
        return result
    except Exception as e:
        print(f"Warning: Could not load zoning features: {e}")
        return pd.DataFrame(columns=["ZIP", "pct_multifamily"])


def _prepare_features(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Merge power, road, and zoning features by ZIP. Returns X with fillna(0)."""
    power = _load_power_line_features_by_zip()
    roads = _load_road_features_by_zip()
    zoning = _load_zoning_features_by_zip()
    df = scored_df[["ZIP"]].copy()
    df = df.merge(power, on="ZIP", how="left")
    df = df.merge(roads, on="ZIP", how="left")
    df = df.merge(zoning, on="ZIP", how="left")
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)
    return X


def _train_model(scored_df: pd.DataFrame):
    """
    Train model for station prediction. Returns (model, scaler).

    Uses LogisticRegression when 2+ classes and 1+ features; otherwise
    DummyClassifier. Features are scaled with StandardScaler.
    """
    X = _prepare_features(scored_df)
    y = (scored_df["station_count"] >= 1).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Need at least 2 classes and 1 feature for logistic regression
    if y.nunique() < 2 or X.shape[1] == 0:
        const = int(y.iloc[0]) if len(y) else 0
        model = DummyClassifier(strategy="constant", constant=const)
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def _get_or_train_model(scored_df: pd.DataFrame):
    """Return cached (model, scaler) for this scored_df, or train and cache."""
    sc_sum = (
        scored_df["station_count"].sum()
        if "station_count" in scored_df.columns
        else 0
    )
    key = (len(scored_df), sc_sum)
    if key not in _model_cache:
        _model_cache[key] = _train_model(scored_df)
    return _model_cache[key]


def get_zip_predictions(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict probability of station placement for each ZIP.

    Returns DataFrame with columns ["ZIP", "predicted_prob"]. Returns empty
    DataFrame if scored_df is empty or missing station_count column.
    """
    if scored_df.empty or "station_count" not in scored_df.columns:
        return pd.DataFrame({"ZIP": [], "predicted_prob": []})
    model, scaler = _get_or_train_model(scored_df)
    X = _prepare_features(scored_df)
    X_scaled = scaler.transform(X)
    prob_arr = model.predict_proba(X_scaled)
    if prob_arr.shape[1] > 1:
        probs = prob_arr[:, 1]
    elif 1 in model.classes_:
        probs = prob_arr[:, 0]
    else:
        probs = np.zeros(len(prob_arr))
    return pd.DataFrame({"ZIP": scored_df["ZIP"].values, "predicted_prob": probs})


def _get_zip_centroids() -> pd.DataFrame:
    """Return DataFrame with ZIP, Latitude, Longitude for each ZCTA centroid."""
    if "zip_centroids" in _feature_cache:
        return _feature_cache["zip_centroids"]
    zcta = gpd.read_file(ZCTA_PATH)
    zcta = zcta.to_crs("EPSG:4326")
    zcta["ZIP"] = _zcta_zip_series(zcta)
    zcta["centroid"] = zcta.geometry.centroid
    zcta["Longitude"] = zcta["centroid"].x
    zcta["Latitude"] = zcta["centroid"].y
    result = zcta[["ZIP", "Latitude", "Longitude"]]
    _feature_cache["zip_centroids"] = result
    return result


def generate_electric_lines_cache():
    """
    Clip electric lines to Seattle, convert to WGS84, save for fast map loading.

    Only builds if ELECTRIC_SEATTLE_CACHE does not exist. Call when
    electric_lines.geojson changes to rebuild.
    """
    if os.path.isfile(ELECTRIC_SEATTLE_CACHE):
        return
    try:
        os.makedirs(os.path.dirname(ELECTRIC_SEATTLE_CACHE), exist_ok=True)
        print("Building electric lines cache for map...")
        neighborhoods = gpd.read_file(NEIGHBORHOODS_PATH)
        neighborhoods = neighborhoods[neighborhoods.geometry.notna()].to_crs(GRID_CRS)
        seattle = neighborhoods.dissolve().geometry.iloc[0]
        lines = gpd.read_file(ELECTRIC_PATH).to_crs(GRID_CRS)
        lines = lines[lines.intersects(seattle)]
        if lines.empty:
            return
        lines = gpd.clip(lines, seattle)
        lines = lines.to_crs("EPSG:4326")
        lines.geometry = lines.geometry.simplify(0.00005)
        lines.to_file(ELECTRIC_SEATTLE_CACHE, driver="GeoJSON")
        print(f"Saved electric lines cache to {ELECTRIC_SEATTLE_CACHE}")
    except Exception as e:
        print(f"Warning: Could not build electric lines cache: {e}")


def geoms_to_plotly(geoms):
    """
    Convert LineString/MultiLineString geometries to (lons, lats) for Plotly.

    Returns (lons, lats) where lons/lats are lists with None separators.
    Returns (None, None) if no valid geometries. Pure function, easy to test.
    """
    lons = []
    lats = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            parts = [geom]
        for part in parts:
            for coord in part.coords:
                lons.append(coord[0])
                lats.append(coord[1])
            lons.append(None)
            lats.append(None)
    if len(lons) == 0:
        return None, None
    return lons, lats


def get_electric_lines_for_map():
    """
    Load pre-clipped electric lines for the map.

    Returns ((oh_lons, oh_lats), (ug_lons, ug_lats)) for Plotly Scattermapbox.
    Each tuple is (lons_list, lats_list) with None separators between lines.
    Returns (None, None) if cache file missing or empty.
    """
    if not os.path.isfile(ELECTRIC_SEATTLE_CACHE):
        return None, None
    gdf = gpd.read_file(ELECTRIC_SEATTLE_CACHE)
    if gdf.empty:
        return None, None
    col = "ConductorType1" if "ConductorType1" in gdf.columns else None
    oh_geoms = gdf[gdf[col].eq("OH")].geometry if col else gdf.geometry
    ug_geoms = gdf[gdf[col].eq("UG")].geometry if col else gpd.GeoSeries()
    return geoms_to_plotly(oh_geoms), geoms_to_plotly(ug_geoms)


EMPTY_RECOMMENDED = pd.DataFrame(
    columns=["Latitude", "Longitude", "ZIP", "predicted_prob", "Location"]
)


def filter_recommended_by_probability(
    recommended_df: pd.DataFrame, prob_min: float
) -> pd.DataFrame:
    """
    Return only recommended locations with predicted_prob >= prob_min.

    Returns input unchanged if empty or if predicted_prob column is missing.
    Pure function: same inputs always give same output.
    """
    if recommended_df.empty:
        return recommended_df
    if "predicted_prob" not in recommended_df.columns:
        return recommended_df
    return recommended_df[recommended_df["predicted_prob"] >= prob_min].copy()


def load_recommended_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load recommended stations from a CSV file.

    Returns DataFrame with columns: Latitude, Longitude, ZIP, predicted_prob,
    Location. Requires Latitude and Longitude columns. Returns EMPTY_RECOMMENDED
    if file missing, empty, or missing required columns. Fills predicted_prob
    and Location with defaults when those columns are missing.
    """
    if not os.path.isfile(file_path):
        return EMPTY_RECOMMENDED.copy()
    grid_df = pd.read_csv(file_path)
    if grid_df.empty:
        return EMPTY_RECOMMENDED.copy()
    if "Latitude" not in grid_df.columns or "Longitude" not in grid_df.columns:
        return EMPTY_RECOMMENDED.copy()
    # Build output with required columns
    out = pd.DataFrame()
    out["Latitude"] = grid_df["Latitude"]
    out["Longitude"] = grid_df["Longitude"]
    if "predicted_prob" in grid_df.columns:
        out["predicted_prob"] = grid_df["predicted_prob"]
    else:
        out["predicted_prob"] = 0.5
    if "Location" in grid_df.columns:
        out["Location"] = grid_df["Location"]
    else:
        out["Location"] = "Recommended"
    # Add ZIP from cell_id if available
    if "cell_id" in grid_df.columns:
        out["ZIP"] = [f"Cell {int(c)}" for c in grid_df["cell_id"]]
    else:
        out["ZIP"] = [f"Cell {i}" for i in range(len(out))]
    return out.sort_values("predicted_prob", ascending=False)


def get_recommended_stations() -> pd.DataFrame:
    """
    Load recommended locations (red dots) for the Prediction map.

    Reads from data/processed/recommended_grid_locations.csv. Run
    python -m interactive_map.ml_model (from project root) to generate.
    """
    return load_recommended_from_csv(GRID_RECOMMENDED_PATH)


def generate_grid_recommendations():
    """
    Build 500m grid over Seattle, train model, save recommended locations.

    Creates grid with features (power lines, roads, zoning), trains logistic
    regression, predicts cells needing stations. Saves recommended locations
    (predicted but no station) to recommended_grid_locations.csv. Caches grid
    to grid_with_features.geojson. Also builds electric lines cache if missing.
    """
    os.makedirs(os.path.dirname(GRID_RECOMMENDED_PATH), exist_ok=True)

    # 1. Load or build grid with features (cache avoids slow geo processing on reruns)
    if os.path.isfile(GRID_CACHE_PATH):
        print("Loading cached grid with features...")
        grid = gpd.read_file(GRID_CACHE_PATH).to_crs(GRID_CRS)
    else:
        print("Loading neighborhoods...")
        neighborhoods = gpd.read_file(NEIGHBORHOODS_PATH)
        neighborhoods = neighborhoods[neighborhoods.geometry.notna()].to_crs(GRID_CRS)

        print("Creating 500m grid...")
        xmin, ymin, xmax, ymax = neighborhoods.total_bounds
        cell_size = 1640  # feet ≈ 500 meters (same as notebook)
        cols = list(np.arange(xmin, xmax + cell_size, cell_size))
        rows = list(np.arange(ymin, ymax + cell_size, cell_size))
        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(Polygon([
                    (x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)
                ]))
        grid = gpd.GeoDataFrame({"geometry": polygons}, crs=GRID_CRS)
        grid = gpd.clip(grid, neighborhoods)
        grid["cell_id"] = range(len(grid))

        # 3. Compute features (slow – cached for next run)
        print("Computing dist_to_major_road...")
        streets = gpd.read_file(STREETS_PATH).to_crs(GRID_CRS)
        if "ARTERIAL_CODE" in streets.columns:
            arterials = streets[streets["ARTERIAL_CODE"] > 0]
        else:
            arterials = streets
        if arterials.empty:
            grid["dist_to_major_road"] = 0.0
        else:
            arterials_union = arterials.geometry.union_all()
            grid["dist_to_major_road"] = grid.geometry.centroid.distance(arterials_union)

        print("Computing power line features...")
        lines = gpd.read_file(ELECTRIC_PATH).to_crs(GRID_CRS)
        lines["line_length"] = lines.geometry.length
        lines["is_underground"] = (
            lines["ConductorType1"].eq("UG").astype(int)
            if "ConductorType1" in lines.columns
            else np.zeros(len(lines), dtype=int)
        )
        grid_with_power = gpd.sjoin(
            grid[["geometry"]], lines, how="left", predicate="intersects"
        )
        agg = grid_with_power.groupby(grid_with_power.index).agg(
            is_underground=("is_underground", "mean"),
            line_length=("line_length", "sum"),
        )
        agg = agg.rename(
            columns={
                "is_underground": "pct_underground_power",
                "line_length": "total_power_line_length",
            }
        )
        grid = grid.merge(agg, left_index=True, right_index=True, how="left")
        grid["pct_underground_power"] = grid["pct_underground_power"].fillna(0)
        grid["total_power_line_length"] = grid["total_power_line_length"].fillna(0)

        print("Computing pct_multifamily...")
        zoning = gpd.read_file(ZONING_PATH).to_crs(GRID_CRS)
        if "ZONE_TYPE" in zoning.columns:
            zoning["is_multifamily"] = zoning["ZONE_TYPE"].isin(["MF"]).astype(int)
        elif "ZONELUT" in zoning.columns or "ZONE" in zoning.columns:
            zone_col = "ZONELUT" if "ZONELUT" in zoning.columns else "ZONE"
            zoning["is_multifamily"] = (
                zoning[zone_col]
                .str.upper()
                .str.contains(r"MF", na=False, regex=True)
                .astype(int)
            )
        else:
            zoning["is_multifamily"] = 0
        zoning_pts = gpd.GeoDataFrame(
            zoning[["is_multifamily"]],
            geometry=zoning.geometry.centroid,
            crs=zoning.crs,
        )
        joined_z = gpd.sjoin(zoning_pts, grid[["geometry"]], how="inner", predicate="within")
        agg_z = joined_z.groupby("index_right")["is_multifamily"].mean().rename("pct_multifamily")
        grid = grid.merge(agg_z, left_index=True, right_index=True, how="left")
        grid["pct_multifamily"] = grid["pct_multifamily"].fillna(0)

        print("Saving grid cache for faster reruns...")
        grid.to_file(GRID_CACHE_PATH, driver="GeoJSON")

    # 2. Mark which grid cells already have EV stations (recomputed each run – EV data may change)
    print("Assigning has_station...")
    stations_df = pd.read_csv(EV_STATIONS_PATH).dropna(subset=["Latitude", "Longitude"])
    if "City" in stations_df.columns:
        stations_df = stations_df[stations_df["City"].fillna("").str.upper() == "SEATTLE"]
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
        crs="EPSG:4326",
    ).to_crs(GRID_CRS)
    joined = gpd.sjoin(grid, stations_gdf, how="left", predicate="intersects")
    cells_with_stations = joined.dropna(subset=["index_right"]).index.unique()
    grid["has_station"] = 0
    grid.loc[cells_with_stations, "has_station"] = 1

    # 3. Train logistic regression and predict
    print("Training model and predicting...")
    X = grid[FEATURE_COLS].fillna(0)
    y = grid["has_station"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if y.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=int(y.iloc[0]))
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    grid["predicted_station"] = model.predict(X_scaled)
    grid["prediction_proba"] = model.predict_proba(X_scaled)[:, 1]

    recommended = grid[
        (grid["predicted_station"] == 1) & (grid["has_station"] == 0)
    ]
    if recommended.empty:
        print(
            "No recommended locations "
            "(all high-prob cells already have stations)."
        )
        out_df = pd.DataFrame(
            columns=["Latitude", "Longitude", "predicted_prob", "cell_id", "Location"]
        )
    else:
        centroids_wgs84 = recommended.geometry.centroid
        centroids_gdf = gpd.GeoDataFrame(geometry=centroids_wgs84, crs=GRID_CRS).to_crs("EPSG:4326")
        out_df = pd.DataFrame({
            "Latitude": centroids_gdf.geometry.y.values,
            "Longitude": centroids_gdf.geometry.x.values,
            "predicted_prob": recommended["prediction_proba"].values,
            "cell_id": recommended["cell_id"].values,
        })
        def loc_label(row):
            return f"Grid cell {int(row['cell_id'])} (prob: {row['predicted_prob']:.2f})"

        out_df["Location"] = out_df.apply(loc_label, axis=1)

    # 4. Save recommended locations (red dots for the map)
    out_df.to_csv(GRID_RECOMMENDED_PATH, index=False)
    print(f"Saved {len(out_df)} recommended locations to {GRID_RECOMMENDED_PATH}")

    # 5. Build electric lines cache for map (if not exists)
    generate_electric_lines_cache()


if __name__ == "__main__":
    generate_grid_recommendations()
