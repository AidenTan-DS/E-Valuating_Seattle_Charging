"""
app_folium.py – Seattle EV Station Explorer (Folium version)
Run: streamlit run app_folium.py

Requires: pip install streamlit-folium folium

Key differences vs app.py (Plotly version):
──────────────────────────────────────────────────────────────────────────────
Feature          │ Plotly (app.py)                │ Folium (app_folium.py)
─────────────────┼────────────────────────────────┼────────────────────────────
Renderer         │ WebGL/Canvas (fast large data) │ SVG/Leaflet.js (HTML iframe)
Choropleth       │ Choroplethmapbox trace         │ folium.Choropleth + GeoJson
Click detection  │ on_select callback + fragment  │ st_folium dict + spatial join
Partial reruns   │ @st.fragment (map only reruns) │ Full page rerun every time
Station markers  │ Scattermapbox (single trace)   │ CircleMarker loop per station
Hover/popup      │ hovertemplate string           │ GeoJsonTooltip / folium.Tooltip
Detail map render│ st.plotly_chart                │ folium_static (iframe)
──────────────────────────────────────────────────────────────────────────────

NOTE (Folium 0.20.x): point_to_layer in GeoJson must be JsCode, not a Python
lambda — otherwise the renderer tries to JSON-serialize the function and raises
TypeError. Workaround used here: individual CircleMarker loop for stations.
"""

import json
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium, folium_static
from shapely.geometry import Point

# ── file paths ────────────────────────────────────────────────────────────────
EV_PATH      = "ev_station.csv"
TRAFFIC_PATH = "traffic_flow_with_zip.csv"
ZCTA_PATH    = "zcta.json"

st.set_page_config(page_title="Seattle EV Explorer (Folium)", page_icon="⚡", layout="wide")

# ── ADT colour bins (same as app.py) ─────────────────────────────────────────
ADT_BINS   = [0, 10_000, 20_000, 35_000, 50_000, float("inf")]
ADT_LABELS = ["< 10K", "10K – 20K", "20K – 35K", "35K – 50K", "> 50K"]
ADT_COLORS = ["#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a", "#172554"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (identical to app.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_ev_stations() -> pd.DataFrame:
    cols = [
        "Station Name", "Latitude", "Longitude", "ZIP", "State",
        "EV Level2 EVSE Num", "EV DC Fast Count", "EV Network",
    ]
    df = pd.read_csv(EV_PATH, usecols=cols, low_memory=False)
    df = df[df["State"] == "WA"].dropna(subset=["Latitude", "Longitude"]).copy()
    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)
    df["EV Level2 EVSE Num"] = df["EV Level2 EVSE Num"].fillna(0)
    df["EV DC Fast Count"]   = df["EV DC Fast Count"].fillna(0)
    return df


def fix_missing_zips(ev_df: pd.DataFrame, zcta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    valid_zips   = set(zcta_gdf["ZIP_zcta"])
    missing_mask = ~ev_df["ZIP"].isin(valid_zips)
    if not missing_mask.any():
        return ev_df
    pts = gpd.GeoDataFrame(
        ev_df[missing_mask].copy(),
        geometry=gpd.points_from_xy(
            ev_df.loc[missing_mask, "Longitude"],
            ev_df.loc[missing_mask, "Latitude"],
        ),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, zcta_gdf, how="left", predicate="within")
    remap  = joined["ZIP_zcta"].dropna()
    ev_df  = ev_df.copy()
    ev_df.loc[remap.index, "ZIP"] = remap.values
    return ev_df


def aggregate_ev_by_zip(ev_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ev_df.groupby("ZIP")
        .agg(
            station_count=("Station Name",       "count"),
            level2_spots =("EV Level2 EVSE Num", "sum"),
            dcfast_count =("EV DC Fast Count",   "sum"),
        )
        .reset_index()
    )


def aggregate_traffic_from_csv() -> pd.DataFrame:
    df = pd.read_csv(TRAFFIC_PATH)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    adt_col = "STUDY_ADT" if "STUDY_ADT" in df.columns else "STUDY_AWDT"
    return (
        df.groupby("zip_code")
        .agg(
            mean_ADT   =(adt_col,               "mean"),
            population =("Population Estimate", "first"),
            city       =("city",                "first"),
        )
        .reset_index()
        .rename(columns={"zip_code": "ZIP"})
    )


def load_zcta(path: str = ZCTA_PATH) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    zip_col = "ZCTA5CE10" if "ZCTA5CE10" in gdf.columns else "GEOID10"
    gdf["ZIP_zcta"] = gdf[zip_col].astype(str).str.zfill(5)
    return gdf[["ZIP_zcta", "geometry"]].set_crs("EPSG:4326", allow_override=True)


def build_master(ev_df, traffic_df) -> pd.DataFrame:
    master = traffic_df.merge(ev_df, on="ZIP", how="left")
    master["station_count"] = master["station_count"].fillna(0)
    master["level2_spots"]  = master["level2_spots"].fillna(0)
    master["dcfast_count"]  = master["dcfast_count"].fillna(0)
    return master


@st.cache_data(show_spinner="Loading data…")
def load_all():
    zcta_gdf  = load_zcta()
    ev_raw    = load_ev_stations()
    ev_df     = fix_missing_zips(ev_raw, zcta_gdf)
    ev_by_zip = aggregate_ev_by_zip(ev_df)
    t_by_zip  = aggregate_traffic_from_csv()
    scored    = build_master(ev_by_zip, t_by_zip)
    return zcta_gdf, ev_df, scored


@st.cache_data(show_spinner=False)
def load_roads() -> gpd.GeoDataFrame:
    df = pd.read_csv(TRAFFIC_PATH)
    df = df.dropna(subset=["longitude", "latitude"])
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    adt_col = "STUDY_ADT" if "STUDY_ADT" in gdf.columns else "STUDY_AWDT"
    gdf["ADT"] = gdf[adt_col].fillna(0)
    for c in ["TITLE", "STUDY_DIRFLOW"]:
        if c in gdf.columns:
            gdf[c] = gdf[c].fillna("—").astype(str)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Folium map builders
# ─────────────────────────────────────────────────────────────────────────────

def build_main_map_folium(
    ev_df: pd.DataFrame,
    scored: pd.DataFrame,
    zcta_gdf: gpd.GeoDataFrame,
    selected_zip: str | None = None,
) -> folium.Map:
    """
    Main overview map:
    - ZIP choropleth coloured by mean ADT  ← folium.Choropleth (YlOrRd scale)
    - Hover tooltip on ZIP polygons        ← GeoJsonTooltip on choropleth.geojson
    - Selected ZIP highlighted yellow      ← separate GeoJson overlay
    - EV station dots (green)              ← CircleMarker loop

    Difference from Plotly:
    * Plotly: single Choroplethmapbox trace handles color + hover together.
    * Folium: Choropleth only handles fill color; tooltip data must be
      injected manually into choropleth.geojson.data["features"] then a
      GeoJsonTooltip child is added. Two separate steps vs one in Plotly.
    * point_to_layer (Python lambda) cannot be JSON-serialized by Folium 0.20;
      so EV stations use an explicit CircleMarker loop instead of a GeoJson layer.
    """
    m = folium.Map(location=[47.61, -122.33], zoom_start=11, tiles="CartoDB positron")

    # Merge ZCTA geometry with scored data, convert to plain GeoJSON dict
    # (avoids GeoDataFrame-specific serialization quirks in Folium 0.20)
    gdf = zcta_gdf.merge(scored, left_on="ZIP_zcta", right_on="ZIP", how="inner")
    gdf_json = json.loads(gdf[["ZIP_zcta", "geometry"]].to_json())

    # ── Choropleth: ADT fill ──────────────────────────────────────────────────
    choropleth = folium.Choropleth(
        geo_data   = gdf_json,
        name       = "Traffic by ZIP",
        data       = scored[["ZIP", "mean_ADT"]],
        columns    = ["ZIP", "mean_ADT"],
        key_on     = "feature.properties.ZIP_zcta",
        fill_color = "YlOrRd",
        fill_opacity   = 0.6,
        line_opacity   = 0.5,
        line_color     = "#888",
        legend_name    = "Avg Daily Traffic (ADT)",
        nan_fill_color = "lightgray",
    )
    choropleth.add_to(m)

    # ── Tooltip: inject extra properties then add GeoJsonTooltip child ────────
    # Difference from Plotly: Plotly's hovertemplate reads data from the trace
    # directly. Folium requires mutating the already-built GeoJSON features to
    # add display fields, then attaching a tooltip widget as a child element.
    scored_lookup = scored.set_index("ZIP")[["mean_ADT", "city"]].to_dict("index")
    for feature in choropleth.geojson.data["features"]:
        zc = feature["properties"].get("ZIP_zcta", "")
        if zc in scored_lookup:
            adt  = scored_lookup[zc]["mean_ADT"]
            city = scored_lookup[zc]["city"]
            feature["properties"]["adt_display"]  = f"{adt:,.0f}"
            feature["properties"]["city_display"] = str(city) if city == city else "—"
        else:
            feature["properties"]["adt_display"]  = "N/A"
            feature["properties"]["city_display"] = "—"

    choropleth.geojson.add_child(
        folium.GeoJsonTooltip(
            fields  = ["ZIP_zcta", "adt_display", "city_display"],
            aliases = ["ZIP:", "Avg Daily Traffic:", "City:"],
            sticky  = True,
            style   = "font-size: 12px; padding: 6px;",
        )
    )

    # ── Selected ZIP highlight ────────────────────────────────────────────────
    if selected_zip:
        sel_row = zcta_gdf[zcta_gdf["ZIP_zcta"] == selected_zip]
        if not sel_row.empty:
            sel_json = json.loads(sel_row.to_json())
            folium.GeoJson(
                sel_json,
                style_function = lambda _: {
                    "fillColor"  : "#fbbf24",
                    "color"      : "#f59e0b",
                    "weight"     : 3,
                    "fillOpacity": 0.45,
                },
            ).add_to(m)

    # ── EV station dots ───────────────────────────────────────────────────────
    # Difference from Plotly: Plotly renders all stations as one Scattermapbox
    # trace (WebGL vectorized). Folium 0.20 requires individual CircleMarker
    # objects; point_to_layer with a Python lambda fails JSON serialization.
    seattle_ev = ev_df[ev_df["ZIP"].isin(set(scored["ZIP"]))].copy()
    ev_group = folium.FeatureGroup(name="EV Stations", show=True)
    for _, row in seattle_ev.iterrows():
        tt = (
            f"<b>{row['Station Name']}</b><br>"
            f"ZIP: {row['ZIP']}<br>"
            f"Level 2: {int(row['EV Level2 EVSE Num'])} | "
            f"DC Fast: {int(row['EV DC Fast Count'])}<br>"
            f"Network: {row['EV Network']}"
        )
        folium.CircleMarker(
            location     = [row["Latitude"], row["Longitude"]],
            radius       = 5,
            color        = "#16a34a",
            fill         = True,
            fill_color   = "#22c55e",
            fill_opacity = 0.8,
            weight       = 1.5,
            tooltip      = folium.Tooltip(tt, sticky=True),
        ).add_to(ev_group)
    ev_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_detail_map_folium(
    zip_code: str,
    roads_gdf: gpd.GeoDataFrame,
    zcta_gdf: gpd.GeoDataFrame,
    ev_df: pd.DataFrame,
) -> folium.Map:
    """
    Detail map for a selected ZIP:
    - ZIP boundary (yellow fill)          ← GeoJson with static style_function
    - Traffic study points by ADT bin     ← CircleMarker per point, sized by ADT
    - EV station dots (green)             ← CircleMarker with popup

    Difference from Plotly: Plotly draws horizontal dash lines (Scattermapbox
    lines mode) at each traffic study point. Folium uses CircleMarkers sized
    by ADT bin — cleaner for Folium's circle-based rendering style.
    """
    zip_row = zcta_gdf[zcta_gdf["ZIP_zcta"] == zip_code]
    center  = [47.61, -122.33]
    if not zip_row.empty:
        c      = zip_row.iloc[0]["geometry"].centroid
        center = [c.y, c.x]

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # ZIP boundary highlight
    if not zip_row.empty:
        zip_json = json.loads(zip_row.to_json())
        folium.GeoJson(
            zip_json,
            style_function = lambda _: {
                "fillColor"  : "#fef9c3",
                "color"      : "#f59e0b",
                "weight"     : 2.5,
                "fillOpacity": 0.25,
            },
        ).add_to(m)

    # Traffic study points, one FeatureGroup per ADT bin
    roads_zip = roads_gdf[roads_gdf["zip_code"] == zip_code].copy()
    if not roads_zip.empty:
        roads_zip = roads_zip.copy()
        roads_zip["lon"] = roads_zip.geometry.x
        roads_zip["lat"] = roads_zip.geometry.y
        loc_col = "TITLE"
        dir_col = "STUDY_DIRFLOW" if "STUDY_DIRFLOW" in roads_zip.columns else None

        for i, (lo, hi) in enumerate(zip(ADT_BINS[:-1], ADT_BINS[1:])):
            subset = roads_zip[(roads_zip["ADT"] >= lo) & (roads_zip["ADT"] < hi)]
            if subset.empty:
                continue
            layer  = folium.FeatureGroup(name=f"ADT {ADT_LABELS[i]}", show=True)
            radius = 4 + i * 1.5   # bigger circle = higher traffic band

            for _, row in subset.iterrows():
                adt_str = f"{row['ADT']:,.0f}"
                loc     = str(row.get(loc_col, "—")) if loc_col in row.index else "—"
                direc   = str(row.get(dir_col, "—")) if dir_col else "—"
                tt      = f"<b>{loc}</b>"
                if direc not in ("—", "nan", ""):
                    tt += f"<br>Dir: {direc}"
                tt += f"<br>ADT: <b>{adt_str}</b> vehicles/day"

                folium.CircleMarker(
                    location     = [row["lat"], row["lon"]],
                    radius       = radius,
                    color        = ADT_COLORS[i],
                    fill         = True,
                    fill_color   = ADT_COLORS[i],
                    fill_opacity = 0.75,
                    weight       = 2,
                    tooltip      = folium.Tooltip(tt, sticky=True),
                ).add_to(layer)
            layer.add_to(m)

    # EV stations for this ZIP
    ev_zip = ev_df[ev_df["ZIP"] == zip_code]
    if not ev_zip.empty:
        ev_layer = folium.FeatureGroup(name="EV Stations", show=True)
        for _, row in ev_zip.iterrows():
            popup_html = (
                f"<b>{row['Station Name']}</b><br>"
                f"Level 2 ports: <b>{int(row['EV Level2 EVSE Num'])}</b><br>"
                f"DC Fast ports: <b>{int(row['EV DC Fast Count'])}</b><br>"
                f"Network: <b>{row['EV Network']}</b>"
            )
            folium.CircleMarker(
                location     = [row["Latitude"], row["Longitude"]],
                radius       = 8,
                color        = "#16a34a",
                fill         = True,
                fill_color   = "#22c55e",
                fill_opacity = 0.9,
                weight       = 2,
                tooltip      = row["Station Name"],
                popup        = folium.Popup(popup_html, max_width=250),
            ).add_to(ev_layer)
        ev_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    zcta_gdf, ev_df, scored = load_all()
    roads_gdf  = load_roads()
    valid_zips = set(scored["ZIP"])

    for key, default in [("selected_zip", None), ("last_click_key", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    sel = st.session_state.selected_zip

    st.title("⚡ Seattle EV Station Explorer (Folium)")
    st.info(
        "**How to use this map**  \n"
        "- The left map shows **average daily traffic (ADT)** by ZIP code — darker = higher traffic.  \n"
        "- Green dots are **EV charging stations**. Hover over a dot to see station details.  \n"
        "- **Click any ZIP area** to load the traffic study map and charging stats on the right.",
        icon="ℹ️",
    )

    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        # ── Folium map + click detection ──────────────────────────────────────
        # Difference from Plotly: st_folium returns a plain dict.
        # We read last_clicked (lat/lng) and use gpd.sjoin to find the ZIP.
        # Plotly's on_select="rerun" directly returns the clicked feature's
        # location/customdata — no spatial join needed.
        m = build_main_map_folium(ev_df, scored, zcta_gdf, sel)
        map_data = st_folium(m, width="100%", height=460, key="main_map_folium")

        clicked = map_data.get("last_clicked")
        if clicked:
            click_key = f"{clicked['lat']:.5f},{clicked['lng']:.5f}"
            if click_key != st.session_state.last_click_key:
                st.session_state.last_click_key = click_key
                pt_gdf = gpd.GeoDataFrame(
                    geometry=[Point(clicked["lng"], clicked["lat"])],
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(pt_gdf, zcta_gdf, how="left", predicate="within")
                if not joined.empty and not pd.isna(joined["ZIP_zcta"].iloc[0]):
                    new_zip = joined["ZIP_zcta"].iloc[0]
                    if new_zip in valid_zips and new_zip != sel:
                        st.session_state.selected_zip = new_zip
                        st.rerun()

    sel = st.session_state.selected_zip  # re-read after potential update

    with right_col:
        if sel:
            scored_row = scored[scored["ZIP"] == sel]
            ev_in_zip  = ev_df[ev_df["ZIP"] == sel]
            city_val   = scored_row["city"].iloc[0] if not scored_row.empty else "—"
            pop_val    = int(scored_row["population"].iloc[0]) if not scored_row.empty else None

            hdr_col, btn_col = st.columns([5, 1])
            hdr_col.markdown(f"### {city_val} · ZIP {sel}")
            if btn_col.button("✕ Clear", use_container_width=True):
                st.session_state.selected_zip  = None
                st.session_state.last_click_key = None
                st.rerun()

            st.markdown(
                "**Traffic study volume** (ADT – Average Daily Traffic)",
                help="ADT = total annual vehicle count ÷ 365. "
                     "Covers all days and seasons, averaged across all study "
                     "points within this ZIP code.",
            )
            # folium_static: renders an interactive Leaflet iframe but sends no
            # events back to Python. Equivalent to st.plotly_chart without on_select.
            detail_m = build_detail_map_folium(sel, roads_gdf, zcta_gdf, ev_df)
            folium_static(detail_m, height=400)

            st.markdown("**Population & Charging**")
            c1, c2 = st.columns(2)
            c1.metric("Population",    f"{pop_val:,}" if pop_val else "N/A")
            c2.metric("Stations",      len(ev_in_zip))
            c3, c4 = st.columns(2)
            c3.metric("Level2 Spots",  int(ev_in_zip["EV Level2 EVSE Num"].sum()),
                      help="Level 2 AC charging ports (240V). Typical charge time: 4–12 hrs.")
            c4.metric("DC Fast Spots", int(ev_in_zip["EV DC Fast Count"].sum()),
                      help="DC Fast Charging (DCFC) ports (50–350+ kW). ~80% in 20–45 min.")

            if not scored_row.empty:
                st.metric("Avg Daily Traffic",
                          f"{scored_row['mean_ADT'].iloc[0]:,.0f}",
                          help="ADT averaged across all traffic study points in this ZIP.")

            st.markdown(f"**Stations in ZIP {sel}**")
            if not ev_in_zip.empty:
                disp = {
                    "Station Name":       "Station",
                    "EV Level2 EVSE Num": "Level2",
                    "EV DC Fast Count":   "DC Fast",
                    "EV Network":         "Network",
                }
                st.dataframe(
                    ev_in_zip[list(disp)].rename(columns=disp)
                        .sort_values("Level2", ascending=False)
                        .reset_index(drop=True),
                    use_container_width=True,
                    height=200,
                )
            else:
                st.info("No stations in this ZIP.")

        else:
            st.markdown("### Click a ZIP to explore")
            st.markdown("Select any ZIP code on the map to see traffic study volume, "
                        "population, and charging station details.")
            st.divider()
            seattle_ev = ev_df[ev_df["ZIP"].isin(valid_zips)]
            st.markdown("**City-wide summary**")
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("ZIPs with data",      len(valid_zips))
            r1c2.metric("Total EV Stations",    len(seattle_ev))
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Total Level2 Spots",   int(seattle_ev["EV Level2 EVSE Num"].sum()),
                        help="Level 2 AC charging ports (240V).")
            r2c2.metric("Total DC Fast Spots",  int(seattle_ev["EV DC Fast Count"].sum()),
                        help="DC Fast Charging (DCFC) ports.")

    st.caption(
        "Data sources: Population estimates and traffic flow (ADT) from Seattle 2025–2026 "
        "traffic study data. EV station data from AFDC (Alternative Fuels Data Center)."
    )


if __name__ == "__main__":
    main()
