"""
app_v2.py – Seattle EV Station Explorer  (Version 2)
Run: streamlit run app_v2.py

Main map  : ZIP choropleth (avg_daily_flow) + EV station dots  → click a ZIP
Detail map: real Seattle road lines coloured by avg_daily_flow + EV station dots
Data      : all_variables.csv  +  seattle_streets.geojson  +  ev_station.csv
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st

# ── file paths ────────────────────────────────────────────────────────────────
EV_PATH      = "data/cleaned/ev_station.csv"
TRAFFIC_PATH = "data/cleaned/all_variables.csv"
ZCTA_PATH    = "data/geo/zcta.json"
STREETS_PATH = "data/geo/seattle_streets.geojson"   # WGS84, Seattle-only, pre-built

st.set_page_config(page_title="Seattle EV Explorer", page_icon="⚡", layout="wide")

# ── ADT colour bins ───────────────────────────────────────────────────────────
ADT_BINS   = [0, 10_000, 20_000, 35_000, 50_000, float("inf")]
ADT_LABELS = ["< 10K", "10K – 20K", "20K – 35K", "35K – 50K", "> 50K"]
ADT_COLORS = ["#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a", "#172554"]
ADT_WIDTHS = [2.0, 3.0, 4.0, 5.0, 6.5]


# ─────────────────────────────────────────────────────────────────────────────
# Data functions
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
    # Clean income column (may contain "-" or "**")
    df["Median Household Income"] = pd.to_numeric(
        df["Median Household Income"], errors="coerce"
    )
    df["Total EV Registrations"] = pd.to_numeric(
        df["Total EV Registrations"], errors="coerce"
    )
    return (
        df.groupby("zip_code")
        .agg(
            mean_ADT       =("avg_daily_flow",          "mean"),
            population     =("Population Estimate",     "first"),
            city           =("city",                    "first"),
            med_income     =("Median Household Income", "first"),
            ev_registrations=("Total EV Registrations", "first"),
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


def build_geojson(zcta_gdf: gpd.GeoDataFrame, zip_set: set) -> dict:
    subset = zcta_gdf[zcta_gdf["ZIP_zcta"].isin(zip_set)]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": row["ZIP_zcta"],
                "properties": {"ZIP": row["ZIP_zcta"]},
                "geometry": row["geometry"].__geo_interface__,
            }
            for _, row in subset.iterrows()
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_all():
    zcta_gdf  = load_zcta()
    ev_raw    = load_ev_stations()
    ev_df     = fix_missing_zips(ev_raw, zcta_gdf)
    ev_by_zip = aggregate_ev_by_zip(ev_df)
    t_by_zip  = aggregate_traffic_from_csv()
    scored    = build_master(ev_by_zip, t_by_zip)
    return zcta_gdf, ev_df, scored


@st.cache_data(show_spinner="Loading road network…")
def load_streets_with_adt() -> gpd.GeoDataFrame:
    """
    Load Seattle street center lines (WGS84) and join avg_daily_flow
    from all_variables.csv by (ZIP, street_name).
    Returns GeoDataFrame with columns: ORD_STNAME_CONCAT, L_ZIP, R_ZIP,
    ARTERIAL_CODE, adt_l, adt_r, geometry.
    """
    streets = gpd.read_file(STREETS_PATH)
    streets["L_ZIP"] = streets["L_ZIP"].astype(str).str.zfill(5)
    streets["R_ZIP"] = streets["R_ZIP"].astype(str).str.zfill(5)

    # Build ADT lookup: (zip_code, street_name) → avg_daily_flow
    df = pd.read_csv(TRAFFIC_PATH)
    df["zip_code"]    = df["zip_code"].astype(str).str.zfill(5)
    df["street_name"] = df["STDY_TITLE_PART"].str.split(",").str[0].str.strip()
    agg = (
        df.groupby(["zip_code", "street_name"])["avg_daily_flow"]
        .mean()
        .reset_index()
    )

    # Join from the left side (L_ZIP)
    m_l = streets.merge(
        agg.rename(columns={"zip_code": "_zl", "street_name": "_sl", "avg_daily_flow": "adt_l"}),
        left_on=["L_ZIP", "ORD_STNAME_CONCAT"],
        right_on=["_zl", "_sl"],
        how="left",
    ).drop(columns=["_zl", "_sl"], errors="ignore")

    # Join from the right side (R_ZIP)
    m_r = streets.merge(
        agg.rename(columns={"zip_code": "_zr", "street_name": "_sr", "avg_daily_flow": "adt_r"}),
        left_on=["R_ZIP", "ORD_STNAME_CONCAT"],
        right_on=["_zr", "_sr"],
        how="left",
    ).drop(columns=["_zr", "_sr"], errors="ignore")

    streets = streets.copy()
    streets["adt_l"] = m_l["adt_l"].values
    streets["adt_r"] = m_r["adt_r"].values
    return streets


@st.cache_data(show_spinner=False)
def cached_geojson(_zcta_gdf, zip_tuple):
    return build_geojson(_zcta_gdf, set(zip_tuple))


@st.cache_data(show_spinner=False)
def cached_road_fig(zip_code, _streets_gdf, _zcta_gdf, _ev_df):
    return build_road_map(zip_code, _streets_gdf, _zcta_gdf, _ev_df)


@st.cache_data(show_spinner=False)
def single_zip_geojson(_zcta_gdf, zip_code):
    row = _zcta_gdf[_zcta_gdf["ZIP_zcta"] == zip_code]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": zip_code,
            "properties": {},
            "geometry": r["geometry"].__geo_interface__,
        }],
    }


@st.cache_data(show_spinner=False)
def zip_centroid(_zcta_gdf, zip_code):
    row = _zcta_gdf[_zcta_gdf["ZIP_zcta"] == zip_code]
    if row.empty:
        return {"lat": 47.61, "lon": -122.33}
    c = row.iloc[0]["geometry"].centroid
    return {"lat": c.y, "lon": c.x}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _line_coords(geom):
    """Return (lons, lats) lists for a LineString, with a trailing None separator."""
    coords = list(geom.coords)
    lons = [c[0] for c in coords] + [None]
    lats = [c[1] for c in coords] + [None]
    return lons, lats


def _midpoint(geom):
    coords = list(geom.coords)
    mid = coords[len(coords) // 2]
    return mid[0], mid[1]


# ─────────────────────────────────────────────────────────────────────────────
# Main map: ZIP choropleth + EV station dots
# ─────────────────────────────────────────────────────────────────────────────
def build_main_map(ev_df, scored, geojson, selected_zip):
    zip_list = scored["ZIP"].tolist()
    sel_idx  = zip_list.index(selected_zip) if selected_zip in zip_list else None

    fig = go.Figure(go.Choroplethmapbox(
        geojson        = geojson,
        locations      = scored["ZIP"],
        z              = scored["mean_ADT"],
        colorscale     = "Oranges",
        zmin           = 0,
        zmax           = scored["mean_ADT"].quantile(0.95),
        marker_opacity = 0.55,
        marker_line_width  = 1.0,
        marker_line_color  = "#888",
        colorbar = dict(title=dict(text="Avg Daily<br>Flow", font=dict(size=11)),
                        thickness=12, len=0.45, x=1.01),
        hovertemplate  = "<b>ZIP %{location}</b><br>Avg daily flow: %{z:,.0f}<br><i>Click to explore</i><extra></extra>",
        name           = "Traffic per ZIP",
        showlegend     = False,
        selectedpoints = [sel_idx] if sel_idx is not None else None,
    ))

    seattle_ev = ev_df[ev_df["ZIP"].isin(set(scored["ZIP"]))].copy()
    fig.add_trace(go.Scattermapbox(
        lat  = seattle_ev["Latitude"],
        lon  = seattle_ev["Longitude"],
        mode = "markers",
        marker = dict(size=6, color="#22c55e", opacity=0.75),
        text       = seattle_ev["Station Name"],
        customdata = seattle_ev[["ZIP", "EV Level2 EVSE Num", "EV DC Fast Count", "EV Network"]].values,
        hovertemplate = (
            "<b>%{text}</b><br>"
            "<br>"
            "ZIP:  <b>%{customdata[0]}</b><br>"
            "Level 2 ports:  <b>%{customdata[1]:.0f}</b><br>"
            "DC Fast ports:  <b>%{customdata[2]:.0f}</b><br>"
            "Network:  <b>%{customdata[3]}</b>"
            "<extra></extra>"
        ),
        name       = "EV Station",
        showlegend = True,
        hoverlabel = dict(bgcolor="white", bordercolor="#22c55e", font=dict(size=13)),
    ))

    fig.update_layout(
        mapbox_style  = "carto-positron",
        mapbox_zoom   = 10.5,
        mapbox_center = {"lat": 47.61, "lon": -122.33},
        margin        = dict(l=0, r=0, t=0, b=0),
        height        = 460,
        showlegend    = True,
        legend        = dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.90)",
                             bordercolor="#22c55e", borderwidth=1.5,
                             font=dict(size=13)),
        uirevision    = "main",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Detail map: real road lines coloured by avg_daily_flow
# ─────────────────────────────────────────────────────────────────────────────
def build_road_map(zip_code, streets_gdf, zcta_gdf, ev_df):
    """
    Draw Seattle road line segments for the selected ZIP, coloured by
    avg_daily_flow ADT bins.  Background roads (no data) shown in light gray.
    """
    center = zip_centroid(zcta_gdf, zip_code)
    zip_gj = single_zip_geojson(zcta_gdf, zip_code)

    zip_poly = zcta_gdf[zcta_gdf["ZIP_zcta"] == zip_code]
    fig = go.Figure()

    # ZIP boundary fill
    if zip_gj:
        fig.add_trace(go.Choroplethmapbox(
            geojson           = zip_gj,
            locations         = [zip_code],
            z                 = [1],
            colorscale        = [[0, "rgba(250,204,21,0.12)"],
                                  [1, "rgba(250,204,21,0.12)"]],
            marker_opacity    = 1,
            marker_line_width = 0,
            showscale         = False,
            hoverinfo         = "skip",
            name              = "ZIP boundary",
        ))

    if zip_poly.empty:
        fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=13,
                          mapbox_center=center, margin=dict(l=0, r=0, t=0, b=0), height=400)
        return fig

    # Filter street segments that belong to this ZIP (either side)
    mask = (streets_gdf["L_ZIP"] == zip_code) | (streets_gdf["R_ZIP"] == zip_code)
    zip_streets = streets_gdf[mask].copy()

    if zip_streets.empty:
        st.caption("No road data for this ZIP.")
        fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=13,
                          mapbox_center=center, margin=dict(l=0, r=0, t=0, b=0), height=400)
        return fig

    # Pick ADT: prefer the side that matches the selected ZIP
    def _pick_adt(row):
        if row["L_ZIP"] == zip_code and pd.notna(row["adt_l"]):
            return row["adt_l"]
        if row["R_ZIP"] == zip_code and pd.notna(row["adt_r"]):
            return row["adt_r"]
        # fallback: use whichever side has data
        if pd.notna(row["adt_l"]):
            return row["adt_l"]
        if pd.notna(row["adt_r"]):
            return row["adt_r"]
        return np.nan

    zip_streets["plot_adt"] = zip_streets.apply(_pick_adt, axis=1)

    # ── Coloured roads per ADT bin ───────────────────────────────────────────
    with_adt = zip_streets[zip_streets["plot_adt"].notna()]
    for i, (lo, hi) in enumerate(zip(ADT_BINS[:-1], ADT_BINS[1:])):
        subset = with_adt[(with_adt["plot_adt"] >= lo) & (with_adt["plot_adt"] < hi)]
        if subset.empty:
            continue

        # Collect all line coordinates (None-separated)
        lons_line, lats_line = [], []
        for geom in subset.geometry:
            lons, lats = _line_coords(geom)
            lons_line.extend(lons)
            lats_line.extend(lats)

        fig.add_trace(go.Scattermapbox(
            lat=lats_line, lon=lons_line, mode="lines",
            line=dict(width=ADT_WIDTHS[i], color=ADT_COLORS[i]),
            name=f"ADT {ADT_LABELS[i]}",
            hoverinfo="skip", showlegend=True,
            legendgroup=f"adt_{i}",
        ))

        # Invisible midpoint markers for hover
        mid_lons = [_midpoint(geom)[0] for geom in subset.geometry]
        mid_lats = [_midpoint(geom)[1] for geom in subset.geometry]
        hover_texts = [
            f"<b>{row['ORD_STNAME_CONCAT']}</b><br>"
            f"ADT: <b>{row['plot_adt']:,.0f}</b> vehicles/day"
            for _, row in subset.iterrows()
        ]
        fig.add_trace(go.Scattermapbox(
            lat=mid_lats, lon=mid_lons, mode="markers",
            marker=dict(size=16, color=ADT_COLORS[i], opacity=0),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(namelength=-1, font=dict(size=12), bgcolor="white"),
            name=f"ADT {ADT_LABELS[i]}", showlegend=False,
            legendgroup=f"adt_{i}",
        ))

    # ── EV station dots ──────────────────────────────────────────────────────
    ev_zip = ev_df[ev_df["ZIP"] == zip_code]
    if not ev_zip.empty:
        fig.add_trace(go.Scattermapbox(
            lat  = ev_zip["Latitude"],
            lon  = ev_zip["Longitude"],
            mode = "markers",
            marker = dict(size=11, color="#22c55e", opacity=0.9),
            text       = ev_zip["Station Name"],
            customdata = ev_zip[["EV Level2 EVSE Num", "EV DC Fast Count", "EV Network"]].values,
            hovertemplate = (
                "<b>%{text}</b><br>"
                "<br>"
                "Level 2 ports:  <b>%{customdata[0]:.0f}</b><br>"
                "DC Fast ports:  <b>%{customdata[1]:.0f}</b><br>"
                "Network:  <b>%{customdata[2]}</b>"
                "<extra></extra>"
            ),
            name = "EV Station",
            hoverlabel = dict(bgcolor="white", bordercolor="#22c55e", font=dict(size=13)),
        ))

    fig.update_layout(
        mapbox_style  = "carto-positron",
        mapbox_zoom   = 13.0,
        mapbox_center = center,
        margin        = dict(l=0, r=0, t=0, b=0),
        height        = 400,
        legend        = dict(
            title            = dict(text="ADT (vehicles/day)", font=dict(size=11)),
            x=0.01, y=0.99,
            bgcolor          = "rgba(255,255,255,0.88)",
            bordercolor      = "#ddd", borderwidth=1,
            font             = dict(size=11),
            itemclick        = False,
            itemdoubleclick  = False,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Map fragment
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment
def map_fragment():
    zcta_gdf, ev_df, scored = load_all()
    valid_zips  = set(scored["ZIP"])
    overview_gj = cached_geojson(zcta_gdf, tuple(sorted(valid_zips)))
    chart_key   = f"main_map_{st.session_state.get('map_version', 0)}"

    _pre = st.session_state.get(chart_key)
    if _pre is not None:
        _pts = getattr(getattr(_pre, "selection", None), "points", None) or []
        if _pts:
            pt = _pts[0]
            ek = repr(pt)
            if ek != st.session_state.get("last_event_key"):
                st.session_state.last_event_key = ek
                raw = pt.get("location")
                if raw is None:
                    cd = pt.get("customdata")
                    if cd:
                        raw = cd[0]
                if raw:
                    new_zip = str(raw).zfill(5)
                    if new_zip in valid_zips:
                        if new_zip != st.session_state.get("selected_zip"):
                            st.session_state.selected_zip = new_zip
                            st.rerun(scope="app")

    sel = st.session_state.get("selected_zip")
    main_fig = build_main_map(ev_df, scored, overview_gj, sel)
    st.plotly_chart(
        main_fig,
        use_container_width = True,
        config    = {"scrollZoom": True},
        on_select = "rerun",
        key       = chart_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    zcta_gdf, ev_df, scored = load_all()
    streets_gdf = load_streets_with_adt()
    valid_zips  = set(scored["ZIP"])

    for key, default in [("selected_zip", None), ("map_version", 0),
                         ("last_event_key", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    sel = st.session_state.selected_zip

    st.title("⚡ Seattle EV Station Explorer")
    st.info(
        "**How to use this map**  \n"
        "- The left map shows **average daily traffic flow** by ZIP code — darker orange = higher traffic.  \n"
        "- Green dots are **EV charging stations**. Hover over a dot to see station details.  \n"
        "- **Click any ZIP area or station** to load the road network map and charging stats on the right.",
        icon="ℹ️",
    )

    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        map_fragment()

    with right_col:
        if sel:
            scored_row = scored[scored["ZIP"] == sel]
            ev_in_zip  = ev_df[ev_df["ZIP"] == sel]
            city_val   = scored_row["city"].iloc[0] if not scored_row.empty else "—"
            pop_val    = int(scored_row["population"].iloc[0]) if not scored_row.empty else None
            income_val = scored_row["med_income"].iloc[0] if not scored_row.empty else None
            ev_reg_val = scored_row["ev_registrations"].iloc[0] if not scored_row.empty else None

            hdr_col, btn_col = st.columns([5, 1])
            hdr_col.markdown(f"### {city_val} · ZIP {sel}")
            if btn_col.button("✕ Clear", use_container_width=True):
                st.session_state.selected_zip   = None
                st.session_state.last_event_key = None
                st.session_state.map_version   += 1
                st.rerun()

            # Road map
            st.markdown(
                "**Road traffic volume** (ADT – Average Daily Traffic)",
                help="ADT = avg_daily_flow aggregated from Seattle 2025–2026 traffic study data. "
                     "Road lines are coloured by measured traffic volume. "
                     "Gray lines have no traffic measurement data."
            )
            road_fig = cached_road_fig(sel, streets_gdf, zcta_gdf, ev_df)
            st.plotly_chart(road_fig, use_container_width=True,
                            config={"scrollZoom": True})

            # Stats
            st.markdown("**Population & Charging**")
            c1, c2 = st.columns(2)
            c1.metric("Population",    f"{pop_val:,}" if pop_val else "N/A")
            c2.metric("Stations",      len(ev_in_zip))
            c3, c4 = st.columns(2)
            c3.metric("Level2 Spots",  int(ev_in_zip["EV Level2 EVSE Num"].sum()),
                      help="Level 2 AC charging ports (240V). Typical charge time: 4–12 hrs.")
            c4.metric("DC Fast Spots", int(ev_in_zip["EV DC Fast Count"].sum()),
                      help="DC Fast Charging ports (50–350+ kW). Charge to ~80% in 20–45 min.")

            c5, c6 = st.columns(2)
            if not scored_row.empty:
                c5.metric("Avg Daily Flow",
                          f"{scored_row['mean_ADT'].iloc[0]:,.0f}",
                          help="Average of avg_daily_flow across all traffic study points in this ZIP.")
            if pd.notna(income_val):
                c6.metric("Median Household Income",
                          f"${int(income_val):,}",
                          help="Median household income estimate for this ZIP code.")

            if pd.notna(ev_reg_val):
                st.metric("EV Registrations",
                          f"{int(ev_reg_val):,}",
                          help="Total EV vehicle registrations in this ZIP code.")

            # Station table
            st.markdown(f"**Stations in ZIP {sel}**")
            if not ev_in_zip.empty:
                disp = {
                    "Station Name":      "Station",
                    "EV Level2 EVSE Num":"Level2",
                    "EV DC Fast Count":  "DC Fast",
                    "EV Network":        "Network",
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
            st.markdown("Select any ZIP code on the map to see road traffic, population, and charging station details.")
            st.divider()
            seattle_ev = ev_df[ev_df["ZIP"].isin(valid_zips)]
            st.markdown("**City-wide summary**")
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("ZIPs with data",     len(valid_zips))
            r1c2.metric("Total EV Stations",   len(seattle_ev))
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Total Level2 Spots",  int(seattle_ev["EV Level2 EVSE Num"].sum()),
                        help="Level 2 AC charging ports (240V). Typical charge time: 4–12 hrs.")
            r2c2.metric("Total DC Fast Spots", int(seattle_ev["EV DC Fast Count"].sum()),
                        help="DC Fast Charging ports (50–350+ kW). Charge to ~80% in 20–45 min.")

    st.caption("Data sources: Traffic flow and demographics from Seattle 2025–2026 traffic study. "
               "Road network from Seattle Open Data (Street Network Database). "
               "EV station data from AFDC (Alternative Fuels Data Center).")


if __name__ == "__main__":
    main()
