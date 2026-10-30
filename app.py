"""
GeoEarthworm web app.

Upload a borehole CSV (+ optional surface DEM), interpolate bedrock level
(IDW, kriging, or spline), and view/download the results as GeoTIFFs ready
for GIS software plus a summary image.

Run with:
    streamlit run app.py
"""

import base64
import io
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio as rio
import streamlit as st
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform as warp_transform

from pipeline import PRIMARY_CMAPS, run

MAX_RUNS = 6  # runs kept in memory for comparing methods

EPSG_PRESETS = {
    "GPS latitude/longitude, WGS 84 (EPSG:4326)": 4326,
    "Latitude/longitude, NAD83 (EPSG:4269)": 4269,
    "Web Mercator (EPSG:3857)": 3857,
    "Another coordinate system": None,
}

COLUMN_GUESSES = {
    "lon": ("longitude", "lon", "long", "lng", "x", "easting"),
    "lat": ("latitude", "lat", "y", "northing"),
    "value": ("bedrock_depth", "bedrock_altitude", "bedrock", "depth", "altitude", "elevation", "elev"),
}

st.set_page_config(page_title="GeoEarthworm", page_icon=":world_map:", layout="wide")

OCHRE = "#8A5A14"  # keep in sync with primaryColor in .streamlit/config.toml

# Streamlit's own focus ring is a faint primary-colour glow; give keyboard users a solid one.
st.markdown(
    f"""<style>
    button:focus-visible, [role="button"]:focus-visible, input:focus-visible {{
        outline: 2px solid {OCHRE} !important; outline-offset: 2px;
    }}
    ::selection {{ background: #E9C98A; }}
    h2 {{ font-size: 1.4rem !important; padding-top: 1.25rem !important; }}
    .gew-legend {{ display: flex; align-items: center; gap: .6rem; font-size: .85rem; margin: .4rem 0 .2rem; }}
    .gew-legend .bar {{ flex: 1; height: .7rem; border-radius: 2px; }}
    [data-testid="stApp"] {{ background-size: cover; background-position: center; background-attachment: fixed; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    </style>""",
    unsafe_allow_html=True,
)
background_slot = st.empty()  # filled at the end, once we know the latest surface

if "runs" not in st.session_state:
    st.session_state.runs = []
    st.session_state.run_counter = 0


# ---------- helpers ----------

def guess_column(columns, kind, fallback):
    lower = [c.lower().strip() for c in columns]
    for key in COLUMN_GUESSES[kind]:
        if key in lower:
            return lower.index(key)
    for key in COLUMN_GUESSES[kind]:
        if len(key) > 2:  # single letters like "x" would match far too much
            for i, c in enumerate(lower):
                if key in c:
                    return i
    return min(fallback, len(columns) - 1)


def to_lonlat(x, y, epsg):
    if epsg == 4326:
        return np.asarray(x, float), np.asarray(y, float)
    lon, lat = warp_transform(CRS.from_epsg(epsg), CRS.from_epsg(4326), list(x), list(y))
    return np.asarray(lon), np.asarray(lat)


def rgba_for(values, cmap_name, vmin, vmax):
    norm = (np.asarray(values, float) - vmin) / ((vmax - vmin) or 1)
    return matplotlib.colormaps[cmap_name](np.clip(norm, 0, 1))


def raster_overlay(tif_bytes, cmap_name):
    """Reproject a GeoTIFF to lon/lat and colour it, for a pydeck BitmapLayer."""
    with rio.MemoryFile(tif_bytes) as mem, mem.open() as src:
        dst = CRS.from_epsg(4326)
        transform, width, height = calculate_default_transform(src.crs, dst, src.width, src.height, *src.bounds)
        grid = np.full((height, width), np.nan)
        reproject(
            rio.band(src, 1), grid, src_transform=src.transform, src_crs=src.crs,
            src_nodata=src.nodata, dst_transform=transform, dst_crs=dst,
            dst_nodata=np.nan, resampling=Resampling.bilinear,
        )
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    rgba = rgba_for(grid, cmap_name, vmin, vmax)
    rgba[..., 3] = np.where(np.isnan(grid), 0, 0.85)
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bounds": [west, south, east, north],
        "vmin": vmin, "vmax": vmax, "cmap": cmap_name,
    }


def contour_background(grid):
    """Faint ochre contour lines of a surface (north-up array), as an SVG data URL for the page background."""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.contour(np.flipud(grid), levels=14, colors=OCHRE, linewidths=0.6, alpha=0.16)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def default_background():
    """Contours of an invented terrain, shown until the user has a surface of their own."""
    yy, xx = np.mgrid[0:1:160j, 0:1.6:256j]
    hills = np.random.default_rng(7).uniform([-1, 0, 0, 0.08], [1, 1.6, 1, 0.3], (14, 4))
    z = sum(a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)) for a, cx, cy, s in hills)
    return contour_background(z)


def show_image(png_bytes, alt):
    # st.image writes alt="0"; an <img> tag lets screen readers hear what the map shows.
    src = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    st.markdown(f'<img src="{src}" alt="{alt}" style="width:100%; height:auto">', unsafe_allow_html=True)


def describe_settings(s):
    if s["method"] == "idw":
        text = f"IDW, power {s['power']:g}"
    elif s["method"] == "kriging":
        text = f"Kriging, {s['variogram_model']} variogram"
    else:
        text = f"Spline, smoothing {s['smoothing']:g}"
    if s["method"] == "idw" and s["smoothing"]:
        text += f", smoothing {s['smoothing']:g}"
    if s["max_neighbors"]:
        text += f", {s['max_neighbors']} neighbours"
    return text


def legend(overlay, label):
    stops = ", ".join(
        matplotlib.colors.to_hex(c) for c in matplotlib.colormaps[overlay["cmap"]](np.linspace(0, 1, 8))
    )
    st.markdown(
        f'<div class="gew-legend"><span>{overlay["vmin"]:,.1f} m</span>'
        f'<div class="bar" style="background: linear-gradient(to right, {stops})" role="img" '
        f'aria-label="{label} colour scale from {overlay["vmin"]:,.1f} to {overlay["vmax"]:,.1f} metres"></div>'
        f'<span>{overlay["vmax"]:,.1f} m</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(label)


def stat_tiles(title, stats):
    st.markdown(f"**{title}**")
    cols = st.columns(4)
    for col, key in zip(cols, ("min", "mean", "max", "std")):
        label = {"min": "Minimum", "mean": "Mean", "max": "Maximum", "std": "Std. deviation"}[key]
        value = stats.get(key)
        col.metric(label, "–" if value is None or np.isnan(value) else f"{value:,.1f} m")


def run_map(run_record, show_points):
    overlay = run_record["overlay"]
    pts = run_record["points"]
    # pdk.types.String: a plain str would be prefixed as a JS expression and break on "data:".
    layers = [pdk.Layer("BitmapLayer", image=pdk.types.String(overlay["image"]), bounds=overlay["bounds"])]
    if show_points:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=pts, get_position=["lon", "lat"], get_fill_color="color",
            get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=1.5,
            get_radius=6, radius_units=pdk.types.String("pixels"), pickable=True,
        ))
    view = pdk.data_utils.compute_view(pts[["lon", "lat"]])
    deck = pdk.Deck(
        layers=layers, initial_view_state=view, map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"html": "<b>{label}</b><br/>{where}"} if show_points else False,
    )
    st.pydeck_chart(deck, height=520)
    legend(overlay, run_record["primary_label"])


# ---------- page ----------

st.title("GeoEarthworm")
st.caption("Interpolate bedrock depth and overburden thickness from borehole measurements.")

settings_col, results_col = st.columns([1, 2.1], gap="large")

df = None
can_run = False
settings = {}

with settings_col:
    st.header("1. Borehole data")
    csv_file = st.file_uploader(
        "Borehole CSV", type=["csv"],
        help="A spreadsheet (.csv) with one row per borehole. It needs at least a "
             "longitude column, a latitude column, and a bedrock value column.",
    )

    if csv_file is not None:
        try:
            df = pd.read_csv(io.BytesIO(csv_file.getvalue()))
        except Exception as e:  # malformed CSVs raise a variety of parser errors
            st.error(f"Couldn't read '{csv_file.name}' as a CSV file: {e}")
        if df is not None and len(df.columns) < 3:
            st.error(f"'{csv_file.name}' has {len(df.columns)} column(s). It needs at least three: "
                     "longitude, latitude and a bedrock value.")
            df = None

    if df is not None:
        columns = list(df.columns)
        lon_col = st.selectbox(
            "Longitude / X column", columns, index=guess_column(columns, "lon", 0),
            help="The column holding each borehole's east-west position.",
        )
        lat_col = st.selectbox(
            "Latitude / Y column", columns, index=guess_column(columns, "lat", 1),
            help="The column holding each borehole's north-south position.",
        )
        value_col = st.selectbox(
            "Bedrock value column", columns, index=guess_column(columns, "value", 2),
            help="The column holding the bedrock measurement: a depth or an altitude, as set below.",
        )
        looks_like_altitude = any(k in value_col.lower() for k in ("alt", "elev"))
        values_kind = st.radio(
            "This column holds",
            ["Bedrock depth (below surface)", "Bedrock altitude (elevation)"],
            index=1 if looks_like_altitude else 0,
            help="Depth = how far down from the surface to bedrock. Altitude = the bedrock "
                 "surface's own elevation above sea level. Pick whichever matches your data.",
        )

        crs_label = st.selectbox(
            "Coordinate system", list(EPSG_PRESETS),
            help="The coordinate system your X/Y values use. Ordinary GPS coordinates are WGS 84. "
                 "Projected data in metres (e.g. UTM) needs its own EPSG code.",
        )
        epsg = EPSG_PRESETS[crs_label]
        if epsg is None:
            epsg = int(st.number_input("EPSG code", value=32619, step=1, format="%d",
                                       help="Look up your code at epsg.io, e.g. 32619 for WGS 84 / UTM zone 19N."))
        try:
            crs = CRS.from_epsg(epsg)
        except CRSError:
            st.error(f"EPSG:{epsg} isn't a coordinate system code we recognise. Check it at epsg.io.")
            crs = None

        x = pd.to_numeric(df[lon_col], errors="coerce")
        y = pd.to_numeric(df[lat_col], errors="coerce")
        v = pd.to_numeric(df[value_col], errors="coerce")
        valid = x.notna() & y.notna() & v.notna()
        n_valid = int(valid.sum())

        if crs is not None and n_valid:
            in_degree_range = x[valid].abs().max() <= 180 and y[valid].abs().max() <= 90
            if crs.is_geographic and not in_degree_range:
                st.warning("These coordinates are too large to be latitude/longitude. They look like a "
                           "projected system in metres, so pick 'Another coordinate system' and enter its EPSG code.")
            elif not crs.is_geographic and in_degree_range:
                st.warning("These coordinates look like latitude/longitude degrees, but the chosen system "
                           "is projected. WGS 84 (EPSG:4326) is probably right.")

        if n_valid < 3:
            st.error(f"Only {n_valid} row(s) have numbers in all three chosen columns. At least 3 are needed.")
        can_run = crs is not None and n_valid >= 3

    st.header("2. Surface elevation")
    dem_file = st.file_uploader(
        "DEM GeoTIFF (optional)", type=["tif", "tiff"],
        help="A surface elevation raster (GeoTIFF). Needed to compute overburden thickness, "
             "and to derive bedrock altitude from depth (or vice versa).",
    )

    st.header("3. Method")
    method_label = st.selectbox(
        "Interpolation method",
        ["IDW (Inverse Distance Weighting)", "Kriging", "Spline"],
        help="How values between boreholes are estimated. IDW is fast and predictable. "
             "Kriging uses the data's own spatial statistics and can be more accurate, but needs "
             "enough points with real spatial correlation. Spline fits a smooth surface through "
             "(or near) every point. IDW is a solid default.",
    )
    method = {"IDW (Inverse Distance Weighting)": "idw", "Kriging": "kriging", "Spline": "spline"}[method_label]

    power, smoothing, variogram_model, max_neighbors = 2.0, 0.0, "spherical", 0
    if method == "idw":
        power = st.slider(
            "IDW power", 0.5, 5.0, 2.0, 0.5,
            help="How strongly nearby points dominate. Higher values give sharper transitions "
                 "around each borehole; lower values give smoother maps. 2.0 is a good default.",
        )
    elif method == "kriging":
        variogram_model = st.selectbox(
            "Variogram model", ["spherical", "gaussian", "exponential", "linear", "power"],
            help="How bedrock values become less correlated with distance. Spherical and gaussian "
                 "suit most geological data; try another if the result looks flat or unrealistic.",
        )
    if method in ("idw", "spline"):
        smoothing = st.slider(
            "Smoothing", 0.0, 1.0, 0.0, 0.05,
            help="0 passes exactly through every borehole value, which can look spiky with noisy "
                 "data. Increase it for a looser, smoother fit.",
        )

    with st.expander("Advanced settings"):
        if method in ("idw", "spline"):
            max_neighbors = st.number_input(
                "Max neighbours (0 = use all)", value=0, min_value=0,
                help="Limits each map cell's estimate to its N nearest boreholes. 0 uses all points "
                     "(best for small datasets); a smaller number is faster on large datasets.",
            )
        resolution_auto = st.checkbox(
            "Choose grid resolution automatically", value=True,
            help="Picks the output pixel size from your data's extent (about 300 cells across).",
        )
        resolution = None
        if not resolution_auto:
            resolution = st.number_input(
                "Grid resolution (coordinate units)", value=0.01, format="%.5f",
                help="Size of each output pixel, in your coordinate system's units: degrees for "
                     "latitude/longitude, metres for most projected systems.",
            )
        buffer = st.slider(
            "Margin around boreholes", 0.0, 0.5, 0.1, 0.05,
            help="Extends the map beyond the outermost boreholes by this fraction of the data's "
                 "width/height.",
        )

    settings = dict(
        csv=csv_file.name if csv_file else None, dem=dem_file.name if dem_file else None,
        lon_col=lon_col if df is not None else None, lat_col=lat_col if df is not None else None,
        value_col=value_col if df is not None else None,
        is_altitude=df is not None and values_kind.startswith("Bedrock altitude"),
        epsg=epsg if df is not None else None, method=method, power=power, smoothing=smoothing,
        variogram_model=variogram_model, max_neighbors=int(max_neighbors), resolution=resolution,
        buffer=buffer,
    )

    run_clicked = st.button("Run interpolation", type="primary", disabled=not can_run, width="stretch")
    if csv_file is None:
        st.caption("Upload a borehole CSV to enable this.")


with results_col:
    if run_clicked:
        with st.status("Running interpolation…", expanded=True) as status:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_dir = Path(tmp)
                    csv_path = tmp_dir / csv_file.name
                    csv_path.write_bytes(csv_file.getvalue())
                    dem_path = None
                    if dem_file is not None:
                        dem_path = tmp_dir / dem_file.name
                        dem_path.write_bytes(dem_file.getvalue())

                    result = run(
                        csv_path=csv_path, lon_col=settings["lon_col"], lat_col=settings["lat_col"],
                        value_col=settings["value_col"], epsg=settings["epsg"],
                        is_altitude=settings["is_altitude"],
                        dem_path=str(dem_path) if dem_path else None,
                        buffer_frac=buffer, resolution=resolution, method=method, power=power,
                        max_neighbors=settings["max_neighbors"] or None, smoothing=smoothing,
                        variogram_model=variogram_model, output_dir=tmp_dir / "output",
                        on_step=status.write,
                    )

                    # Read everything out now: the temp directory is gone after this block.
                    status.write("Preparing the interactive map")
                    downloads = [
                        (label, Path(p).name, Path(p).read_bytes()) for label, p in [
                            ("Summary image (PNG)", result.image_path),
                            ("Bedrock depth (GeoTIFF)", result.bedrock_depth_path),
                            ("Bedrock altitude (GeoTIFF)", result.bedrock_altitude_path),
                            ("Overburden (GeoTIFF)", result.overburden_path),
                        ] if p is not None
                    ]
                    native_path = result.bedrock_altitude_path if settings["is_altitude"] else result.bedrock_depth_path
                    primary_label = "Bedrock altitude (m)" if settings["is_altitude"] else "Bedrock depth (m)"
                    native_bytes = Path(native_path).read_bytes()
                    with rio.MemoryFile(native_bytes) as mem, mem.open() as src:
                        grid_shape, cell = (src.height, src.width), src.res[0]
                        native_grid = src.read(1, masked=True).astype(float).filled(np.nan)
                    overlay = raster_overlay(native_bytes, PRIMARY_CMAPS[primary_label])

                pts_lon, pts_lat = to_lonlat(x[valid], y[valid], settings["epsg"])
                pts = pd.DataFrame({"lon": pts_lon, "lat": pts_lat, "value": v[valid].to_numpy()})
                pts["color"] = [
                    [int(c * 255) for c in rgba[:3]]
                    for rgba in rgba_for(pts["value"], overlay["cmap"], overlay["vmin"], overlay["vmax"])
                ]
                pts["label"] = [f"{val:,.1f} m" for val in pts["value"]]
                pts["where"] = [f"{la:.5f}, {lo:.5f}" for lo, la in zip(pts["lon"], pts["lat"])]

                st.session_state.run_counter += 1
                st.session_state.runs.append({
                    "name": f"Run {st.session_state.run_counter}: {describe_settings(settings)}",
                    "settings": dict(settings),
                    "warnings": result.warnings,
                    "image": downloads[0][2],
                    "downloads": downloads,
                    "stats": [(t, s) for t, s in [
                        ("Bedrock depth", result.bedrock_depth_stats),
                        ("Bedrock altitude", result.bedrock_altitude_stats),
                        ("Overburden thickness", result.overburden_stats),
                    ] if s],
                    "input_stats": result.input_stats,
                    "summary": (
                        f"{result.input_stats['count']} boreholes interpolated onto a "
                        f"{grid_shape[1]} × {grid_shape[0]} grid "
                        f"(cell size {cell:.4g} {'°' if CRS.from_epsg(settings['epsg']).is_geographic else 'units'})."
                    ),
                    "overlay": overlay,
                    "points": pts,
                    "primary_label": primary_label,
                    "background": contour_background(native_grid),
                })
                st.session_state.runs = st.session_state.runs[-MAX_RUNS:]
                status.update(label="Interpolation finished", state="complete", expanded=False)
            except (ValueError, FileNotFoundError, CRSError) as e:
                status.update(label="Interpolation failed", state="error", expanded=True)
                st.error(f"{e}")

    runs = st.session_state.runs
    if not runs:
        if df is None:
            st.header("Get started")
            st.markdown("Upload a borehole CSV on the left. It needs at least three columns: X/longitude, "
                    "Y/latitude, and a bedrock value (depth below the surface, or elevation). "
                    "You'll pick which column is which after uploading.")
        else:
            st.header("Your data")
            st.caption(f"{n_valid} of {len(df)} rows have numbers in all three chosen columns. "
                       "Check them below, then run the interpolation.")
            if crs is not None and n_valid:
                try:
                    p_lon, p_lat = to_lonlat(x[valid], y[valid], epsg)
                    preview = pd.DataFrame({"lon": p_lon, "lat": p_lat})
                    st.pydeck_chart(pdk.Deck(
                        layers=[pdk.Layer(
                            "ScatterplotLayer", data=preview, get_position=["lon", "lat"],
                            get_fill_color=[138, 90, 20], get_line_color=[255, 255, 255], stroked=True,
                            line_width_min_pixels=1.5, get_radius=5, radius_units=pdk.types.String("pixels"),
                        )],
                        initial_view_state=pdk.data_utils.compute_view(preview),
                        map_style=pdk.map_styles.CARTO_LIGHT,
                    ), height=320)
                except Exception:  # bad coordinates for the chosen CRS; the warning above explains
                    st.caption("Can't place these points on a map with the chosen coordinate system.")
            st.dataframe(df.head(20), height=280)
    else:
        latest = runs[-1]
        st.header("Results")
        st.markdown(f"**{latest['name']}**  \n{latest['summary']}")
        if latest["settings"] != settings:
            st.caption("Your settings have changed since this run. Run the interpolation again to update it.")
        for w in latest["warnings"]:
            st.warning(w)

        tab_names = ["Map", "Summary image"] + (["Compare runs"] if len(runs) > 1 else [])
        tabs = st.tabs(tab_names)
        with tabs[0]:
            show_points = st.toggle("Show boreholes", value=True)
            run_map(latest, show_points)
        with tabs[1]:
            show_image(latest["image"], f"Summary map of {latest['primary_label'].lower()} for {latest['name']}")
        if len(runs) > 1:
            with tabs[2]:
                names = [r["name"] for r in runs]
                picked = st.multiselect("Runs to compare", names, default=names[-2:], max_selections=3)
                if picked:
                    for col, r in zip(st.columns(len(picked)), [r for r in runs if r["name"] in picked]):
                        with col:
                            st.markdown(f"**{r['name']}**")
                            show_image(r["image"], f"Summary map of {r['primary_label'].lower()} for {r['name']}")
                            primary = r["stats"][0][1] if r["stats"] else None
                            if primary:
                                st.caption(f"{r['stats'][0][0]}: mean {primary['mean']:,.1f} m, "
                                           f"range {primary['min']:,.1f} to {primary['max']:,.1f} m")

        st.subheader("Statistics")
        stat_tiles(f"Boreholes ({latest['input_stats']['count']} measured)", latest["input_stats"])
        for title, stats in latest["stats"]:
            stat_tiles(f"{title} (interpolated surface)", stats)

        st.subheader("Downloads")
        dl_cols = st.columns(len(latest["downloads"]))
        for col, (label, name, data) in zip(dl_cols, latest["downloads"]):
            col.download_button(label, data=data, file_name=name, on_click="ignore", width="stretch")

        if st.button("Clear all runs", type="tertiary"):
            st.session_state.runs = []
            st.rerun()

background = st.session_state.runs[-1]["background"] if st.session_state.runs else default_background()
background_slot.markdown(
    f'<style>[data-testid="stApp"] {{ background-image: url("{background}"); }}</style>',
    unsafe_allow_html=True,
)
