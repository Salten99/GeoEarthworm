"""
GeoEarthworm web app.

Upload a borehole CSV (+ optional surface DEM), interpolate bedrock level
(IDW, kriging, or spline), and view/download the results as GeoTIFFs ready
for GIS software plus a summary image.

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from pipeline import run

st.set_page_config(page_title="GeoEarthworm", page_icon=":world_map:", layout="wide")
st.title("GeoEarthworm — Bedrock & Overburden Interpolation")
st.caption("Upload borehole data to interpolate bedrock level, and optionally overburden thickness.")

with st.sidebar:
    st.header("1. Borehole data")
    csv_file = st.file_uploader(
        "Borehole CSV", type=["csv"],
        help="A spreadsheet (.csv) with one row per borehole. It needs at least a "
             "longitude column, a latitude column, and a bedrock value column.",
    )
    lon_col = st.text_input(
        "Longitude column", "longitude",
        help="The exact column header in your CSV that holds each point's longitude "
             "(east-west position).",
    )
    lat_col = st.text_input(
        "Latitude column", "latitude",
        help="The exact column header in your CSV that holds each point's latitude "
             "(north-south position).",
    )
    value_col = st.text_input(
        "Bedrock value column", "bedrock_depth",
        help="The exact column header in your CSV that holds the bedrock measurement "
             "for each point — either a depth or an altitude, as set below.",
    )
    values_kind = st.radio(
        "This column holds", ["Bedrock depth (below surface)", "Bedrock altitude (elevation)"],
        help="Depth = how far down from the surface to bedrock. Altitude = the bedrock "
             "surface's own elevation above sea level. Pick whichever matches your data.",
    )
    is_altitude = values_kind.startswith("Bedrock altitude")
    epsg = st.number_input(
        "EPSG code", value=4326, step=1,
        help="The coordinate system your longitude/latitude values use. 4326 is the "
             "standard for ordinary GPS coordinates and is correct for most CSV exports. "
             "Only change this if you know your data uses a different coordinate system.",
    )

    st.header("2. Surface elevation (optional)")
    dem_file = st.file_uploader(
        "DEM GeoTIFF", type=["tif", "tiff"],
        help="A surface elevation raster (GeoTIFF). Optional, but required to compute "
             "overburden thickness, and to derive bedrock altitude from depth (or vice versa).",
    )

    st.header("3. Interpolation settings")
    method_label = st.selectbox(
        "Interpolation method",
        ["IDW (Inverse Distance Weighting)", "Kriging", "Spline"],
        help="How the values between your boreholes are estimated. IDW is fast and "
             "predictable. Kriging uses the data's own spatial statistics and can be more "
             "accurate, but needs enough points with real spatial correlation. Spline fits "
             "a smooth surface through (or near) every point. IDW is a solid default.",
    )
    method = {
        "IDW (Inverse Distance Weighting)": "idw",
        "Kriging": "kriging",
        "Spline": "spline",
    }[method_label]

    power = 2.0
    max_neighbors = None
    max_distance = None
    variogram_model = "spherical"
    smoothing = 0.0

    if method == "idw":
        power = st.slider(
            "IDW power", 0.5, 5.0, 2.0, 0.5,
            help="Controls how strongly nearby points dominate the interpolation. Higher "
                 "values create sharper transitions around each borehole; lower values produce "
                 "smoother, more gradual maps. 2.0 is a good default.",
        )
        max_neighbors = st.number_input(
            "Max neighbors (0 = use all)", value=0, min_value=0,
            help="Limits how many nearby boreholes influence each map cell. 0 uses all "
                 "points (best for small datasets); a smaller number speeds up processing "
                 "for large datasets and keeps predictions more local.",
        )
        smoothing = st.slider(
            "Smoothing", 0.0, 1.0, 0.0, 0.05,
            help="Softens the interpolation near data points to avoid sharp spikes. 0 "
                 "passes exactly through every borehole value; higher values smooth it out.",
        )
    elif method == "kriging":
        variogram_model = st.selectbox(
            "Variogram model", ["spherical", "gaussian", "exponential", "linear", "power"],
            help="The statistical model of how bedrock values become less correlated with "
                 "distance. Spherical and gaussian work well for most geological data; try "
                 "another if the result looks flat or unrealistic.",
        )
    elif method == "spline":
        smoothing = st.slider(
            "Smoothing", 0.0, 1.0, 0.0, 0.05,
            help="0 makes the surface pass exactly through every borehole value, which can "
                 "look spiky with noisy data. Increase it for a looser, smoother fit.",
        )
        max_neighbors = st.number_input(
            "Max neighbors (0 = use all)", value=0, min_value=0,
            help="Limits each grid cell's estimate to its N nearest boreholes instead of "
                 "all of them. 0 uses all points; useful to speed up large datasets.",
        )

    resolution_auto = st.checkbox(
        "Auto grid resolution", value=True,
        help="When checked, the output pixel size is chosen automatically from your "
             "data's extent. Uncheck to set a specific resolution yourself.",
    )
    resolution = None
    if not resolution_auto:
        resolution = st.number_input(
            "Grid resolution (CRS units)", value=0.01, format="%.5f",
            help="The size of each output pixel, in the same units as your coordinate "
                 "system (degrees for EPSG:4326, meters for most projected systems). "
                 "Smaller values give a more detailed but slower map.",
        )
    buffer = st.slider(
        "Bounding-box buffer (fraction)", 0.0, 0.5, 0.1, 0.05,
        help="Expands the map beyond your borehole locations by this fraction of the "
             "data's width/height, so the map doesn't cut off right at your outermost points.",
    )
    show_points = st.checkbox(
        "Show borehole points on map", value=True,
        help="When checked, your original borehole locations are plotted as circles on "
             "top of the bedrock-level map. Uncheck to show just the interpolated surface.",
    )

    run_clicked = st.button(
        "Run interpolation", type="primary", disabled=csv_file is None,
        help="Runs the interpolation with the current settings and shows the resulting map below.",
    )

if csv_file is None:
    st.info("Upload a borehole CSV in the sidebar to get started.")
    st.markdown(
        "Your CSV needs at least three columns: longitude, latitude, and a bedrock value "
        "(depth below the surface, or elevation). Column names are configurable in the sidebar."
    )
elif run_clicked:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        csv_path = tmp_dir / csv_file.name
        csv_path.write_bytes(csv_file.getvalue())

        dem_path = None
        if dem_file is not None:
            dem_path = tmp_dir / dem_file.name
            dem_path.write_bytes(dem_file.getvalue())

        out_dir = tmp_dir / "output"

        try:
            with st.spinner("Interpolating..."):
                result = run(
                    csv_path=csv_path,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    value_col=value_col,
                    epsg=int(epsg),
                    is_altitude=is_altitude,
                    dem_path=str(dem_path) if dem_path else None,
                    buffer_frac=buffer,
                    resolution=resolution,
                    method=method,
                    power=power,
                    max_neighbors=(int(max_neighbors) or None) if max_neighbors is not None else None,
                    max_distance=max_distance,
                    smoothing=smoothing,
                    variogram_model=variogram_model,
                    output_dir=out_dir,
                    show_points=show_points,
                )
        except (ValueError, FileNotFoundError) as e:
            st.error(f"Interpolation failed: {e}")
        else:
            for w in result.warnings:
                st.warning(w)

            st.caption(f"Interpolation method: **{result.method}**")
            st.image(str(result.image_path), use_container_width=True)

            stat_blocks = [
                ("Input points", result.input_stats),
                ("Bedrock depth", result.bedrock_depth_stats),
                ("Bedrock altitude", result.bedrock_altitude_stats),
                ("Overburden", result.overburden_stats),
            ]
            stat_blocks = [(t, s) for t, s in stat_blocks if s]
            cols = st.columns(len(stat_blocks))
            for col, (title, stats) in zip(cols, stat_blocks):
                col.subheader(title)
                col.json(stats)

            st.subheader("Download outputs")
            downloads = [
                ("Summary image (PNG)", result.image_path),
                ("Bedrock depth (GeoTIFF)", result.bedrock_depth_path),
                ("Bedrock altitude (GeoTIFF)", result.bedrock_altitude_path),
                ("Overburden (GeoTIFF)", result.overburden_path),
            ]
            downloads = [(label, path) for label, path in downloads if path is not None]
            for col, (label, path) in zip(st.columns(len(downloads)), downloads):
                col.download_button(label, data=Path(path).read_bytes(), file_name=Path(path).name)
else:
    st.info("Set your options in the sidebar, then click **Run interpolation**.")
