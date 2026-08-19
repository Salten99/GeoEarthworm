"""
GeoEarthworm web app.

Upload a borehole CSV (+ optional surface DEM), run IDW interpolation, and
view/download the bedrock-level and overburden-thickness results.

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
    csv_file = st.file_uploader("Borehole CSV", type=["csv"])
    lon_col = st.text_input("Longitude column", "longitude")
    lat_col = st.text_input("Latitude column", "latitude")
    value_col = st.text_input("Bedrock value column", "bedrock_depth")
    values_kind = st.radio(
        "This column holds", ["Bedrock depth (below surface)", "Bedrock altitude (elevation)"]
    )
    is_altitude = values_kind.startswith("Bedrock altitude")
    epsg = st.number_input("EPSG code", value=4326, step=1)

    st.header("2. Surface elevation (optional)")
    dem_file = st.file_uploader("DEM GeoTIFF", type=["tif", "tiff"])
    st.caption("Required to compute overburden thickness, and to derive bedrock altitude from depth (or vice versa).")

    st.header("3. Interpolation settings")
    power = st.slider("IDW power", 0.5, 5.0, 2.0, 0.5)
    resolution_auto = st.checkbox("Auto grid resolution", value=True)
    resolution = None
    if not resolution_auto:
        resolution = st.number_input("Grid resolution (CRS units)", value=0.01, format="%.5f")
    max_neighbors = st.number_input("Max neighbors (0 = use all)", value=0, min_value=0)
    buffer = st.slider("Bounding-box buffer (fraction)", 0.0, 0.5, 0.1, 0.05)

    run_clicked = st.button("Run interpolation", type="primary", disabled=csv_file is None)

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
                    power=power,
                    max_neighbors=(int(max_neighbors) or None),
                    output_dir=out_dir,
                )
        except (ValueError, FileNotFoundError) as e:
            st.error(f"Interpolation failed: {e}")
        else:
            for w in result.warnings:
                st.warning(w)

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
