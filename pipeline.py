"""
GeoEarthworm pipeline.

Turns a borehole CSV (+ optional surface DEM) into bedrock-level and
overburden-thickness GeoTIFFs plus a summary PNG. Shared by the CLI
(run_geoearthworm.py) and the Streamlit UI (app.py) so both stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

from geo_earthworm import GeoEarthworm

PathLike = Union[str, Path]

# Depth: one-hue-family ramp, light = shallow, dark = deep. Altitude keeps the conventional terrain ramp.
PRIMARY_CMAPS = {
    "Bedrock depth (m)": "YlGnBu",
    "Bedrock altitude (m)": "terrain",
}

METHOD_LABELS = {
    "idw": "IDW",
    "kriging": "Kriging",
    "spline": "Spline",
}


@dataclass
class PipelineResult:
    output_dir: Path
    image_path: Path
    input_stats: dict
    method: str
    bedrock_depth_path: Optional[Path] = None
    bedrock_altitude_path: Optional[Path] = None
    overburden_path: Optional[Path] = None
    bedrock_depth_stats: Optional[dict] = None
    bedrock_altitude_stats: Optional[dict] = None
    overburden_stats: Optional[dict] = None
    warnings: list = field(default_factory=list)


def _auto_bbox(lon: np.ndarray, lat: np.ndarray, buffer_frac: float) -> Tuple[float, float, float, float]:
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_pad = (lon_max - lon_min) * buffer_frac or 1.0
    lat_pad = (lat_max - lat_min) * buffer_frac or 1.0
    return (lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad)


def _auto_resolution(bbox: Tuple[float, float, float, float], target_cells: int = 300) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    span = max(max_lon - min_lon, max_lat - min_lat)
    return span / target_cells


def _mask_nodata(array: np.ndarray, nodata: float) -> np.ndarray:
    return np.where(array == nodata, np.nan, array)


def _build_grid(bbox: Tuple[float, float, float, float], resolution: float):
    """Build the output grid + its GeoTIFF affine transform from a bbox and resolution."""
    min_lon, min_lat, max_lon, max_lat = bbox
    cols = int(np.ceil((max_lon - min_lon) / resolution))
    rows = int(np.ceil((max_lat - min_lat) / resolution))

    x = np.linspace(min_lon, max_lon, cols)
    y = np.linspace(max_lat, min_lat, rows)  # top to bottom, matches raster row order
    grid_lon, grid_lat = np.meshgrid(x, y)

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, cols, rows)
    return grid_lon, grid_lat, transform


def _interpolate_kriging(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
    grid_lon: np.ndarray, grid_lat: np.ndarray, variogram_model: str,
) -> np.ndarray:
    from pykrige.ok import OrdinaryKriging

    ok = OrdinaryKriging(
        lon, lat, values,
        variogram_model=variogram_model,
        verbose=False, enable_plotting=False,
    )
    x = grid_lon[0, :]
    y = grid_lat[:, 0]
    z, _ = ok.execute("grid", x, y)
    return np.asarray(z, dtype=np.float64)


def _interpolate_spline(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
    grid_lon: np.ndarray, grid_lat: np.ndarray,
    smoothing: float, max_neighbors: Optional[int],
) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator

    points = np.column_stack([lon, lat])
    rbf = RBFInterpolator(
        points, values, kernel="thin_plate_spline",
        smoothing=smoothing, neighbors=max_neighbors,
    )
    query_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    return rbf(query_points).reshape(grid_lon.shape)


def _interpolate(
    method: str,
    interpolator: GeoEarthworm,
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
    grid_lon: np.ndarray, grid_lat: np.ndarray,
    *,
    power: float,
    max_neighbors: Optional[int],
    max_distance: Optional[float],
    smoothing: float,
    variogram_model: str,
) -> np.ndarray:
    """Dispatch to the chosen interpolation method. Returns a grid with NaN for unresolved cells."""
    if method == "idw":
        return interpolator.IDW(
            grid_lon, grid_lat, power=power, max_neighbors=max_neighbors,
            max_distance=max_distance, smoothing=smoothing,
        )
    if method == "kriging":
        return _interpolate_kriging(lon, lat, values, grid_lon, grid_lat, variogram_model)
    if method == "spline":
        return _interpolate_spline(lon, lat, values, grid_lon, grid_lat, smoothing, max_neighbors)
    raise ValueError(f"Unknown interpolation method: {method!r}. Expected one of {sorted(METHOD_LABELS)}.")


def _resample_to_grid(src_path: str, ref_path: str) -> np.ndarray:
    """Reproject/resample a raster onto the exact grid of a reference raster."""
    with rio.open(ref_path) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    with rio.open(src_path) as src:
        out = np.full(ref_shape, np.nan, dtype=np.float64)
        reproject(
            source=rio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        if src.nodata is not None:
            out = _mask_nodata(out, src.nodata)
    return out


def _write_geotiff(path: Path, array: np.ndarray, transform, epsg: int, nodata: float) -> None:
    out = np.where(np.isnan(array), nodata, array)
    with rio.open(
        path, "w", driver="GTiff",
        height=out.shape[0], width=out.shape[1], count=1,
        dtype=out.dtype, crs=CRS.from_epsg(epsg),
        transform=transform, nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(out, 1)


def _render_panel(ax, grid, cmap, label, title, bbox, epsg, fig):
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    # Geographic degrees: stretch y so a degree of longitude isn't drawn as wide as a degree of latitude.
    geographic = CRS.from_epsg(epsg).is_geographic
    aspect = 1 / np.cos(np.radians((bbox[1] + bbox[3]) / 2)) if geographic else 1
    im = ax.imshow(grid, extent=extent, origin="upper", cmap=cmap, aspect=aspect)
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_xlabel("Longitude (°)" if geographic else f"Easting (EPSG:{epsg})")
    ax.set_ylabel("Latitude (°)" if geographic else f"Northing (EPSG:{epsg})")
    ax.set_title(title, loc="left")
    ax.ticklabel_format(useOffset=False, style="plain")
    for spine in ax.spines.values():
        spine.set_color("#9aa3a0")
    ax.tick_params(colors="#4a5552")
    return im


def _plot(
    path: Path,
    bbox: Tuple[float, float, float, float],
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    primary_grid: np.ndarray,
    primary_label: str,
    overburden_grid: Optional[np.ndarray],
    epsg: int,
    method_label: str,
    dpi: int = 200,
    show_points: bool = True,
) -> None:
    ncols = 2 if overburden_grid is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    axes = np.atleast_1d(axes)

    cmap = PRIMARY_CMAPS[primary_label]
    im = _render_panel(
        axes[0], primary_grid, cmap, primary_label,
        f"{primary_label}, {method_label}", bbox, epsg, fig,
    )
    if show_points:
        # Same norm as the grid so a dot's colour reads off the same colour bar.
        axes[0].scatter(lon, lat, c=values, cmap=cmap, norm=im.norm, s=36,
                        edgecolors="white", linewidths=1.2, zorder=5)

    if overburden_grid is not None:
        _render_panel(
            axes[1], overburden_grid, "YlOrBr", "Overburden thickness (m)",
            "Overburden thickness (m)", bbox, epsg, fig,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    csv_path: PathLike,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    value_col: str = "bedrock_depth",
    epsg: int = 4326,
    is_altitude: bool = False,
    dem_path: Optional[PathLike] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    buffer_frac: float = 0.1,
    resolution: Optional[float] = None,
    method: str = "idw",
    power: float = 2.0,
    max_neighbors: Optional[int] = None,
    max_distance: Optional[float] = None,
    smoothing: float = 0.0,
    variogram_model: str = "spherical",
    nodata: float = -9999.0,
    output_dir: PathLike = "geoearthworm_output",
    dpi: int = 200,
    show_points: bool = True,
    on_step: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Run the full borehole -> bedrock level (+ overburden) -> image pipeline.

    on_step, if given, is called with a short description as each stage starts.
    """
    step = on_step or (lambda _msg: None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: list = []
    csv_name = Path(csv_path).name

    step("Reading borehole data")
    try:
        lon, lat, values, epsg = GeoEarthworm.load_points_from_csv(
            filepath=str(csv_path), lon_col=lon_col, lat_col=lat_col,
            bedrock_col=value_col, epsg=epsg,
        )
    except KeyError as e:
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        raise ValueError(
            f"Column {e} not found in '{csv_name}'. Available columns: {cols}"
        ) from e

    if len(lon) < 3:
        raise ValueError(
            f"Only {len(lon)} valid point(s) found in '{csv_name}' after dropping rows with "
            "missing values — at least 3 are required for interpolation."
        )

    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown interpolation method: {method!r}. Expected one of {sorted(METHOD_LABELS)}.")

    bbox = bbox or _auto_bbox(lon, lat, buffer_frac)
    resolution = resolution or _auto_resolution(bbox)

    interpolator = GeoEarthworm(lon=lon, lat=lat, bedrock=values, epsg=epsg, is_altitude=is_altitude)
    input_stats = interpolator.get_statistics()

    # Native raster: matches the input's own representation (depth or altitude)
    # directly from the chosen method — never needs a DEM.
    step(f"Interpolating with {METHOD_LABELS[method]}")
    grid_lon, grid_lat, native_transform = _build_grid(bbox, resolution)
    native_grid = _interpolate(
        method, interpolator, lon, lat, values, grid_lon, grid_lat,
        power=power, max_neighbors=max_neighbors, max_distance=max_distance,
        smoothing=smoothing, variogram_model=variogram_model,
    )

    value_std = float(np.std(values))
    result_std = float(np.nanstd(native_grid))
    if value_std > 0 and result_std < 0.02 * value_std:
        remedy = {
            "idw": "try a higher IDW power or less smoothing",
            "kriging": "try a different variogram model, or switch to IDW",
            "spline": "try less smoothing, or switch to IDW",
        }[method]
        warnings_out.append(
            f"The interpolated surface is nearly flat (barely varies across the map). This usually "
            f"means {METHOD_LABELS[method]} couldn't find enough spatial structure in your data; {remedy}."
        )

    native_path = output_dir / ("bedrock_altitude.tif" if is_altitude else "bedrock_depth.tif")
    _write_geotiff(native_path, native_grid, native_transform, epsg, nodata)

    depth_path = altitude_path = overburden_path = None
    depth_grid = altitude_grid = overburden_grid = None
    depth_stats = altitude_stats = overburden_stats = None

    if is_altitude:
        altitude_path, altitude_grid = native_path, native_grid
    else:
        depth_path, depth_grid = native_path, native_grid

    if dem_path is not None:
        step("Combining with the surface elevation model")
        # Derive whichever of depth/altitude is missing by resampling the DEM
        # onto the native raster's grid (reprojects if the DEM's CRS differs).
        dem_on_grid = _resample_to_grid(str(dem_path), str(native_path))
        if is_altitude:
            depth_grid = dem_on_grid - altitude_grid
            depth_path = output_dir / "bedrock_depth.tif"
            _write_geotiff(depth_path, depth_grid, native_transform, epsg, nodata)
        else:
            altitude_grid = dem_on_grid - depth_grid
            altitude_path = output_dir / "bedrock_altitude.tif"
            _write_geotiff(altitude_path, altitude_grid, native_transform, epsg, nodata)

        overburden_path = output_dir / "overburden.tif"
        overburden_grid, _ = GeoEarthworm.calculate_overburden(
            interpolated_path=str(altitude_path), dem_path=str(dem_path),
            output_path=str(overburden_path), nodata_value=nodata,
        )
        overburden_grid = _mask_nodata(overburden_grid, nodata)

        depth_stats = interpolator.get_statistics(depth_grid, nodata_value=nodata)
        altitude_stats = interpolator.get_statistics(altitude_grid, nodata_value=nodata)
        overburden_stats = interpolator.get_statistics(overburden_grid, nodata_value=nodata)
    else:
        native_stats = interpolator.get_statistics(native_grid, nodata_value=nodata)
        if is_altitude:
            altitude_stats = native_stats
        else:
            depth_stats = native_stats
        kind = "bedrock altitude" if is_altitude else "bedrock depth"
        warnings_out.append(
            f"No DEM supplied: overburden thickness was not computed, and only the native "
            f"{kind} raster was produced."
        )

    # Plot the quantity the boreholes measured, so the dots and the grid share one colour scale.
    if is_altitude:
        primary_grid, primary_label = altitude_grid, "Bedrock altitude (m)"
    else:
        primary_grid, primary_label = depth_grid, "Bedrock depth (m)"

    step("Drawing the summary map")
    image_path = output_dir / "bedrock_overburden_map.png"
    _plot(
        image_path, bbox, lon, lat, values,
        primary_grid=primary_grid, primary_label=primary_label,
        overburden_grid=overburden_grid, epsg=epsg, dpi=dpi,
        method_label=METHOD_LABELS[method], show_points=show_points,
    )

    return PipelineResult(
        output_dir=output_dir,
        image_path=image_path,
        input_stats=input_stats,
        method=METHOD_LABELS[method],
        bedrock_depth_path=depth_path,
        bedrock_altitude_path=altitude_path,
        overburden_path=overburden_path,
        bedrock_depth_stats=depth_stats,
        bedrock_altitude_stats=altitude_stats,
        overburden_stats=overburden_stats,
        warnings=warnings_out,
    )
