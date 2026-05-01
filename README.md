# GeoEarthworm

**Bedrock depth interpolation from scattered borehole measurements.**

GeoEarthworm provides rapid, easy-to-use bedrock interpolations using Inverse Distance Weighting (IDW). It supports coordinate transformations, optional DEM integration for bedrock-to-depth conversion, and GeoTIFF raster output.

---

## Features

- IDW interpolation with configurable power parameter
- Automatic bounding box detection from input data
- GeoTIFF output with LZW compression via rasterio
- Optional DEM integration for bedrock bedrock ↔ depth conversion
- CSV data loader with missing value handling
- Statistics utility for input and output arrays
- Supports any EPSG coordinate reference system

---

## Installation

```bash
git clone https://github.com/Salten99/geoearthworm.git
cd geo-earthworm
pip install -r requirements.txt
```

```bash
# Conda recommended install
conda install -c conda-forge rasterio scipy numpy pandas
```

---

## Quick Start

```python
import numpy as np
from geo_earthworm import GeoEarthworm

# Borehole observations
lon    = [-77.20, -77.10, -77.00, -77.25, -77.00]
lat    = [ 38.90,  38.90,  38.90,  38.80,  38.60]
depth  = [  500,    500,    500,    500,    300 ]

# Create interpolator
gew = GeoEarthworm(
    lon=lon,
    lat=lat,
    bedrock=depth,
    epsg=4326,
    is_bedrock=False   # values are depths, not bedrocks
)

# Auto-detect bounding box from data
bbox = (min(lon) - 0.1, min(lat) - 0.1, max(lon) + 0.1, max(lat) + 0.1)

# Interpolate and save GeoTIFF
result, transform = gew.interpolate_to_raster(
    bbox=bbox,
    resolution=0.01,
    output_path="bedrock_depth.tif",
    power=2.0
)
```

---

## Loading from CSV

Your CSV must have columns for longitude, latitude, and bedrock value. Column names are configurable.

```
longitude,latitude,bedrock_depth
-77.20,38.90,500
-77.10,38.90,450
-77.00,38.80,380
```

```python
from geo_earthworm import load_points_from_csv

lon, lat, depth = load_points_from_csv(
    filepath="points.csv",
    lon_col="longitude",
    lat_col="latitude",
    value_col="bedrock_depth"
)
```

---

## DEM Integration

If your input data is bedrock **bedrock** (not depth), GeoEarthworm can convert to depth using a DEM:

```python
gew = GeoEarthworm(
    lon=lon,
    lat=lat,
    bedrock=bedrock_bedrock,
    epsg=4326,
    is_bedrock=True,
    dem_path="path/to/dem.tif"   # surface elevation raster
)

result, transform = gew.interpolate_to_raster(
    bbox=bbox,
    resolution=0.01,
    output_path="bedrock_depth.tif",
    output_depth=True   # converts bedrock → depth using DEM
)
```

Alternatively, pass a DEM array directly:

```python
import rasterio

with rasterio.open("dem.tif") as src:
    dem_arr = src.read(1)
    dem_transform = src.transform

gew = GeoEarthworm(
    lon=lon, lat=lat, bedrock=bedrock_bedrock,
    epsg=4326, is_bedrock=True,
    dem_array=dem_arr,
    dem_transform=dem_transform
)
```

---

## API Reference

### `GeoEarthworm(lon, lat, bedrock, epsg, is_bedrock, dem_path, dem_array, dem_transform)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lon` | list / ndarray | required | Longitude coordinates |
| `lat` | list / ndarray | required | Latitude coordinates |
| `bedrock` | list / ndarray | required | Bedrock depth or bedrock values |
| `epsg` | int | `4326` | EPSG CRS code |
| `is_bedrock` | bool | `False` | `True` if values are bedrock, `False` if depth |
| `dem_path` | str | `None` | Path to DEM GeoTIFF (optional) |
| `dem_array` | ndarray | `None` | DEM array (alternative to `dem_path`) |
| `dem_transform` | Affine | `None` | Required when using `dem_array` |

---

### `GeoEarthworm.IDW(grid_lon, grid_lat, power, max_neighbors, max_distance, smoothing)`

Perform IDW interpolation on a 2D grid.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_lon` | ndarray | required | 2D longitude grid |
| `grid_lat` | ndarray | required | 2D latitude grid |
| `power` | float | `2.0` | IDW power — higher values increase local influence |
| `max_neighbors` | int | `None` | Limit neighbors used (None = all) |
| `max_distance` | float | `None` | Maximum neighbor distance (None = no limit) |
| `smoothing` | float | `0.0` | Smoothing factor to avoid singularities |

Returns `np.ndarray` matching the shape of `grid_lon`.

---

### `GeoEarthworm.interpolate_to_raster(bbox, resolution, output_path, ...)`

Interpolate to a regular grid and write a GeoTIFF.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | tuple | required | `(min_lon, min_lat, max_lon, max_lat)` |
| `resolution` | float | required | Grid cell size in CRS units |
| `output_path` | str | required | Output `.tif` path |
| `power` | float | `2.0` | IDW power parameter |
| `max_neighbors` | int | `None` | Max neighbors |
| `max_distance` | float | `None` | Max neighbor distance |
| `smoothing` | float | `0.0` | Smoothing factor |
| `output_depth` | bool | `True` | Output depth (`True`) or bedrock (`False`) |
| `nodata_value` | float | `-9999.0` | NoData fill value |

Returns `(np.ndarray, rasterio.Affine)`.

---

### `GeoEarthworm.get_statistics(values, nodata_value)`

Returns a dict with `min`, `max`, `mean`, `std`, `count` for input or interpolated arrays.

---

### `load_points_from_csv(filepath, lon_col, lat_col, value_col, delimiter, skiprows)`

Loads borehole data from CSV. Returns `(lon, lat, values)` as NumPy arrays with NaN rows removed.

---

## IDW Power Parameter Guide

| Power (`p`) | Effect |
|-------------|--------|
| 1 | Smooth, gradual transitions |
| **2** | **Recommended default — balances local and regional influence** |
| 3–4 | Emphasises nearest points, sharper transitions |

---

## Output Visualisation

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(result, extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
           origin='upper', cmap='plasma', aspect='auto')
plt.colorbar(label='Bedrock Depth (m)')
plt.scatter(lon, lat, c='white', s=100, edgecolors='black',
            linewidths=1.5, label='points', zorder=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Bedrock Depth — IDW Interpolation')
plt.legend()
plt.show()
```

---

## Requirements

- Python ≥ 3.8
- numpy
- scipy
- rasterio
- pandas (only required for `load_points_from_csv`)

See `requirements.txt` for pinned versions.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use GeoEarthworm in published work, please cite:

```
GeoEarthworm (2026). Bedrock depth interpolation library. https://github.com/Salten99/geoearthworm
```
