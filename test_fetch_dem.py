"""Checks for the automatic USGS DEM path. Needs internet except for the failure case. Run: python test_fetch_dem.py"""
import tempfile
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.warp import transform

import pipeline


def boreholes(folder, lons, lats, epsg=4326):
    """Grid of boreholes at a constant 10 m depth, written in the given CRS."""
    lon, lat = np.meshgrid(lons, lats)
    x, y = transform("EPSG:4326", f"EPSG:{epsg}", lon.ravel().tolist(), lat.ravel().tolist())
    path = Path(folder) / "boreholes.csv"
    pd.DataFrame({"longitude": x, "latitude": y, "bedrock_depth": 10.0}).to_csv(path, index=False)
    return path


def grid(path):
    with rio.open(path) as src:
        return src.read(1, masked=True).astype(float).filled(np.nan)


RI_LON, RI_LAT = np.linspace(-71.7, -71.3, 4), np.linspace(41.6, 41.9, 4)

with tempfile.TemporaryDirectory() as tmp:
    # US site, degrees: real elevation comes back and altitude = surface - depth.
    r = pipeline.run(boreholes(tmp, RI_LON, RI_LAT), output_dir=Path(tmp) / "us", fetch_dem=True)
    assert r.dem_source == "USGS 3DEP" and r.overburden_path, r.warnings
    altitude = grid(r.bedrock_altitude_path)
    assert np.isfinite(altitude).all() and -100 < np.nanmin(altitude) < np.nanmax(altitude) < 300
    assert not (altitude < -1000).any(), "nodata leaked into the surface"

    # Same site in UTM metres: the 4326 download must land on the projected grid, at matching heights.
    u = pipeline.run(boreholes(tmp, RI_LON, RI_LAT, 32619), epsg=32619, output_dir=Path(tmp) / "utm", fetch_dem=True)
    utm = grid(u.bedrock_altitude_path)
    assert u.dem_source == "USGS 3DEP" and np.isfinite(utm).all()
    assert abs(np.nanmean(utm) - np.nanmean(altitude)) < 5, (np.nanmean(utm), np.nanmean(altitude))

    # Outside coverage: the run still succeeds, with a clear warning and no surface products.
    ch = pipeline.run(boreholes(tmp, np.linspace(7.4, 7.6, 4), np.linspace(46.8, 47.0, 4)),
                      output_dir=Path(tmp) / "ch", fetch_dem=True)
    assert ch.dem_source is None and ch.overburden_path is None and ch.bedrock_depth_path
    assert any("no elevation data" in w for w in ch.warnings), ch.warnings

    # Service down: same outcome, and the failure is explained.
    def down(*a, **k):
        raise URLError("service unavailable")
    real, pipeline.urlopen = pipeline.urlopen, down
    d = pipeline.run(boreholes(tmp, RI_LON, RI_LAT), output_dir=Path(tmp) / "down", fetch_dem=True)
    pipeline.urlopen = real
    assert d.dem_source is None and d.bedrock_depth_path
    assert any("Couldn't get elevation data" in w for w in d.warnings), d.warnings

    # Service answers with its JSON error body instead of a TIFF.
    class Body:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"error":{"code":400}}'
    pipeline.urlopen = lambda *a, **k: Body()
    j = pipeline.run(boreholes(tmp, RI_LON, RI_LAT), output_dir=Path(tmp) / "json", fetch_dem=True)
    pipeline.urlopen = real
    assert j.dem_source is None and any("Couldn't get elevation data" in w for w in j.warnings), j.warnings

    # Off by default, so the CLI and existing callers never touch the network.
    pipeline.urlopen = down
    o = pipeline.run(boreholes(tmp, RI_LON, RI_LAT), output_dir=Path(tmp) / "off")
    pipeline.urlopen = real
    assert o.dem_source is None and any("No DEM supplied" in w for w in o.warnings)

print("all DEM checks passed")
