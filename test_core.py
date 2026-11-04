"""Offline checks for the interpolation core and bad input. Run: python test_core.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_bounds

import pipeline
from geo_earthworm import GeoEarthworm


def expect_error(text, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        assert text in str(e), str(e)
    else:
        raise AssertionError(f"expected an error about: {text}")


def write_dem(path, bbox, data, nodata=-9999.0, crs="EPSG:4326"):
    rows, cols = data.shape
    with rio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1, dtype="float32", crs=crs,
                  transform=from_bounds(*bbox, cols, rows), nodata=nodata) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def read(path):
    with rio.open(path) as src:
        return src.read(1, masked=True).astype(float).filled(np.nan), src.transform


# Values are calculated where the GeoTIFF says they are.
gx, gy, t = pipeline._build_grid((0, 0, 10, 10), 1)
assert gx[0, 0] == (t * (0.5, 0.5))[0] and gx[0, -1] == (t * (9.5, 0.5))[0]
assert gy[0, 0] == (t * (0.5, 0.5))[1] and gy[-1, 0] == (t * (0.5, 9.5))[1]
written = GeoEarthworm([0, 10, 5], [0, 0, 10], [1, 2, 3], epsg=32619)
with tempfile.TemporaryDirectory() as tmp:
    arr, tr = written.interpolate_to_raster((0, 0, 10, 10), 1, str(Path(tmp) / "a.tif"))
    centre = written.IDW(np.array([[(tr * (0.5, 0.5))[0]]]), np.array([[(tr * (0.5, 0.5))[1]]]))
    assert arr[0, 0] == centre[0, 0]

# Longitude degrees count for less than latitude degrees: east 0.1 deg at 60N is about half as far as north 0.1 deg.
east_near = GeoEarthworm([0.1, 0.0, -9], [60.0, 60.1, 60.0], [0, 100, 50])
value = east_near.IDW(np.array([[0.0]]), np.array([[60.0]]), max_neighbors=2)[0, 0]
assert 15 < value < 25, value
projected = GeoEarthworm([100, 0, -9000], [0, 100, 0], [0, 100, 50], epsg=32619)
assert abs(projected.IDW(np.array([[0.0]]), np.array([[0.0]]), max_neighbors=2)[0, 0] - 50) < 1e-9

# A cell on a borehole takes its value; two boreholes on one spot average.
pts = GeoEarthworm([0, 1, 0, 0], [0, 0, 1, 1], [10, 20, 30, 50], epsg=32619)
assert pts.IDW(np.array([[0.0]]), np.array([[0.0]]))[0, 0] == 10
assert pts.IDW(np.array([[0.0]]), np.array([[1.0]]))[0, 0] == 40

with tempfile.TemporaryDirectory() as folder:
    tmp = Path(folder)
    rows = [(-71.5 + .1 * i, 41.5 + .07 * ((i * 3) % 5), 10 + 3.0 * i) for i in range(12)]
    good = tmp / "good.csv"
    pd.DataFrame(rows, columns=("longitude", "latitude", "bedrock_depth")).to_csv(good, index=False)
    bbox = (-72, 41, -70, 43)

    # Text in a number column is skipped and reported, not a crash.
    text = tmp / "text.csv"
    pd.DataFrame([(*rows[0][:2], "unknown"), *rows[1:]], columns=("longitude", "latitude", "bedrock_depth")).to_csv(text, index=False)
    r = pipeline.run(text, output_dir=tmp / "text")
    assert r.input_stats["count"] == 11 and any("Skipped 1 row" in w for w in r.warnings), r.warnings

    # Unusable DEMs get a clear error, which is what the app and CLI catch.
    garbage = tmp / "garbage.tif"
    garbage.write_bytes(b"not a tiff")
    expect_error("couldn't be read", pipeline.run, good, dem_path=str(garbage), output_dir=tmp / "g")
    far = write_dem(tmp / "far.tif", (10, 10, 11, 11), np.full((10, 10), 100.0))
    expect_error("doesn't overlap", pipeline.run, good, dem_path=str(far), output_dir=tmp / "f")
    nocrs = write_dem(tmp / "nocrs.tif", bbox, np.full((10, 10), 100.0), crs=None)
    expect_error("no coordinate system", pipeline.run, good, dem_path=str(nocrs), output_dir=tmp / "n")

    # Part-covering DEM: results for the covered part, and a warning.
    part = write_dem(tmp / "part.tif", (-72, 41, -71.1, 43), np.full((40, 40), 100.0))
    r = pipeline.run(good, dem_path=str(part), output_dir=tmp / "p")
    assert any("covers only" in w for w in r.warnings), r.warnings

    # Gaps the file doesn't declare are missing data, not -9999 m ground.
    for gap in pipeline.UNDECLARED_DEM_GAPS:
        holes = np.full((50, 50), 100.0)
        holes[20:30, 20:30] = gap
        untagged = write_dem(tmp / f"untagged{gap}.tif", bbox, holes, nodata=None)
        r = pipeline.run(good, dem_path=str(untagged), output_dir=tmp / f"u{gap}")
        altitude, _ = read(r.bedrock_altitude_path)
        assert np.nanmin(altitude) > 0 and np.isnan(altitude).any(), np.nanmin(altitude)
        assert any("as missing" in w for w in r.warnings), r.warnings

    # A tagged DEM's gaps don't bleed into nearby cells either, and clean DEMs aren't copied.
    holes = np.full((50, 50), 100.0)
    holes[20:30, 20:30] = -9999
    r = pipeline.run(good, dem_path=str(write_dem(tmp / "tagged.tif", bbox, holes)), output_dir=tmp / "t")
    assert np.nanmin(read(r.bedrock_altitude_path)[0]) > 0
    assert not (tmp / "t" / "dem_gaps_marked.tif").exists()

    # Server limits refuse oversized work before allocating anything; off by default for local use.
    expect_error("allowed here", pipeline.run, good, resolution=1e-5, output_dir=tmp / "s", server_limits=True)
    expect_error("must be a positive", pipeline.run, good, resolution=-1, output_dir=tmp / "neg")
    many = tmp / "many.csv"
    rng = np.random.default_rng(1)
    pd.DataFrame({"longitude": rng.uniform(-72, -71, 2001), "latitude": rng.uniform(41, 42, 2001),
                  "bedrock_depth": rng.uniform(1, 50, 2001)}).to_csv(many, index=False)
    expect_error("limited to", pipeline.run, many, method="kriging", output_dir=tmp / "k", server_limits=True)

    # Many boreholes use the nearest few, with a note; asking for a number is respected.
    r = pipeline.run(many, output_dir=tmp / "auto", dpi=50)
    assert any("nearest 32" in w for w in r.warnings), r.warnings
    r = pipeline.run(many, max_neighbors=5, output_dir=tmp / "five", dpi=50)
    assert not any("nearest 32" in w for w in r.warnings), r.warnings

    # All three methods still run and stay within sensible bounds on ordinary degree data.
    for method in ("idw", "kriging", "spline"):
        r = pipeline.run(good, method=method, output_dir=tmp / method, dpi=50)
        assert 0 < r.bedrock_depth_stats["mean"] < 60, (method, r.bedrock_depth_stats)

# A zero margin keeps the map tight; the 1-unit pad is only for boreholes with no width.
assert pipeline._auto_bbox(np.array([0.0, 2.0]), np.array([0.0, 1.0]), 0.0) == (0, 0, 2, 1)
assert pipeline._auto_bbox(np.array([0.0, 2.0]), np.array([5.0, 5.0]), 0.1) == (-0.2, 4, 2.2, 6)

with tempfile.TemporaryDirectory() as folder:
    tmp = Path(folder)
    rows = [(-71.5 + .1 * i, 41.5 + .07 * ((i * 3) % 5), 10 + 3.0 * i) for i in range(12)]

    # A location measured twice is averaged, so spline and kriging can still solve.
    dup = tmp / "dup.csv"
    pd.DataFrame([*rows, (rows[0][0], rows[0][1], rows[0][2] + 2)],
                 columns=("longitude", "latitude", "bedrock_depth")).to_csv(dup, index=False)
    for method in ("idw", "kriging", "spline"):
        r = pipeline.run(dup, method=method, output_dir=tmp / f"dup_{method}", dpi=50)
        assert r.input_stats["count"] == 12 and any("Averaged 1" in w for w in r.warnings), r.warnings

    # Kriging in row bands gives the same surface as all at once, without the memory spike.
    lon, lat, val = (np.array(c) for c in zip(*rows))
    gx, gy, _ = pipeline._build_grid((-72, 41, -70, 43), 0.05)
    whole = pipeline._interpolate_kriging(lon, lat, val, gx, gy, "spherical")
    real, pipeline.KRIGING_CELLS_X_POINTS = pipeline.KRIGING_CELLS_X_POINTS, 50 * 12  # 1 row per band
    banded = pipeline._interpolate_kriging(lon, lat, val, gx, gy, "spherical")
    pipeline.KRIGING_CELLS_X_POINTS = real
    assert np.allclose(whole, banded), np.nanmax(np.abs(whole - banded))

    import tracemalloc
    rng = np.random.default_rng(2)
    big = tmp / "big.csv"
    pd.DataFrame({"longitude": rng.uniform(-72, -71, 500), "latitude": rng.uniform(41, 42, 500),
                  "bedrock_depth": rng.uniform(1, 50, 500)}).to_csv(big, index=False)
    tracemalloc.start()
    pipeline.run(big, method="kriging", output_dir=tmp / "big", dpi=50)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 0.5e9, f"kriging peaked at {peak / 1e9:.2f} GB"  # was ~1.95 GB before banding

print("all core checks passed")
