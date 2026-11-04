#!/usr/bin/env python
"""
GeoEarthworm CLI.

Interpolate bedrock level from a borehole CSV, optionally combine it with a
surface DEM to compute overburden thickness, and save GeoTIFFs plus a
summary PNG.

Example:
    python run_geoearthworm.py --csv boreholes.csv --dem dem.tif

Without a DEM you still get a bedrock-level GeoTIFF and image; overburden
requires a DEM.
"""

import argparse
import sys

from pipeline import run


def _print_stats(title: str, stats: dict) -> None:
    if not stats:
        return
    print(f"\n{title}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="CSV file of borehole points")
    parser.add_argument("--lon-col", default="longitude", help="Longitude column name (default: longitude)")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column name (default: latitude)")
    parser.add_argument("--value-col", default="bedrock_depth", help="Bedrock value column name (default: bedrock_depth)")
    parser.add_argument("--epsg", type=int, default=4326, help="EPSG code of the input coordinates (default: 4326)")
    parser.add_argument(
        "--values", choices=["depth", "altitude"], default="depth",
        help="Whether --value-col holds bedrock depth below the surface, or bedrock "
             "altitude/elevation (default: depth)",
    )
    parser.add_argument("--dem", default=None, help="Surface elevation GeoTIFF. Required for overburden.")
    parser.add_argument(
        "--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"),
        default=None, help="Output extent. Default: auto-detected from the data plus --buffer",
    )
    parser.add_argument("--buffer", type=float, default=0.1, help="Fractional buffer around the auto bbox (default: 0.1)")
    parser.add_argument("--resolution", type=float, default=None, help="Grid cell size in CRS units. Default: auto (~300 cells across the longest side)")
    parser.add_argument(
        "--method", choices=["idw", "kriging", "spline"], default="idw",
        help="Interpolation method (default: idw)",
    )
    parser.add_argument("--power", type=float, default=2.0, help="IDW power parameter (default: 2.0)")
    parser.add_argument("--max-neighbors", type=int, default=None, help="Limit neighbors used per cell (IDW and spline; default: all, or the nearest 32 above 1,000 points)")
    parser.add_argument("--max-distance", type=float, default=None, help="Max neighbor distance, in CRS units (IDW only; default: none)")
    parser.add_argument("--smoothing", type=float, default=0.0, help="Smoothing factor for IDW or spline (default: 0.0)")
    parser.add_argument(
        "--variogram-model", choices=["linear", "power", "gaussian", "spherical", "exponential"],
        default="spherical", help="Variogram model for kriging (default: spherical)",
    )
    parser.add_argument("--nodata", type=float, default=-9999.0, help="NoData sentinel value (default: -9999.0)")
    parser.add_argument("--output-dir", default="geoearthworm_output", help="Directory for outputs (default: geoearthworm_output)")
    parser.add_argument("--dpi", type=int, default=200, help="Output image resolution (default: 200)")
    parser.add_argument(
        "--hide-points", action="store_true",
        help="Don't plot borehole locations on the output map (default: shown)",
    )
    args = parser.parse_args(argv)

    try:
        result = run(
            csv_path=args.csv,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            value_col=args.value_col,
            epsg=args.epsg,
            is_altitude=(args.values == "altitude"),
            dem_path=args.dem,
            bbox=tuple(args.bbox) if args.bbox else None,
            buffer_frac=args.buffer,
            resolution=args.resolution,
            method=args.method,
            power=args.power,
            max_neighbors=args.max_neighbors,
            max_distance=args.max_distance,
            smoothing=args.smoothing,
            variogram_model=args.variogram_model,
            nodata=args.nodata,
            output_dir=args.output_dir,
            dpi=args.dpi,
            show_points=not args.hide_points,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Interpolation method: {result.method}")
    print(f"Outputs written to: {result.output_dir.resolve()}")
    if result.bedrock_depth_path:
        print(f"  Bedrock depth raster:    {result.bedrock_depth_path}")
    if result.bedrock_altitude_path:
        print(f"  Bedrock altitude raster: {result.bedrock_altitude_path}")
    if result.overburden_path:
        print(f"  Overburden raster:       {result.overburden_path}")
    print(f"  Summary image:           {result.image_path}")

    _print_stats("Input point statistics", result.input_stats)
    _print_stats("Bedrock depth statistics", result.bedrock_depth_stats)
    _print_stats("Bedrock altitude statistics", result.bedrock_altitude_stats)
    _print_stats("Overburden statistics", result.overburden_stats)

    for w in result.warnings:
        print(f"\nNote: {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
