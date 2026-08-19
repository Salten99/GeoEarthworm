"""
GeoEarthworm - Bedrock Depth Interpolation Module

This module provides quick and easy bedrock interpolations from scattered
point measurements using Inverse Distance Weighting (IDW).
It supports coordinate transformations, DEM integration, and GeoTIFF output.

"""

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
import warnings
from typing import Tuple, Optional, Union, List
from pyproj import CRS, Transformer


class GeoEarthworm:

    """

    This class is for interpolation of bedrock depth or bedrock values using IDW.

    Attributes:
    lon (np.ndarray): Longitude coordinates
    lat (np.ndarray): Latitude coordinates
    bedrock (np.ndarray): Bedrock values, either depth or bedrock
    epsg (int): EPSG coordinate reference system code
    is_altitude (bool): Whether bedrock values are bedrock (true) or depth (false)
    dem_path (str): Path to DEM file (optional)
    dem_array (np.ndarray): DEM array (optional)
    dem_transform (rasterio.Affine): DEM transformation (optional)

    """

    def __init__(
        self,
        lon: Union[List[float], np.ndarray],
        lat: Union[List[float], np.ndarray],
        bedrock: Union[List[float], np.ndarray],
        epsg: int = 4326,
        is_altitude: bool = False,
        dem_path: Optional[str] = None,
        dem_array: Optional[np.ndarray] = None,
        dem_transform: Optional[rio.Affine] = None
    ):
        # Initialize
        self.lon = np.asarray(lon, dtype=np.float64)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.bedrock = np.asarray(bedrock, dtype=np.float64)
        self.epsg = epsg
        self.is_altitude = is_altitude

        # Validate input dimensions
        if not (len(self.lon) == len(self.lat) == len(self.bedrock)):
            raise ValueError("lon, lat and bedrock must have the same amount of values.")

        if len(self.lon) < 3:
            raise ValueError("Must have at least 3 points for interpolation.")

        # DEM
        self.dem_array = None
        self.dem_transform = None
        self.dem_crs = None

        if dem_path is not None:
            with rio.open(dem_path) as dem:
                self.dem_array = dem.read(1)
                self.dem_transform = dem.transform
                self.dem_crs = dem.crs

        elif dem_array is not None:
            if dem_transform is None:
                raise ValueError("dem_transform is required with dem_array")
            self.dem_array = dem_array
            self.dem_transform = dem_transform
            self.dem_crs = CRS.from_epsg(epsg)

        # Interpolation requires a KDTree of values in a 2D array
        self.points = np.column_stack([self.lon, self.lat])
        self.tree = cKDTree(self.points)

    def IDW(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        power: float = 2.0,
        max_neighbors: Optional[int] = None,
        max_distance: Optional[float] = None,
        smoothing: float = 0.0,
        chunk_size: int = 10000
    ) -> np.ndarray:
        """
        Perform IDW interpolation on a grid.

        Parameters:
            grid_lon: 2D array of longitude coordinates for output grid
            grid_lat: 2D array of latitude coordinates for output grid
            power: Power parameter for IDW (default: 2.0). Higher values increase local influence.
            max_neighbors: Maximum number of neighbors to use (None = use all)
            max_distance: Maximum distance for neighbors in same units as coordinates (None = no limit)
            smoothing: Smoothing factor to avoid singularities at known points (default: 0.0)
            chunk_size: Number of grid points to process at once to limit RAM usage (default: 10000)

        Returns:
            np.ndarray: Interpolated values reshaped to match grid_lon/grid_lat shape
        """

        query_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        interpolated = np.full(len(query_points), np.nan)

        k = len(self.points) if max_neighbors is None else min(max_neighbors, len(self.points))

        for start in range(0, len(query_points), chunk_size):
            end = min(start + chunk_size, len(query_points))
            chunk = query_points[start:end]

            if max_distance is not None:
                raw_indices = self.tree.query_ball_point(chunk, max_distance)
                max_k = max((len(idx) for idx in raw_indices), default=1)

                distances = np.full((len(chunk), max_k), np.inf)
                indices = np.full((len(chunk), max_k), len(self.points))

                for i, idx_list in enumerate(raw_indices):
                    if len(idx_list) > 0:
                        dists = np.linalg.norm(chunk[i] - self.points[idx_list], axis=1)
                        distances[i, :len(idx_list)] = dists
                        indices[i, :len(idx_list)] = idx_list
            else:
                distances, indices = self.tree.query(chunk, k=k)

            if distances.ndim == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            valid_mask = (distances < np.inf) & (indices < len(self.points))

            if smoothing > 0.0:
                weights = 1.0 / (distances + smoothing) ** power
            else:
                safe_distances = np.where(distances == 0, np.nan, distances)
                weights = np.where(distances == 0, np.inf, 1.0 / (safe_distances ** power))

            weights = np.where(valid_mask, weights, 0.0)
            weight_sum = weights.sum(axis=1, keepdims=True)
            valid_points = weight_sum.ravel() > 0

            safe_indices = np.where(valid_mask, indices, 0).astype(int)
            weights_norm = np.where(weight_sum > 0, weights / weight_sum, 0.0)

            chunk_result = np.full(len(chunk), np.nan)
            chunk_result[valid_points] = (
                weights_norm[valid_points] * self.bedrock[safe_indices[valid_points]]
            ).sum(axis=1)

            interpolated[start:end] = chunk_result

        return interpolated.reshape(grid_lon.shape)

    def _convert_to_depth(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        interpolated: np.ndarray
    ) -> np.ndarray:
        """
        Convert interpolated bedrock altitude to depth using DEM.
        depth = surface elevation - bedrock altitude
        """

        if self.dem_array is None:
            raise ValueError(
                "DEM is required to convert bedrock to depth. "
                "Provide dem_path or dem_array when initialising GeoEarthworm."
            )

        rows_dem, cols_dem = rio.transform.rowcol(
            self.dem_transform,
            grid_lon.ravel(),
            grid_lat.ravel()
        )
        rows_dem = np.clip(rows_dem, 0, self.dem_array.shape[0] - 1)
        cols_dem = np.clip(cols_dem, 0, self.dem_array.shape[1] - 1)

        dem_values = self.dem_array[rows_dem, cols_dem].reshape(grid_lon.shape)
        depth = dem_values - interpolated

        return depth

    def interpolate_to_raster(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float,
        output_path: str,
        power: float = 2.0,
        max_neighbors: Optional[int] = None,
        max_distance: Optional[float] = None,
        smoothing: float = 0.0,
        output_depth: bool = True,
        nodata_value: float = -9999.0,
        chunk_size: int = 10000
    ) -> Tuple[np.ndarray, rio.Affine]:
        """
        Interpolate bedrock depth to a raster and save as GeoTIFF.

        Parameters:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid resolution in units of the coordinate system
            output_path: Path for output GeoTIFF file
            power: Power parameter for IDW (default: 2.0)
            max_neighbors: Maximum number of neighbors to use (None = use all)
            max_distance: Maximum distance for neighbors (None = no limit)
            smoothing: Smoothing parameter (default: 0.0)
            output_depth: If True, output bedrock depth; if False, output bedrock altitude
            nodata_value: Value to use for NoData pixels
            chunk_size: Number of grid points per chunk for memory efficiency (default: 10000)

        Returns:
            Tuple of (interpolated array, affine transform)
        """

        min_lon, min_lat, max_lon, max_lat = bbox

        # Create grid
        cols = int(np.ceil((max_lon - min_lon) / resolution))
        rows = int(np.ceil((max_lat - min_lat) / resolution))

        x = np.linspace(min_lon, max_lon, cols)
        y = np.linspace(max_lat, min_lat, rows)  # Reversed for raster (top to bottom)

        grid_lon, grid_lat = np.meshgrid(x, y)

        # Perform interpolation
        interpolated = self.IDW(
            grid_lon, grid_lat,
            power=power,
            max_neighbors=max_neighbors,
            max_distance=max_distance,
            smoothing=smoothing,
            chunk_size=chunk_size
        )

        # Convert to depth if needed
        if output_depth:
            if self.is_altitude:
                interpolated = self._convert_to_depth(grid_lon, grid_lat, interpolated)
        else:
            if not self.is_altitude:
                if self.dem_array is None:
                    raise ValueError("DEM is required to convert depth to bedrock altitude")
                rows_dem, cols_dem = rio.transform.rowcol(
                    self.dem_transform,
                    grid_lon.ravel(),
                    grid_lat.ravel()
                )
                rows_dem = np.clip(rows_dem, 0, self.dem_array.shape[0] - 1)
                cols_dem = np.clip(cols_dem, 0, self.dem_array.shape[1] - 1)
                dem_values = self.dem_array[rows_dem, cols_dem].reshape(grid_lon.shape)
                interpolated = dem_values - interpolated

        # Replace NaN with nodata value
        interpolated = np.where(np.isnan(interpolated), nodata_value, interpolated)

        # Create affine transform
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, cols, rows)

        # Write to GeoTIFF
        with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=interpolated.dtype,
            crs=CRS.from_epsg(self.epsg),
            transform=transform,
            nodata=nodata_value,
            compress='lzw'
        ) as dst:
            dst.write(interpolated, 1)

        print(f"Raster saved to: {output_path}")
        print(f"Dimensions: {rows} rows x {cols} cols")
        print(f"Resolution: {resolution}")
        print(f"EPSG: {self.epsg}")

        return interpolated, transform

    @staticmethod
    def reproj_match(infile, match, outfile):
        """
        Reproject a raster to match the exact grid of another raster.

        Parameters
        ----------
        infile : str
            Path to input raster to be reprojected
        match : str
            Path to raster with desired shape and projection
        outfile : str
            Path for output reprojected raster
        """
        with rio.open(infile) as src:
            with rio.open(match) as match_raster:
                dst_kwargs = src.profile.copy()
                dst_kwargs.update(
                    {
                        "crs": match_raster.crs,
                        "transform": match_raster.transform,
                        "width": match_raster.width,
                        "height": match_raster.height,
                        "nodata": src.nodata,
                    }
                )

                with rio.open(outfile, "w", **dst_kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rio.band(src, i),
                            destination=rio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=match_raster.transform,
                            dst_crs=match_raster.crs,
                            resampling=Resampling.nearest,
                        )

    @staticmethod
    def calculate_overburden(
        interpolated_path: str,
        dem_path: str,
        output_path: str,
        nodata_value: float = -9999.0
    ) -> Tuple[np.ndarray, rio.Affine]:
        """
        Calculate bedrock overburden (surface elevation - bedrock altitude).
        Reprojects and resamples both inputs to a common grid before subtraction.

        Parameters:
            interpolated_path: Path to interpolated bedrock altitude GeoTIFF
            dem_path: Path to DEM GeoTIFF
            output_path: Path for output overburden GeoTIFF
            nodata_value: Value to use for NoData pixels

        Returns:
            Tuple of (overburden array, affine transform)
        """

        # Load bedrock raster as reference grid 
        with rio.open(interpolated_path) as bedrock_src:
            bedrock_crs = bedrock_src.crs
            bedrock_transform = bedrock_src.transform
            bedrock_shape = (bedrock_src.height, bedrock_src.width)
            bedrock_array = bedrock_src.read(1).astype(np.float64)
            bedrock_nodata = bedrock_src.nodata

        # Mask bedrock nodata
        if bedrock_nodata is not None:
            bedrock_array = np.where(bedrock_array == bedrock_nodata, np.nan, bedrock_array)
        bedrock_array = np.where(bedrock_array == nodata_value, np.nan, bedrock_array)

        # Reproject DEM to match bedrock raster exactly
        with rio.open(dem_path) as dem_src:
            dem_reproject = np.full(bedrock_shape, np.nan, dtype=np.float64)

            reproject(
                source=rio.band(dem_src, 1),
                destination=dem_reproject,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=bedrock_transform,
                dst_crs=bedrock_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan
            )

            dem_nodata = dem_src.nodata

        # Mask DEM nodata
        if dem_nodata is not None:
            dem_reproject = np.where(dem_reproject == dem_nodata, np.nan, dem_reproject)

        # Confirm shapes match before math
        if dem_reproject.shape != bedrock_array.shape:
            raise ValueError(
                f"Shape mismatch after reprojection: "
                f"DEM {dem_reproject.shape} vs bedrock {bedrock_array.shape}. "
                f"Check that both rasters overlap spatially."
            )

        print(f"DEM shape:     {dem_reproject.shape}")
        print(f"Bedrock shape: {bedrock_array.shape}")
        print(f"CRS:           {bedrock_crs}")

        # Calculate overburden
        overburden = dem_reproject - bedrock_array

        # Clamp negative overburden to 0
        # Negative values mean IDW extrapolated bedrock above the surface — physically impossible
        n_negative = int(np.sum(overburden < 0))
        if n_negative > 0:
            warnings.warn(
                f"{n_negative} pixels had bedrock altitude above surface (IDW extrapolation artifact). "
                f"Clamping to 0."
            )
        overburden = np.where(overburden < 0, 0.0, overburden)

        # Replace NaN with nodata sentinel for GeoTIFF output
        overburden_out = np.where(np.isnan(overburden), nodata_value, overburden)

        # Write output GeoTIFF
        with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=bedrock_shape[0],
            width=bedrock_shape[1],
            count=1,
            dtype=overburden_out.dtype,
            crs=bedrock_crs,
            transform=bedrock_transform,
            nodata=nodata_value,
            compress='lzw'
        ) as dst:
            dst.write(overburden_out, 1)

        stats = {
            'min':  float(np.nanmin(overburden)),
            'max':  float(np.nanmax(overburden)),
            'mean': float(np.nanmean(overburden)),
            'std':  float(np.nanstd(overburden)),
        }

        print(f"\nOverburden statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}m")
        print(f"\nOverburden raster saved to: {output_path}")

        return overburden, bedrock_transform

    def get_statistics(
        self,
        values: Optional[np.ndarray] = None,
        nodata_value: float = -9999.0
    ) -> dict:
        """
        Get statistics of input or interpolated values.

        Parameters:
            values: Array to compute statistics for (default: use input bedrock values)
            nodata_value: Value to exclude as NoData (only applies to non-input arrays)

        Returns:
            Dictionary with min, max, mean, std, and count
        """
        if values is None:
            valid_values = self.bedrock[~np.isnan(self.bedrock)].ravel()
        else:
            flat = values.ravel().astype(np.float64)
            valid_values = flat[~np.isnan(flat) & (flat != nodata_value)]

        if len(valid_values) == 0:
            return {'min': np.nan, 'max': np.nan, 'mean': np.nan,
                    'std': np.nan, 'count': 0}

        return {
            'min':   float(np.min(valid_values)),
            'max':   float(np.max(valid_values)),
            'mean':  float(np.mean(valid_values)),
            'std':   float(np.std(valid_values)),
            'count': len(valid_values)
        }

    @staticmethod
    def load_points_from_csv(
        filepath: str,
        lon_col: str = 'longitude',
        lat_col: str = 'latitude',
        bedrock_col: str = 'bedrock',
        epsg: int = 4326,
        target_epsg: int = None,
        delimiter: str = ',',
        skiprows: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Load point data from CSV file and optionally reproject coordinates.

        Parameters:
            filepath: Path to CSV file
            lon_col: Name of longitude column
            lat_col: Name of latitude column
            bedrock_col: Name of value column
            epsg: Input CRS EPSG code
            target_epsg: Output CRS EPSG code (optional)
            delimiter: CSV delimiter
            skiprows: Number of rows to skip at start

        Returns:
            Tuple of (x, y, values, epsg) arrays
        """

        import pandas as pd

        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skiprows)

        lon = df[lon_col].values
        lat = df[lat_col].values
        bedrock = df[bedrock_col].values

        # Remove missing values
        mask = ~(np.isnan(lon) | np.isnan(lat) | np.isnan(bedrock))
        lon, lat, bedrock = lon[mask], lat[mask], bedrock[mask]

        source_crs = CRS.from_epsg(epsg)
        output_epsg = epsg

        if target_epsg is not None:
            target_crs = CRS.from_epsg(target_epsg)
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            lon, lat = transformer.transform(lon, lat)
            output_epsg = target_epsg

        return lon, lat, bedrock, output_epsg


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    n_points = 50
    lon = np.random.uniform(-120, -119, n_points)
    lat = np.random.uniform(37, 38, n_points)
    bedrock_altitude = np.random.uniform(100, 500, n_points)

    interpolator = GeoEarthworm(
        lon=lon,
        lat=lat,
        bedrock=bedrock_altitude,
        epsg=4326,
        is_altitude=True
    )

    stats = interpolator.get_statistics()
    print("Input data statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    bbox = (-120, 37, -119, 38)

    print("\nInterpolating to raster...")
    result, transform = interpolator.interpolate_to_raster(
        bbox=bbox,
        resolution=0.01,
        output_path="bedrock_altitude.tif",
        power=2.0,
        max_neighbors=10,
        output_depth=False
    )

    print("\nGenerating synthetic DEM for overburden demo...")
    dem_cols, dem_rows = 100, 100
    dem_transform = from_bounds(*bbox, dem_cols, dem_rows)
    dem_x = np.linspace(bbox[0], bbox[2], dem_cols)
    dem_y = np.linspace(bbox[3], bbox[1], dem_rows)
    dem_lon, dem_lat = np.meshgrid(dem_x, dem_y)
    # Surface kept well above bedrock_altitude's 100-500 range so overburden stays positive
    dem_array = 650 + 50 * np.sin(dem_lon * 5) + 30 * np.cos(dem_lat * 5)

    with rio.open(
        "dem.tif", 'w',
        driver='GTiff',
        height=dem_rows,
        width=dem_cols,
        count=1,
        dtype=dem_array.dtype,
        crs=CRS.from_epsg(4326),
        transform=dem_transform,
        nodata=-9999.0,
        compress='lzw'
    ) as dst:
        dst.write(dem_array, 1)

    print("\nCalculating overburden...")
    overburden, transform = GeoEarthworm.calculate_overburden(
        interpolated_path="bedrock_altitude.tif",
        dem_path="dem.tif",
        output_path="overburden.tif"
    )

    print("\nDone!")