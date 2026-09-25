import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS

logger = logging.getLogger("aura_x.dem_loader")

DEM_EXTENSIONS = (".tif", ".tiff", ".img")
GEOGRAPHIC_TARGET_CRS = "ESRI:104905"
MARS_EQUATORIAL_RADIUS_M = 3396190.0


class DemLoadError(Exception):
    pass

@dataclass
class DemGrid:
    elevation: np.ndarray
    cell_size_m: float
    crs: str
    bounds: tuple
    original_shape: tuple
    original_cell_size_m: float


def list_dem_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DEM_EXTENSIONS
    )


def resolve_source_cell_size_m(dataset):
    pixel_width = abs(dataset.transform.a)
    pixel_height = abs(dataset.transform.e)
    average_pixel = (pixel_width + pixel_height) / 2.0

    if dataset.crs and dataset.crs.is_geographic:
        return average_pixel * (np.pi / 180.0) * MARS_EQUATORIAL_RADIUS_M
    return average_pixel


def compute_target_shape(width, height, max_size):
    if max_size is None or (width <= max_size and height <= max_size):
        return width, height
    scale = max_size / max(width, height)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return target_width, target_height


def load_dem(path, max_size=None):
    path = Path(path)
    if not path.is_file():
        raise DemLoadError(f"file not found: {path}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.count < 1:
                raise DemLoadError("dataset has no raster bands")
            if dataset.crs is None:
                raise DemLoadError("dataset has no CRS")

            original_width, original_height = dataset.width, dataset.height
            original_cell_size_m = resolve_source_cell_size_m(dataset)

            needs_reprojection = dataset.crs.is_geographic
            if needs_reprojection:
                target_crs = CRS.from_user_input(GEOGRAPHIC_TARGET_CRS)
                transform, width, height = calculate_default_transform(
                    dataset.crs,
                    target_crs,
                    dataset.width,
                    dataset.height,
                    *dataset.bounds,
                )
            else:
                target_crs = dataset.crs
                transform = dataset.transform
                width, height = dataset.width, dataset.height

            out_width, out_height = compute_target_shape(width, height, max_size)
            scale_x = width / out_width
            scale_y = height / out_height
            output_transform = transform * transform.scale(scale_x, scale_y)

            destination = np.full((out_height, out_width), np.nan, dtype=np.float32)

            source_nodata = dataset.nodata
            if needs_reprojection:
                reproject(
                    source=rasterio.band(dataset, 1),
                    destination=destination,
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    src_nodata=source_nodata,
                    dst_transform=output_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            else:
                band = dataset.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                if source_nodata is not None:
                    band[band == source_nodata] = np.nan
                destination = band

            cell_size_m = abs(output_transform.a)
            bounds_left = output_transform.c
            bounds_top = output_transform.f
            bounds_right = bounds_left + output_transform.a * out_width
            bounds_bottom = bounds_top + output_transform.e * out_height
            bounds = (
                min(bounds_left, bounds_right),
                min(bounds_top, bounds_bottom),
                max(bounds_left, bounds_right),
                max(bounds_top, bounds_bottom),
            )

    except rasterio.errors.RasterioIOError as error:
        raise DemLoadError(f"could not read raster: {error}") from error

    if not np.isfinite(destination).any():
        raise DemLoadError("no valid elevation data after reprojection")

    return DemGrid(
        elevation=destination,
        cell_size_m=float(cell_size_m),
        crs=str(target_crs),
        bounds=bounds,
        original_shape=(original_height, original_width),
        original_cell_size_m=float(original_cell_size_m),
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python -m src.processing.dem_loader <path-to-dem>")
        sys.exit(1)

    grid = load_dem(sys.argv[1], max_size=256)
    print(f"CRS: {grid.crs}")
    print(f"Original shape: {grid.original_shape}, cell size: {grid.original_cell_size_m:.3f} m")
    print(f"Loaded shape: {grid.elevation.shape}, cell size: {grid.cell_size_m:.3f} m")
    print(f"Elevation range: {np.nanmin(grid.elevation):.2f} to {np.nanmax(grid.elevation):.2f} m")
    print(f"Bounds: {grid.bounds}")