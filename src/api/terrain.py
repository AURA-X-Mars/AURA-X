import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from src.processing.dem_loader import DemLoadError, list_dem_files, load_dem
from src.processing.hirise_pipeline import HiRISEProcessingPipeline

load_dotenv()

logger = logging.getLogger("aura_x.terrain")

router = APIRouter(prefix="/api/v1/terrain", tags=["terrain"])

DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
MAX_CONCURRENT_JOBS = 2
INFO_GRID_SIZE = 64
BOULDER_RESOLUTION_LIMIT_M = 0.5

job_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


@lru_cache(maxsize=4)
def cached_grid(path_str, modified_ns, max_size):
    return load_dem(Path(path_str), max_size=max_size)


def find_dataset(dataset_id):
    for path in list_dem_files(DATA_RAW_DIR):
        if path.name == dataset_id:
            return path
    raise HTTPException(status_code=404, detail="Dataset not found")


async def get_grid(dataset_id, max_size):
    path = find_dataset(dataset_id)
    modified_ns = path.stat().st_mtime_ns
    try:
        async with job_slots:
            grid = await run_in_threadpool(cached_grid, str(path), modified_ns, max_size)
    except DemLoadError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception:
        logger.exception("Failed to load dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail="Terrain loading error")
    if not np.isfinite(grid.elevation).any():
        raise HTTPException(status_code=422, detail="Dataset has no valid elevation values")
    return grid


def describe_grid(grid, dataset_id):
    valid = np.isfinite(grid.elevation)
    return {
        "dataset_id": dataset_id,
        "crs": grid.crs,
        "bounds": [float(value) for value in grid.bounds],
        "original_width": int(grid.original_shape[1]),
        "original_height": int(grid.original_shape[0]),
        "original_cell_size_m": round(float(grid.original_cell_size_m), 3),
        "width": int(grid.elevation.shape[1]),
        "height": int(grid.elevation.shape[0]),
        "cell_size_m": round(float(grid.cell_size_m), 3),
        "min_elevation_m": round(float(np.nanmin(grid.elevation)), 2),
        "max_elevation_m": round(float(np.nanmax(grid.elevation)), 2),
        "nodata_fraction": round(float(1.0 - valid.mean()), 4),
    }


def compute_hazards(grid, max_slope_deg, max_roughness_m, boulder_threshold_m):
    pipeline = HiRISEProcessingPipeline(resolution_m_per_px=grid.cell_size_m)
    return pipeline.detect_landing_hazards(
        grid.elevation,
        boulder_threshold_meters=boulder_threshold_m,
        max_slope_deg=max_slope_deg,
        max_roughness_m=max_roughness_m,
    )


@router.get("/datasets")
async def list_datasets():
    return {
        "directory": str(DATA_RAW_DIR),
        "datasets": [
            {"id": path.name, "size_mb": round(path.stat().st_size / 1_048_576, 2)}
            for path in list_dem_files(DATA_RAW_DIR)
        ],
    }


@router.get("/{dataset_id}/info")
async def dataset_info(dataset_id: str):
    grid = await get_grid(dataset_id, INFO_GRID_SIZE)
    return describe_grid(grid, dataset_id)


@router.get("/{dataset_id}/heightmap")
async def heightmap(dataset_id: str, max_size: int = Query(default=256, ge=32, le=512)):
    grid = await get_grid(dataset_id, max_size)
    filled = np.where(np.isfinite(grid.elevation), grid.elevation, np.nanmin(grid.elevation))
    payload = describe_grid(grid, dataset_id)
    payload["elevation_m"] = np.round(filled, 2).flatten().tolist()
    return payload


@router.get("/{dataset_id}/hazards")
async def landing_hazards(
    dataset_id: str,
    max_size: int = Query(default=256, ge=32, le=512),
    max_slope_deg: float = Query(default=8.0, ge=1.0, le=30.0),
    max_roughness_m: float = Query(default=0.2, gt=0.0, le=5.0),
    boulder_threshold_m: float = Query(default=0.5, gt=0.0, le=10.0),
):
    grid = await get_grid(dataset_id, max_size)
    try:
        async with job_slots:
            result = await run_in_threadpool(
                compute_hazards, grid, max_slope_deg, max_roughness_m, boulder_threshold_m
            )
    except Exception:
        logger.exception("Hazard analysis failed for dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail="Hazard analysis error")

    slope = np.nan_to_num(np.asarray(result["slope_degrees_map"], dtype=np.float32), nan=90.0)
    safe_mask = np.asarray(result["safe_landing_mask"], dtype=bool)

    payload = describe_grid(grid, dataset_id)
    payload.update(
        {
            "criteria": {
                "max_slope_deg": max_slope_deg,
                "max_roughness_m": max_roughness_m,
                "boulder_threshold_m": boulder_threshold_m,
            },
            "boulder_detection_reliable": bool(grid.cell_size_m <= BOULDER_RESOLUTION_LIMIT_M),
            "metrics": result["hazard_metrics"],
            "slope_deg": np.round(slope, 1).flatten().tolist(),
            "safe_mask": safe_mask.astype(np.uint8).flatten().tolist(),
        }
    )
    return payload