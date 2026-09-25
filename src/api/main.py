import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.api.terrain import router as terrain_router

from src.ai.pinn_sintering import ISRUSinteringEvaluator
from src.ai.segmentor import SpatialSegmentor

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
PLANETARY_BODY = os.getenv("PLANETARY_BODY", "MARS")
TELEMETRY_RATE_HZ = min(max(float(os.getenv("TELEMETRY_BROADCAST_RATE_HZ", "1")), 0.1), 30.0)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aura_x.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI cores")
    app.state.pinn = await run_in_threadpool(ISRUSinteringEvaluator)
    app.state.segmentor = await run_in_threadpool(SpatialSegmentor)
    app.state.inference_lock = asyncio.Lock()
    logger.info(
        "AI cores ready on %s (pinn weights: %s, segmentor weights: %s)",
        app.state.pinn.device,
        app.state.pinn.weights_loaded,
        app.state.segmentor.weights_loaded,
    )
    yield
    app.state.pinn = None
    app.state.segmentor = None


app = FastAPI(
    title="AURA-X Orbital Engine API",
    description="Autonomous Underground & Regolith Architecture Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(terrain_router)

class SectorAnalysisRequest(BaseModel):
    sector_id: str = Field(min_length=1, max_length=64)
    grid_size: int = Field(default=256, ge=64, le=1024)
    laser_power_kw: float = Field(default=5.0, gt=0, le=100)
    density_g_cm3: float = Field(default=1.5, gt=0.5, le=3.5)


def seed_from_sector(sector_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sector_id.encode("utf-8")).digest()[:4], "big")


def run_sector_analysis(pinn, segmentor, payload: SectorAnalysisRequest):
    rng = np.random.default_rng(seed_from_sector(payload.sector_id))
    cube = rng.random((segmentor.in_channels, payload.grid_size, payload.grid_size))

    mineral_mask = segmentor.segment_orbital_tile(cube)

    sio2_matrix = np.where(mineral_mask == 1, 0.55, 0.35)
    fe2o3_matrix = np.where(mineral_mask == 2, 0.40, 0.20)

    pinn_results = pinn.evaluate_grid(
        sio2_matrix,
        fe2o3_matrix,
        density_grid=payload.density_g_cm3,
        laser_power_kw=payload.laser_power_kw,
    )
    return mineral_mask, pinn_results


@app.get("/")
async def root(request: Request):
    return {
        "status": "ONLINE",
        "system": "AURA-X Planetary Engine",
        "planetary_body": PLANETARY_BODY,
        "environment": ENVIRONMENT,
        "compute_device": str(request.app.state.pinn.device),
        "version": app.version,
    }


@app.get("/api/v1/health")
async def health_check(request: Request):
    state = request.app.state
    return {
        "status": "HEALTHY",
        "compute_device": str(state.pinn.device),
        "ai_cores": {"pinn": "LOADED", "transformer": "LOADED"},
        "weights_loaded": {
            "pinn": state.pinn.weights_loaded,
            "transformer": state.segmentor.weights_loaded,
        },
    }


@app.post("/api/v1/analyze/sector")
async def analyze_sector(payload: SectorAnalysisRequest, request: Request):
    state = request.app.state
    try:
        async with state.inference_lock:
            mineral_mask, pinn_results = await run_in_threadpool(
                run_sector_analysis, state.pinn, state.segmentor, payload
            )
    except Exception:
        logger.exception("Sector analysis failed for sector %s", payload.sector_id)
        raise HTTPException(status_code=500, detail="Inference processing error")

    return {
        "sector_id": payload.sector_id,
        "data_source": "SYNTHETIC_RANDOM_INPUT",
        "models_trained": bool(state.pinn.weights_loaded and state.segmentor.weights_loaded),
        "metrics": {
            "compressive_strength_mpa_avg": round(float(np.mean(pinn_results["compressive_strength_mpa"])), 2),
            "sinterability_score_avg_percent": round(float(np.mean(pinn_results["sintering_feasibility_score"])), 2),
            "optimal_3d_print_zones_count": int(np.sum(pinn_results["optimal_print_zones"])),
            "mineral_breakdown": {
                "basalt_percent": round(float(np.mean(mineral_mask == 0) * 100), 2),
                "silica_percent": round(float(np.mean(mineral_mask == 1) * 100), 2),
                "iron_oxide_percent": round(float(np.mean(mineral_mask == 2) * 100), 2),
                "water_ice_percent": round(float(np.mean(mineral_mask == 3) * 100), 2),
            },
        },
        "status": "COMPLETED",
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    rng = np.random.default_rng()
    interval = 1.0 / TELEMETRY_RATE_HZ
    try:
        while True:
            await websocket.send_json(
                {
                    "timestamp": time.time(),
                    "simulated": True,
                    "orbit_altitude_km": round(250.4 + float(rng.normal(0, 0.1)), 2),
                    "radiation_shielding_index": round(85.5 + float(rng.normal(0, 0.5)), 1),
                    "active_rover_node_status": "SEARCHING_LAVA_TUBES",
                    "sensor_readings": {
                        "surface_temp_c": round(-62.3 + float(rng.normal(0, 0.2)), 1),
                        "subsurface_radar_reflectance_db": round(14.2 + float(rng.normal(0, 0.3)), 2),
                    },
                }
            )
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info("Telemetry client disconnected")
    except Exception:
        logger.exception("Telemetry stream failed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=HOST,
        port=PORT,
        reload=ENVIRONMENT == "development",
        log_level=LOG_LEVEL.lower(),
    )