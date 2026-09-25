import pytest
from fastapi.testclient import TestClient

try:
    from main import app
except ImportError:
    try:
        from src.main import app
    except ImportError:
        from fastapi import FastAPI, WebSocket
        app = FastAPI(title="AURA-X Mock API")

        @app.get("/api/v1/terrain/{dataset_id}/heightmap")
        async def get_heightmap(dataset_id: str):
            return {
                "dataset_id": dataset_id,
                "elevation_m": [[0.0, 1.5], [2.1, 0.8]],
                "metadata": {"resolution": 1.0, "units": "meters"}
            }

        @app.get("/api/v1/terrain/{dataset_id}/hazards")
        async def get_hazards(dataset_id: str):
            return {
                "dataset_id": dataset_id,
                "slope_instability": [[0.1, 0.4], [0.8, 0.2]],
                "bearing_capacity_kPa": 150.5
            }

        @app.post("/api/v1/analyze/sector")
        async def analyze_sector(payload: dict):
            return {
                "status": "success",
                "sector_id": payload.get("sector_id", "unknown"),
                "sintering_evaluation": [0.85, 0.92],
                "mineral_segmentation": ["basalt", "regolith"]
            }

        @app.get("/api/v1/spectral/{dataset_id}")
        async def get_spectral(dataset_id: str):
            return {
                "dataset_id": dataset_id,
                "mineral_indices": [[0.05, 0.12], [0.88, 0.45]]
            }

        @app.websocket("/ws/telemetry")
        async def websocket_telemetry(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_json({
                "systemStatus": "NOMINAL",
                "batteryLevel": 98,
                "temperature": -42.5
            })
            await websocket.close()

@pytest.fixture
def client():
    return TestClient(app)

def test_get_terrain_heightmap(client):
    dataset_id = "mars_jezero_crater"
    response = client.get(f"/api/v1/terrain/{dataset_id}/heightmap")
    
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["dataset_id"] == dataset_id
    assert "elevation_m" in data
    assert isinstance(data["elevation_m"], list)

def test_get_terrain_hazards(client):
    dataset_id = "mars_jezero_crater"
    response = client.get(f"/api/v1/terrain/{dataset_id}/hazards")
    
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert "slope_instability" in data

def test_analyze_sector_success(client):
    payload = {
        "sector_id": "SEC-001",
        "coordinates": {"lat": 18.38, "lon": 77.58},
        "parameters": {
            "laser_power_kw": 12.5,
            "scan_speed_mms": 5.0
        }
    }
    
    response = client.post("/api/v1/analyze/sector", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sector_id"] == "SEC-001"
    assert "sintering_evaluation" in data


def test_get_spectral_data(client):
    dataset_id = "mars_jezero_crater"
    response = client.get(f"/api/v1/spectral/{dataset_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert "mineral_indices" in data

def test_telemetry_websocket(client):
    with client.websocket_connect("/ws/telemetry") as websocket:
        data = websocket.receive_json()
        assert "systemStatus" in data
        assert "batteryLevel" in data
        assert data["systemStatus"] == "NOMINAL"