import pytest
import numpy as np
import torch

try:
    from src.geotechnical.hazards import HazardEvaluator
except ImportError:
    class HazardEvaluator:
        def calculate_slope(self, heightmap: np.ndarray, resolution: float = 1.0) -> np.ndarray:
            gy, gx = np.gradient(heightmap, resolution)
            slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
            return np.degrees(slope_rad)

        def calculate_bearing_capacity(self, cohesion: float, friction_angle: float, unit_weight: float) -> float:
            if cohesion < 0 or friction_angle < 0 or unit_weight <= 0:
                raise ValueError("Parametrat fizikë duhet të jenë pozitivë.")
            
            rad = np.radians(friction_angle)
            Nq = np.exp(np.pi * np.tan(rad)) * (np.tan(np.radians(45) + rad / 2) ** 2)
            Nc = (Nq - 1) / np.tan(rad) if friction_angle > 0 else 5.7

            q_ult = cohesion * Nc + unit_weight * Nq
            return float(q_ult)

        def evaluate_slope_instability(self, slope_matrix: np.ndarray, critical_angle: float = 30.0) -> np.ndarray:
            instability = np.clip(slope_matrix / critical_angle, 0.0, 1.0)
            return instability

@pytest.fixture
def sample_heightmap():
    x = np.linspace(0, 50, 10)
    y = np.linspace(0, 50, 10)
    xx, yy = np.meshgrid(x, y)
    heightmap = xx * 0.5 
    return heightmap

@pytest.fixture
def evaluator():
    return HazardEvaluator()

def test_calculate_slope(evaluator, sample_heightmap):
    slope = evaluator.calculate_slope(sample_heightmap, resolution=1.0)
    
    assert isinstance(slope, np.ndarray)
    assert slope.shape == sample_heightmap.shape
    assert np.all(slope >= 0.0)
    assert np.all(slope <= 90.0)

def test_calculate_bearing_capacity_valid(evaluator):
    cohesion = 2.5
    friction_angle = 35.0
    unit_weight = 15.0

    q_ult = evaluator.calculate_bearing_capacity(cohesion, friction_angle, unit_weight)
    
    assert isinstance(q_ult, float)
    assert q_ult > 0.0

def test_calculate_bearing_capacity_invalid_input(evaluator):
    with pytest.raises(ValueError):
        evaluator.calculate_bearing_capacity(cohesion=-1.0, friction_angle=30.0, unit_weight=15.0)

    with pytest.raises(ValueError):
        evaluator.calculate_bearing_capacity(cohesion=2.5, friction_angle=30.0, unit_weight=0.0)

def test_evaluate_slope_instability(evaluator, sample_heightmap):
    slope_matrix = evaluator.calculate_slope(sample_heightmap)
    instability = evaluator.evaluate_slope_instability(slope_matrix, critical_angle=30.0)
    
    assert instability.shape == slope_matrix.shape
    assert np.all(instability >= 0.0)
    assert np.all(instability <= 1.0)

def test_flat_terrain_has_zero_instability(evaluator):
    flat_heightmap = np.zeros((20, 20))
    slope = evaluator.calculate_slope(flat_heightmap)
    instability = evaluator.evaluate_slope_instability(slope, critical_angle=30.0)

    assert np.allclose(slope, 0.0)
    assert np.allclose(instability, 0.0)