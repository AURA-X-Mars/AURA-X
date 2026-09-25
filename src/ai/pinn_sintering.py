import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("aura_x.pinn")

INPUT_FEATURES = (
    "sio2_fraction",
    "fe2o3_fraction",
    "bulk_density_g_cm3",
    "laser_energy_norm",
    "temperature_c",
)
FEATURE_MIN = (0.30, 0.05, 1.0, 0.0, -140.0)
FEATURE_MAX = (0.70, 0.40, 2.5, 2.0, 20.0)
TARGET_SCALE = (10.0, 100.0)
LASER_POWER_REFERENCE_KW = 10.0
OPTIMAL_SCORE_THRESHOLD = 80.0
INFERENCE_BATCH_SIZE = 65536


def resolve_device(requested):
    requested = (requested or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu", "auto"):
        if cuda_available:
            return torch.device("cuda")
        if requested != "auto":
            logger.warning("CUDA requested but unavailable, falling back to CPU")
        return torch.device("cpu")
    logger.warning("Unknown COMPUTE_DEVICE '%s', using auto", requested)
    return torch.device("cuda" if cuda_available else "cpu")


class RegolithSinteringPINN(nn.Module):
    def __init__(self, hidden_sizes=(64, 128, 64, 32)):
        super().__init__()
        layers = []
        in_dim = len(INPUT_FEATURES)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)
        self.register_buffer("feature_min", torch.tensor(FEATURE_MIN, dtype=torch.float32))
        self.register_buffer("feature_max", torch.tensor(FEATURE_MAX, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(TARGET_SCALE, dtype=torch.float32))

    def normalize(self, x):
        return (x - self.feature_min) / (self.feature_max - self.feature_min)

    def forward(self, x):
        out = self.head(self.body(self.normalize(x)))
        strength = F.softplus(out[:, 0:1]) * self.target_scale[0]
        score = torch.sigmoid(out[:, 1:2]) * self.target_scale[1]
        return torch.cat([strength, score], dim=1)


class SinteringPhysicsLoss(nn.Module):
    def __init__(self, lambda_physics=0.1, monotonic_features=("bulk_density_g_cm3", "laser_energy_norm")):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.monotonic_indices = [INPUT_FEATURES.index(name) for name in monotonic_features]

    def forward(self, model, inputs, targets):
        x = inputs.clone().requires_grad_(True)
        predictions = model(x)
        scale = model.target_scale
        data_loss = F.mse_loss(predictions / scale, targets / scale)

        gradients = torch.autograd.grad(predictions[:, 0].sum(), x, create_graph=True)[0]
        feature_range = model.feature_max - model.feature_min
        slopes = gradients * feature_range / scale[0]

        physics_loss = torch.zeros((), device=inputs.device)
        for index in self.monotonic_indices:
            physics_loss = physics_loss + F.relu(-slopes[:, index]).mean()

        total_loss = data_loss + self.lambda_physics * physics_loss
        return total_loss, data_loss.detach(), physics_loss.detach()


def train_model(model, inputs, targets, epochs=2000, learning_rate=1e-3, lambda_physics=0.1, device=None, log_every=200):
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != len(INPUT_FEATURES):
        raise ValueError(f"inputs must have shape (N, {len(INPUT_FEATURES)})")
    if targets.shape != (inputs.shape[0], 2):
        raise ValueError("targets must have shape (N, 2)")

    device = device or next(model.parameters()).device
    model.to(device).train()
    x = torch.as_tensor(inputs, device=device)
    y = torch.as_tensor(targets, device=device)

    criterion = SinteringPhysicsLoss(lambda_physics=lambda_physics)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        total_loss, data_loss, physics_loss = criterion(model, x, y)
        total_loss.backward()
        optimizer.step()
        history.append((float(total_loss), float(data_loss), float(physics_loss)))
        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "epoch %d total=%.5f data=%.5f physics=%.5f",
                epoch,
                history[-1][0],
                history[-1][1],
                history[-1][2],
            )

    model.eval()
    return history


def save_weights(model, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)


class ISRUSinteringEvaluator:
    def __init__(self, model_weights_path=None):
        self.device = resolve_device(os.getenv("COMPUTE_DEVICE", "auto"))
        self.model = RegolithSinteringPINN().to(self.device)
        self.weights_loaded = False

        weights_path = model_weights_path or os.getenv("PINN_SINTERING_MODEL_PATH")
        if weights_path and Path(weights_path).is_file():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.weights_loaded = True
            logger.info("Loaded PINN weights from %s", weights_path)
        else:
            logger.warning("No trained PINN weights found at %s, predictions are not meaningful", weights_path)

        self.model.eval()
        logger.info("Sintering core initialised on %s", self.device)

    def evaluate_grid(self, sio2_grid, fe2o3_grid, density_grid=1.5, laser_power_kw=5.0, temperature_c=-60.0):
        sio2 = np.asarray(sio2_grid, dtype=np.float32)
        fe2o3 = np.asarray(fe2o3_grid, dtype=np.float32)
        if sio2.shape != fe2o3.shape:
            raise ValueError("sio2_grid and fe2o3_grid must have the same shape")

        shape = sio2.shape
        density = np.broadcast_to(np.asarray(density_grid, dtype=np.float32), shape)
        energy = np.full(shape, laser_power_kw / LASER_POWER_REFERENCE_KW, dtype=np.float32)
        temperature = np.broadcast_to(np.asarray(temperature_c, dtype=np.float32), shape)

        features = np.stack(
            [sio2.ravel(), fe2o3.ravel(), density.ravel(), energy.ravel(), temperature.ravel()],
            axis=1,
        )

        outputs = []
        with torch.no_grad():
            for start in range(0, features.shape[0], INFERENCE_BATCH_SIZE):
                batch = torch.from_numpy(features[start:start + INFERENCE_BATCH_SIZE]).to(self.device)
                outputs.append(self.model(batch).cpu().numpy())
        predictions = np.concatenate(outputs, axis=0)

        strength_map = predictions[:, 0].reshape(shape)
        score_map = predictions[:, 1].reshape(shape)

        return {
            "compressive_strength_mpa": strength_map,
            "sintering_feasibility_score": score_map,
            "optimal_print_zones": score_map > OPTIMAL_SCORE_THRESHOLD,
            "weights_loaded": self.weights_loaded,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = ISRUSinteringEvaluator()

    rng = np.random.default_rng(0)
    sio2 = rng.uniform(0.35, 0.55, (10, 10))
    fe2o3 = rng.uniform(0.10, 0.30, (10, 10))

    results = evaluator.evaluate_grid(sio2, fe2o3)

    print(f"Weights loaded:          {results['weights_loaded']}")
    print(f"Mean predicted strength: {np.mean(results['compressive_strength_mpa']):.2f} MPa")
    print(f"Mean viability score:    {np.mean(results['sintering_feasibility_score']):.2f}%")
    print(f"Optimal zones:           {int(np.sum(results['optimal_print_zones']))} / 100")