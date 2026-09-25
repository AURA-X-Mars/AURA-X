import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ai.pinn_sintering import RegolithSinteringPINN
except ImportError:
    class RegolithSinteringPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(5, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.network(x)

class LocalSinteringLoss(nn.Module):
    def __init__(self, lambda_physics=0.15):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics

    def forward(self, predictions, targets, inputs):
        mse_loss = self.mse(predictions, targets)

        grad_energy = torch.autograd.grad(
            outputs=predictions.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        physics_loss = torch.mean(torch.relu(-grad_energy))
        return mse_loss + self.lambda_physics * physics_loss


def generate_synthetic_sintering_data(num_samples=2000):
    torch.manual_seed(42)
    
    laser_energy = torch.rand(num_samples, 1) * 50.0 + 10.0
    exposure_time = torch.rand(num_samples, 1) * 5.0 + 0.5
    initial_density = torch.rand(num_samples, 1) * 0.8 + 1.2
    iron_content = torch.rand(num_samples, 1) * 0.3 + 0.05
    ambient_temp = torch.rand(num_samples, 1) * 100.0 - 120.0
    
    inputs = torch.cat([laser_energy, exposure_time, initial_density, iron_content, ambient_temp], dim=1)
    
    targets = (
        0.15 * laser_energy 
        + 0.8 * exposure_time 
        + 2.5 * (initial_density ** 2) 
        + 1.2 * iron_content 
        + 0.05 * (ambient_temp + 120.0)
        + torch.randn(num_samples, 1) * 0.1
    )
    
    return inputs, targets


def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA-X] Po përdoret pajisja: {device}")

    inputs, targets = generate_synthetic_sintering_data(num_samples=3000)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = RegolithSinteringPINN().to(device)
    criterion = LocalSinteringLoss(lambda_physics=0.15)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    epochs = 40
    print("[AURA-X] Fillon trajnimi i PINN...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device).requires_grad_(True)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            
            loss = criterion(predictions, batch_y, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.6f}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, "pinn_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[AURA-X] Trajnimi përfundoi! Peshat u ruajtën te: {model_save_path}")

if __name__ == "__main__":
    train_pinn()