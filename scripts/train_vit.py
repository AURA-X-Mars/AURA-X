import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ai.vit_segmentor import MineralSegmentationViT
except ImportError:
    class MineralSegmentationViT(nn.Module):
        def __init__(self, in_channels=8, num_classes=4, image_size=64, patch_size=8, embed_dim=128, num_heads=4, num_layers=4):
            super().__init__()
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2

            self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 64, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            )

        def forward(self, x):
            B, C, H, W = x.shape

            patches = self.patch_embed(x)
            patches_flat = patches.flatten(2).transpose(1, 2)

            tokens = patches_flat + self.pos_embed
            encoded = self.transformer(tokens)

            h_patches = H // self.patch_size
            w_patches = W // self.patch_size
            feature_map = encoded.transpose(1, 2).reshape(B, -1, h_patches, w_patches)

            out = self.decoder(feature_map)
            return out

class SyntheticHyperspectralDataset(Dataset):
    def __init__(self, num_samples=500, in_channels=8, num_classes=4, image_size=64):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.in_channels, self.image_size, self.image_size)

        mask = torch.randint(0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long)
        
        return image, mask

def train_vit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA-X] Po përdoret pajisja për ViT: {device}")

    in_channels = 8
    num_classes = 4
    image_size = 64
    batch_size = 16
    epochs = 40

    dataset = SyntheticHyperspectralDataset(
        num_samples=400, 
        in_channels=in_channels, 
        num_classes=num_classes, 
        image_size=image_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MineralSegmentationViT(
        in_channels=in_channels, 
        num_classes=num_classes, 
        image_size=image_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("[AURA-X] Fillon trajnimi i Vision Transformer (ViT)...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = total_loss / len(dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Humbja e Segmentimit (CrossEntropy): {epoch_loss:.6f}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, "vit_segmentor.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[AURA-X] Trajnimi i ViT përfundoi me sukses! Peshat u ruajtën te: {model_save_path}")

if __name__ == "__main__":
    train_vit()