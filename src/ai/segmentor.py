import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ai.pinn_sintering import resolve_device, save_weights

logger = logging.getLogger("aura_x.segmentor")

CLASS_NAMES = ("Basalt", "Silica_Dust", "Iron_Oxide", "Water_Ice", "Void")
DEFAULT_IMG_SIZE = 256
DEFAULT_PATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 4


class HyperspectralPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class MineralSegmentationViT(nn.Module):
    def __init__(
        self,
        img_size=DEFAULT_IMG_SIZE,
        patch_size=DEFAULT_PATCH_SIZE,
        in_channels=12,
        num_classes=5,
        embed_dim=256,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        if patch_size < 4 or patch_size % 4 != 0:
            raise ValueError("patch_size must be a multiple of 4")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.patch_embed = HyperspectralPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        first_stride = patch_size // 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=first_stride, stride=first_stride),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, num_classes, kernel_size=4, stride=4),
        )

    @staticmethod
    def standardize(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + 1e-6)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        if height != self.img_size or width != self.img_size:
            raise ValueError(f"expected {self.img_size}x{self.img_size} tiles, got {height}x{width}")

        tokens = self.patch_embed(self.standardize(x)) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        grid = self.patch_embed.grid_size
        feature_map = tokens.transpose(1, 2).reshape(batch, -1, grid, grid)
        return self.decoder(feature_map)


def train_segmentor(
    model,
    images,
    masks,
    epochs=50,
    batch_size=8,
    learning_rate=3e-4,
    class_weights=None,
    device=None,
):
    images = np.asarray(images, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.int64)
    if images.ndim != 4 or masks.shape != (images.shape[0], images.shape[2], images.shape[3]):
        raise ValueError("images must be (N, C, H, W) and masks (N, H, W)")

    device = device or next(model.parameters()).device
    model.to(device).train()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(images), torch.from_numpy(masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    history = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_images, batch_masks in loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_images), batch_masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * batch_images.shape[0]
        epoch_loss = running_loss / images.shape[0]
        history.append(epoch_loss)
        logger.info("epoch %d loss=%.5f", epoch, epoch_loss)

    model.eval()
    return history


class SpatialSegmentor:
    def __init__(self, in_channels=12, num_classes=5, model_weights_path=None):
        self.device = resolve_device(os.getenv("COMPUTE_DEVICE", "auto"))
        self.in_channels = in_channels
        self.model = MineralSegmentationViT(in_channels=in_channels, num_classes=num_classes).to(self.device)
        self.img_size = self.model.img_size
        self.weights_loaded = False

        if num_classes == len(CLASS_NAMES):
            self.classes = list(CLASS_NAMES)
        else:
            self.classes = [f"class_{index}" for index in range(num_classes)]

        weights_path = model_weights_path or os.getenv("VISION_TRANSFORMER_MODEL_PATH")
        if weights_path and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.weights_loaded = True
            logger.info("Loaded segmentor weights from %s", weights_path)
        else:
            logger.warning("No trained segmentor weights found at %s, masks are not meaningful", weights_path)

        self.model.eval()
        logger.info("Segmentor initialised on %s", self.device)

    def _predict_batch(self, tiles):
        tensor = torch.from_numpy(np.stack(tiles)).to(self.device)
        logits = self.model(tensor)
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

    def segment_orbital_tile(self, hyperspectral_tensor):
        cube = np.asarray(hyperspectral_tensor, dtype=np.float32)
        if cube.ndim == 4 and cube.shape[0] == 1:
            cube = cube[0]
        if cube.ndim != 3 or cube.shape[0] != self.in_channels:
            raise ValueError(f"expected a cube of shape ({self.in_channels}, H, W)")

        _, height, width = cube.shape
        tile = self.img_size
        pad_h = (-height) % tile
        pad_w = (-width) % tile
        padded = np.pad(cube, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        padded_h, padded_w = padded.shape[1], padded.shape[2]

        coordinates = [
            (top, left)
            for top in range(0, padded_h, tile)
            for left in range(0, padded_w, tile)
        ]
        mask = np.zeros((padded_h, padded_w), dtype=np.uint8)

        with torch.no_grad():
            for start in range(0, len(coordinates), INFERENCE_BATCH_SIZE):
                batch_coordinates = coordinates[start:start + INFERENCE_BATCH_SIZE]
                tiles = [padded[:, top:top + tile, left:left + tile] for top, left in batch_coordinates]
                predictions = self._predict_batch(tiles)
                for (top, left), prediction in zip(batch_coordinates, predictions):
                    mask[top:top + tile, left:left + tile] = prediction

        return mask[:height, :width]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    segmentor = SpatialSegmentor(in_channels=12, num_classes=5)

    rng = np.random.default_rng(0)
    for shape in ((12, 256, 256), (12, 300, 500)):
        cube = rng.random(shape)
        mask = segmentor.segment_orbital_tile(cube)
        print(f"Input {shape} -> mask {mask.shape}")

    print(f"Weights loaded: {segmentor.weights_loaded}")
    values, counts = np.unique(mask, return_counts=True)
    for value, count in zip(values, counts):
        print(f"  {segmentor.classes[value]}: {count} pixels")