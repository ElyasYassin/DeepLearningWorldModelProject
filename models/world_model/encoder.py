import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetEncoder(nn.Module):
    """ResNet18 encoder: (B, 3, H, W) float in [0,1] → (B, embed_dim).

    Uses pretrained ImageNet weights by default (same backbone as PPO-ResNet18 baselines).
    ImageNet normalisation is applied internally so calling code is unchanged.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop FC → (B,512,1,1)
        self.proj = nn.Linear(512, embed_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0,1] → (B, embed_dim)."""
        x = (x - self.mean) / self.std
        return self.proj(self.backbone(x).flatten(1))


class DreamerEncoder(nn.Module):
    """CNN encoder: (B, 3, H, W) float in [0,1] → (B, embed_dim).

    4 conv layers (kernel=4, stride=2, padding=1) each halve the spatial size,
    so image_size must be divisible by 16. Examples:
      64  → 32 → 16 → 8 → 4   (spatial = 4)
      224 → 112 → 56 → 28 → 14 (spatial = 14)
    """

    def __init__(self, image_size: int = 224, depth: int = 48, embed_dim: int = 1024):
        super().__init__()
        assert image_size % 16 == 0, f"image_size must be divisible by 16, got {image_size}"
        spatial = image_size // 16
        channels = [3, depth, depth * 2, depth * 4, depth * 8]
        layers: list = []
        for i in range(4):
            layers += [nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        self.out_proj = nn.Linear(depth * 8 * spatial * spatial, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, embed_dim)."""
        h = self.cnn(x - 0.5)
        return self.out_proj(h.flatten(1))
