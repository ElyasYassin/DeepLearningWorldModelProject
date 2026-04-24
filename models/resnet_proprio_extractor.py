import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class ResnetProprioExtractor(BaseFeaturesExtractor):
    """
    What this is for:
    - Takes dict observations with keys: "image" and "proprio"
    - Runs image through pretrained ResNet-18
    - Concatenates ResNet image features with proprio features
    - Returns one combined feature vector to PPO
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        visual_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        self.image_space = observation_space.spaces["image"]
        self.proprio_space = observation_space.spaces["proprio"]
        proprio_dim = self.proprio_space.shape[0]

        features_dim = visual_dim + proprio_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # Remove final classifier -> output becomes 512-dim
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.visual_proj = nn.Linear(512, visual_dim)

    def forward(self, observations):
        """
        observations["image"]:
            can arrive as (B, H, W, C) or (B, C, H, W)
        observations["proprio"]:
            shape (B, P)
        """
        x_img = observations["image"].float()
        x_prop = observations["proprio"].float()

        # Robust handling for either HWC or CHW
        if x_img.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got {x_img.shape}")

        # If image is BHWC -> convert to BCHW
        if x_img.shape[-1] == 3:
            x_img = x_img.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        x_img = x_img / 255.0

        # ResNet was trained on larger images; resize 64x64 -> 224x224
        x_img = F.interpolate(
            x_img,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x_img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x_img.device).view(1, 3, 1, 1)
        x_img = (x_img - mean) / std

        # ResNet output: (B, 512, 1, 1)
        x_img = self.backbone(x_img)
        x_img = torch.flatten(x_img, 1)  # (B, 512)
        x_img = self.visual_proj(x_img)  # (B, visual_dim)

        # Concatenate visual + proprio
        x = torch.cat([x_img, x_prop], dim=1)
        return x
