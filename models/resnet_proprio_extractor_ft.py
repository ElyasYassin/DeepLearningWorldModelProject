import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class ResnetProprioExtractorFT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        visual_dim: int = 128,
    ):
        self.image_space = observation_space.spaces["image"]
        self.proprio_space = observation_space.spaces["proprio"]
        proprio_dim = self.proprio_space.shape[0]

        features_dim = visual_dim + proprio_dim
        super().__init__(observation_space, features_dim=features_dim)

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # Remove final classifier
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the last residual block: layer4
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Optional: also unfreeze final batchnorm if needed later
        # for param in self.backbone.layer3.parameters():
        #     param.requires_grad = True

        self.visual_proj = nn.Linear(512, visual_dim)

    def forward(self, observations):
        x_img = observations["image"].float()
        x_prop = observations["proprio"].float()

        if x_img.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got {x_img.shape}")

        if x_img.shape[-1] == 3:
            x_img = x_img.permute(0, 3, 1, 2)

        x_img = x_img / 255.0
        x_img = F.interpolate(
            x_img, size=(224, 224), mode="bilinear", align_corners=False
        )

        mean = torch.tensor([0.485, 0.456, 0.406], device=x_img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x_img.device).view(1, 3, 1, 1)
        x_img = (x_img - mean) / std

        x_img = self.backbone(x_img)  # shape: (B, 512)
        x_img = self.visual_proj(x_img)  # shape: (B, visual_dim)

        x = torch.cat([x_img, x_prop], dim=1)
        return x
