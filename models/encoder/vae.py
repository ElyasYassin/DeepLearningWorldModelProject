"""
Variational Autoencoder (VAE) vision encoder.

Compresses high-dimensional RGB camera frames into a compact latent vector z
that captures the essential visual information for tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    """Convolutional encoder: image -> (mu, log_var).

    Uses a pretrained ResNet-18 backbone with layer4 fine-tuned and earlier
    layers frozen. Projects 512-dim features to VAE distribution parameters.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()   # keep named layer access, strip classifier
        self.backbone = backbone

        # Freeze all, then unfreeze layer4 for fine-tuning
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.proj = nn.Linear(512, latent_dim * 2)
        self.latent_dim = latent_dim

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, 3, 64, 64) float in [0, 1] -> (mu, log_var) each (B, latent_dim)."""
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.imagenet_mean) / self.imagenet_std
        features = self.backbone(x)              # (B, 512) — ResNet flattens avgpool before Identity fc
        out = self.proj(features)                 # (B, latent_dim*2)
        mu = out[:, : self.latent_dim]
        log_var = out[:, self.latent_dim :]
        return mu, log_var


class Decoder(nn.Module):
    """Convolutional decoder: latent z -> reconstructed image."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 32->64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        """z: (B, latent_dim) -> (B, 3, 64, 64) in [0, 1]."""
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """Full VAE: encodes images to latent space and reconstructs them."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean latent vector (no sampling) for use at inference."""
        mu, _ = self.encoder(x)
        return mu
