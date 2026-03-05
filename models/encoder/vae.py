"""
Variational Autoencoder (VAE) vision encoder.

Compresses high-dimensional RGB camera frames into a compact latent vector z
that captures the essential visual information for tracking.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional encoder: image -> (mu, log_var)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class Decoder(nn.Module):
    """Convolutional decoder: latent z -> reconstructed image."""

    def __init__(self, latent_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor):
        raise NotImplementedError


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
