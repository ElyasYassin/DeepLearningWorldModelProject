"""
Transformer-based latent dynamics model.

Predicts the next latent state z_{t+1} given a history of (z, a) pairs.
Uses a transformer architecture to capture long-range temporal dependencies,
as an alternative to the RNN used in the original World Models / Dreamer papers.
"""

import torch
import torch.nn as nn


class LatentDynamicsModel(nn.Module):
    """
    Sequence model: (z_0, a_0), ..., (z_t, a_t) -> z_{t+1}

    Args:
        latent_dim:   Dimensionality of the latent state z.
        action_dim:   Dimensionality of the action space.
        d_model:      Internal transformer embedding size.
        nhead:        Number of attention heads.
        num_layers:   Number of transformer encoder layers.
        max_seq_len:  Maximum context window length.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(latent_dim + action_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, latent_dim)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        latents: torch.Tensor,   # (batch, seq_len, latent_dim)
        actions: torch.Tensor,   # (batch, seq_len, action_dim)
    ) -> torch.Tensor:           # (batch, latent_dim)
        """Predict the next latent state from the history of latents and actions."""
        B, T, _ = latents.shape
        x = torch.cat([latents, actions], dim=-1)   # (B, T, latent_dim+action_dim)
        x = self.input_proj(x)                       # (B, T, d_model)

        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        mask = self._make_causal_mask(T, x.device)
        x = self.transformer(x, mask=mask)           # (B, T, d_model)
        return self.output_proj(x)                   # (B, T, latent_dim)

    def step(
        self,
        ctx_latents: torch.Tensor,  # (B, ctx_len, latent_dim)
        ctx_actions: torch.Tensor,  # (B, ctx_len, action_dim)
    ) -> torch.Tensor:              # (B, latent_dim)
        """Single-step autoregressive prediction; truncates context to max_seq_len."""
        if ctx_latents.shape[1] > self.max_seq_len:
            ctx_latents = ctx_latents[:, -self.max_seq_len :, :]
            ctx_actions = ctx_actions[:, -self.max_seq_len :, :]
        return self.forward(ctx_latents, ctx_actions)[:, -1, :]  # (B, latent_dim)


class RewardPredictor(nn.Module):
    """Small MLP: latent z -> scalar reward prediction."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) or (B, T, latent_dim) -> (..., 1)."""
        return self.net(z)
