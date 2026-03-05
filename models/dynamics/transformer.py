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
        raise NotImplementedError

    def forward(
        self,
        latents: torch.Tensor,   # (batch, seq_len, latent_dim)
        actions: torch.Tensor,   # (batch, seq_len, action_dim)
    ) -> torch.Tensor:           # (batch, latent_dim)
        """Predict the next latent state from the history of latents and actions."""
        raise NotImplementedError
