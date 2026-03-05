"""
RL controller / policy.

Operates entirely in latent space: given the current latent state z,
outputs an action that maximizes cumulative reward (minimizing end-effector
distance to the tracked target).
"""

import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    MLP policy: latent z -> action.

    This serves as the actor in an actor-critic RL setup (e.g. SAC or PPO
    trained in latent space via imagined rollouts from the dynamics model).
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ValueFunction(nn.Module):
    """Critic: latent z -> scalar value estimate."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
