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

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action (tanh-squashed) for inference."""
        h = self.net(z)
        return torch.tanh(self.mu_head(h))

    def sample(self, z: torch.Tensor):
        """Reparameterized stochastic sample for training.

        Returns:
            action:  tanh-squashed sample (B, action_dim)
            mu:      pre-tanh mean         (B, action_dim)
            log_std: clamped log std       (B, action_dim)
        """
        h = self.net(z)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        eps = torch.randn_like(std)
        raw = mu + eps * std
        action = torch.tanh(raw)
        return action, mu, log_std


class ValueFunction(nn.Module):
    """Critic: latent z -> scalar value estimate."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) or (B, T, latent_dim) -> (..., 1)."""
        return self.net(z)
