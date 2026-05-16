import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world_model.utils import symlog, two_hot_encode, two_hot_decode, make_bins


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 4) -> nn.Sequential:
    dims = [in_dim] + [hidden_dim] * layers + [out_dim]
    net: list = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            net.append(nn.SiLU())
    return nn.Sequential(*net)


class DreamerActor(nn.Module):
    """Continuous action policy: feat → tanh-squashed Normal.

    Used by the actor-critic and RSSM imagination rollouts.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.trunk = _mlp(feat_dim, hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, feat: torch.Tensor):
        """feat: (*, feat_dim)

        Returns:
            action:   (*, action_dim) tanh-squashed sample
            log_prob: (*) log-probability of the sample
            entropy:  (*) entropy of the distribution
        """
        h = self.trunk(feat)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        action = torch.tanh(raw)

        # Log-prob with tanh Jacobian correction
        log_prob = dist.log_prob(raw).sum(-1) - torch.log1p(-action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy


class DreamerCritic(nn.Module):
    """Value function: feat → two-hot value distribution in symlog space.

    Predicts the expected lambda return using a categorical distribution
    over 255 evenly spaced bins in [-20, 20] (symlog scale).
    """

    N_BINS = 255

    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, self.N_BINS)
        self.register_buffer("bins", make_bins(self.N_BINS))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*, feat_dim) → (*, N_BINS) logits."""
        return self.net(feat)

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*, feat_dim) → (*) scalar value estimate."""
        return two_hot_decode(self.forward(feat), self.bins)

    def loss(self, feat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Two-hot cross-entropy loss.

        feat:    (B, H, feat_dim)
        targets: (B, H) lambda returns
        """
        logits = self.forward(feat).flatten(0, 1)                         # (B*H, N_BINS)
        two_hot = two_hot_encode(symlog(targets.flatten(0, 1)), self.bins) # (B*H, N_BINS)
        return F.cross_entropy(logits, two_hot)


class ReturnNormalizer:
    """Running 5th/95th percentile normalizer for lambda returns (DreamerV3)."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._initialized: bool = False

    def update(self, returns: torch.Tensor) -> None:
        lo = float(torch.quantile(returns.detach(), 0.05))
        hi = float(torch.quantile(returns.detach(), 0.95))
        if not self._initialized:
            self._lo, self._hi = lo, hi
            self._initialized = True
        else:
            self._lo = self.decay * self._lo + (1 - self.decay) * lo
            self._hi = self.decay * self._hi + (1 - self.decay) * hi

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        scale = max(1.0, self._hi - self._lo)
        return (returns - self._lo) / scale
