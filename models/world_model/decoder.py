import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import symlog, two_hot_encode, two_hot_decode, make_bins


class ImageDecoder(nn.Module):
    """Transposed-CNN decoder: feat → (B, 3, H, W).

    Mirrors DreamerEncoder: spatial sizes scale up by 2× per layer.
    image_size must be divisible by 16 (e.g. 64, 224).
    """

    def __init__(self, feat_dim: int, image_size: int = 224, depth: int = 48):
        super().__init__()
        assert image_size % 16 == 0, f"image_size must be divisible by 16, got {image_size}"
        self.depth = depth
        self.image_size = image_size
        self.spatial = image_size // 16
        self.fc = nn.Linear(feat_dim, depth * 8 * self.spatial * self.spatial)
        channels = [depth * 8, depth * 4, depth * 2, depth, 3]
        layers: list = []
        for i in range(4):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1))
            if i < 3:
                layers.append(nn.SiLU())
        self.cnn = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*, feat_dim) → (*, 3, image_size, image_size)."""
        leading = feat.shape[:-1]
        h = self.fc(feat).view(-1, self.depth * 8, self.spatial, self.spatial)
        out = self.cnn(h)
        return out.view(*leading, 3, self.image_size, self.image_size)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss.

        feat:   (B, T, feat_dim)
        target: (B, T, 3, H, W) float in [0, 1]
        """
        pred = self.forward(feat.flatten(0, 1))
        return F.mse_loss(pred, target.flatten(0, 1))


class RewardDecoder(nn.Module):
    """MLP reward predictor: feat → two-hot logits (symlog space).

    Predicts the symlog-transformed reward using a two-hot distribution
    over 255 evenly spaced bins in [-20, 20].
    """

    N_BINS = 255

    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.N_BINS),
        )
        self.register_buffer("bins", make_bins(self.N_BINS))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*) → (*, N_BINS) logits."""
        return self.net(feat)

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*) → (*) scalar reward."""
        return two_hot_decode(self.forward(feat), self.bins)

    def loss(self, feat: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """Two-hot cross-entropy loss in symlog space.

        feat:   (B, T, feat_dim)
        reward: (B, T)
        """
        logits = self.forward(feat).flatten(0, 1)                    # (B*T, N_BINS)
        targets = two_hot_encode(symlog(reward.flatten(0, 1)), self.bins)
        return F.cross_entropy(logits, targets)


class ContinueDecoder(nn.Module):
    """MLP continue predictor: feat → Bernoulli logit."""

    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*, feat_dim) → (*, 1) logit."""
        return self.net(feat)

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (*, feat_dim) → (*) continuation probability."""
        return torch.sigmoid(self.forward(feat).squeeze(-1))

    def loss(self, feat: torch.Tensor, cont_target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss.

        feat:        (B, T, feat_dim)
        cont_target: (B, T) float — 1.0 if episode continues, 0.0 at terminal
        """
        logits = self.forward(feat).squeeze(-1).flatten(0, 1)  # (B*T,)
        return F.binary_cross_entropy_with_logits(logits, cont_target.flatten(0, 1))
