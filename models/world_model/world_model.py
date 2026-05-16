import torch
import torch.nn as nn

from .encoder import DreamerEncoder, ResNetEncoder
from .rssm import RSSM, TransformerRSSM
from .decoder import ImageDecoder, RewardDecoder, ContinueDecoder
from .utils import kl_loss


class WorldModel(nn.Module):
    """DreamerV3 world model: encoder + RSSM + decoders.

    Bundles all world-model components so they can be optimized jointly
    with a single optimizer in train_dreamer.py.
    """

    def __init__(
        self,
        action_dim: int,
        image_size: int = 64,
        depth: int = 48,
        embed_dim: int = 512,
        h_dim: int = 512,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        kl_scale: float = 0.1,
        free_bits: float = 1.0,
        kl_balance: float = 0.8,
        encoder_type: str = "resnet",
        pretrained_encoder: bool = True,
        freeze_encoder: bool = False,
        dynamics_type: str = "transformer",
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        max_ctx_len: int = 64,
    ):
        super().__init__()
        self.kl_scale = kl_scale
        self.free_bits = free_bits
        self.kl_balance = kl_balance

        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                embed_dim=embed_dim,
                pretrained=pretrained_encoder,
                freeze=freeze_encoder,
            )
        else:
            self.encoder = DreamerEncoder(image_size=image_size, depth=depth, embed_dim=embed_dim)

        if dynamics_type == "transformer":
            self.rssm = TransformerRSSM(
                embed_dim=embed_dim,
                action_dim=action_dim,
                h_dim=h_dim,
                stoch_size=stoch_size,
                stoch_classes=stoch_classes,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_ctx_len=max_ctx_len,
            )
        else:
            self.rssm = RSSM(
                embed_dim=embed_dim,
                action_dim=action_dim,
                h_dim=h_dim,
                stoch_size=stoch_size,
                stoch_classes=stoch_classes,
            )
        feat_dim = self.rssm.feat_dim

        self.image_dec = ImageDecoder(feat_dim=feat_dim, image_size=image_size, depth=depth)
        self.reward_dec = RewardDecoder(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.cont_dec = ContinueDecoder(feat_dim=feat_dim, hidden_dim=hidden_dim)

    @property
    def feat_dim(self) -> int:
        return self.rssm.feat_dim

    def forward(
        self,
        obs: torch.Tensor,     # (B, T, 3, 64, 64) float in [0, 1]
        actions: torch.Tensor, # (B, T, action_dim)
        rewards: torch.Tensor, # (B, T)
        cont: torch.Tensor,    # (B, T) float — 1 if episode continues
    ):
        """Compute all world-model losses and return a dict.

        Returns:
            losses: dict with keys 'recon', 'kl', 'reward', 'cont', 'total'
            feats:  (B, T, feat_dim) — detached from the wm graph, for actor-critic seeding
        """
        B, T = obs.shape[:2]

        # Encode all observations in one batched forward pass
        embeds = self.encoder(obs.flatten(0, 1)).view(B, T, -1)  # (B, T, embed_dim)

        feats, prior_lgs, post_lgs = self.rssm.observe(embeds, actions)

        recon_loss  = self.image_dec.loss(feats, obs)
        reward_loss = self.reward_dec.loss(feats, rewards)
        cont_loss   = self.cont_dec.loss(feats, cont)
        kl          = kl_loss(post_lgs, prior_lgs, self.free_bits, self.kl_balance)

        total = recon_loss + self.kl_scale * kl + reward_loss + cont_loss

        losses = {
            "recon":  recon_loss,
            "kl":     kl,
            "reward": reward_loss,
            "cont":   cont_loss,
            "total":  total,
        }
        return losses, feats.detach()
