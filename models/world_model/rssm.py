import torch
import torch.nn as nn
import math

from .utils import straight_through_sample


class RSSM(nn.Module):
    """GRU-based Recurrent State-Space Model (DreamerV3).

    State = (h, z) where:
      h: deterministic recurrent state from GRUCell
      z: stochastic categorical state (stoch_size × stoch_classes, one-hot)

    The feature vector fed to decoders and actor-critic is concat(h, z_flat).
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        h_dim: int = 512,
        stoch_size: int = 32,
        stoch_classes: int = 32,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_size * stoch_classes
        self.feat_dim = h_dim + stoch_size * stoch_classes

        # GRU input projection: concat(z_flat, a) → h_dim
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_flat + action_dim, h_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(h_dim, h_dim)

        # Prior (no observation): h → stoch logits
        self.prior_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, self.stoch_flat),
        )

        # Posterior (with observation): concat(h, embed) → stoch logits
        self.posterior_head = nn.Sequential(
            nn.Linear(h_dim + embed_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, self.stoch_flat),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device)
        return h, z

    def _feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate h and flattened z to form the feature vector."""
        return torch.cat([h, z.flatten(-2, -1)], dim=-1)

    def img_step(
        self,
        h: torch.Tensor,       # (B, h_dim)
        z: torch.Tensor,       # (B, stoch_size, stoch_classes)
        action: torch.Tensor,  # (B, action_dim)
    ):
        """One transition step without an observation (uses prior).

        Returns:
            h_new: (B, h_dim)
            prior_logits: (B, stoch_size, stoch_classes)
        """
        z_flat = z.flatten(-2, -1)
        gru_in = self.pre_gru(torch.cat([z_flat, action], -1))
        h_new = self.gru(gru_in, h)
        prior_logits = self.prior_head(h_new).view(-1, self.stoch_size, self.stoch_classes)
        return h_new, prior_logits

    def obs_step(
        self,
        h: torch.Tensor,       # (B, h_dim)
        z: torch.Tensor,       # (B, stoch_size, stoch_classes)
        action: torch.Tensor,  # (B, action_dim)
        embed: torch.Tensor,   # (B, embed_dim)
    ):
        """One transition step with an observation (computes both prior and posterior).

        Returns:
            h_new: (B, h_dim)
            prior_logits: (B, stoch_size, stoch_classes)
            posterior_logits: (B, stoch_size, stoch_classes)
        """
        h_new, prior_logits = self.img_step(h, z, action)
        posterior_logits = self.posterior_head(
            torch.cat([h_new, embed], -1)
        ).view(-1, self.stoch_size, self.stoch_classes)
        return h_new, prior_logits, posterior_logits

    def observe(
        self,
        embeds: torch.Tensor,   # (B, T, embed_dim)
        actions: torch.Tensor,  # (B, T, action_dim)
    ):
        """Process a full observed sequence (training with posterior z).

        Returns:
            feats:             (B, T, feat_dim)
            prior_logits:      (B, T, stoch_size, stoch_classes)
            posterior_logits:  (B, T, stoch_size, stoch_classes)
        """
        B, T, _ = embeds.shape
        h, z = self.initial_state(B, embeds.device)

        feats, prior_lgs, post_lgs = [], [], []
        for t in range(T):
            h, prior_lg, post_lg = self.obs_step(h, z, actions[:, t], embeds[:, t])
            z = straight_through_sample(post_lg)
            feats.append(self._feat(h, z))
            prior_lgs.append(prior_lg)
            post_lgs.append(post_lg)

        return (
            torch.stack(feats, dim=1),     # (B, T, feat_dim)
            torch.stack(prior_lgs, dim=1), # (B, T, stoch_size, stoch_classes)
            torch.stack(post_lgs, dim=1),  # (B, T, stoch_size, stoch_classes)
        )

    def imagine(
        self,
        init_feat: torch.Tensor,  # (B, feat_dim) — seed from observed sequence
        actor,
        horizon: int,
    ):
        """Unroll in imagination using the prior and an actor.

        Returns:
            feats:         (B, H, feat_dim)   — feats at each imagined step
            actions:       (B, H, action_dim) — actor actions
            terminal_feat: (B, feat_dim)      — state after last action (for bootstrapping)
        """
        B = init_feat.shape[0]
        h = init_feat[:, : self.h_dim]
        z = init_feat[:, self.h_dim :].view(B, self.stoch_size, self.stoch_classes)

        feats, acts = [], []
        for _ in range(horizon):
            feat = self._feat(h, z)
            action, _, _ = actor(feat)
            h, prior_logits = self.img_step(h, z, action)
            z = straight_through_sample(prior_logits)
            feats.append(feat)
            acts.append(action)

        terminal_feat = self._feat(h, z)
        return (
            torch.stack(feats, dim=1),   # (B, H, feat_dim)
            torch.stack(acts, dim=1),    # (B, H, action_dim)
            terminal_feat,               # (B, feat_dim)
        )


class TransformerRSSM(nn.Module):
    """Transformer-based RSSM: replaces GRUCell with a causal Transformer.

    State = (context_buffer, z) where:
      context_buffer: growing sequence of (z_flat, action) tokens
      z: stochastic categorical state (stoch_size × stoch_classes, one-hot)

    The deterministic state h is computed by running a causal Transformer over
    the context buffer and taking the output at the last position.
    Public interface is identical to RSSM so WorldModel/train code is unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        h_dim: int = 512,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        max_ctx_len: int = 64,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_size * stoch_classes
        self.feat_dim = h_dim + stoch_size * stoch_classes
        self.action_dim = action_dim
        self.max_ctx_len = max_ctx_len

        token_in = self.stoch_flat + action_dim
        self.token_emb = nn.Linear(token_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.h_proj = nn.Linear(d_model, h_dim)

        self.prior_head = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, self.stoch_flat),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(h_dim + embed_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, self.stoch_flat),
        )

    def _transformer_h(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx: (B, ctx_len, stoch_flat+action_dim) → h: (B, h_dim)."""
        emb = self.token_emb(ctx)                                    # (B, L, d_model)
        L = emb.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=ctx.device)
        out = self.transformer(emb, mask=mask, is_causal=True)       # (B, L, d_model)
        return self.h_proj(out[:, -1])                               # (B, h_dim)

    def _zero_h(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.h_dim, device=device)

    def initial_state(self, batch_size: int, device: torch.device):
        """Returns (ctx, z). ctx is an empty buffer; z is all-zeros."""
        ctx = torch.zeros(batch_size, 0, self.stoch_flat + self.action_dim, device=device)
        z   = torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device)
        return ctx, z

    def _feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z.flatten(-2, -1)], dim=-1)

    def _append_token(self, ctx: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Append (z_flat, action) to the context buffer, truncating to max_ctx_len."""
        token = torch.cat([z.flatten(-2, -1), action], dim=-1).unsqueeze(1)  # (B,1,token_in)
        ctx = torch.cat([ctx, token], dim=1)
        if ctx.shape[1] > self.max_ctx_len:
            ctx = ctx[:, -self.max_ctx_len:]
        return ctx

    def img_step(self, ctx, z, action):
        """Prior step (no observation). Returns (ctx_new, prior_logits)."""
        h = self._transformer_h(ctx) if ctx.shape[1] > 0 else self._zero_h(ctx.shape[0], ctx.device)
        prior_logits = self.prior_head(h).view(-1, self.stoch_size, self.stoch_classes)
        ctx_new = self._append_token(ctx, z, action)
        return ctx_new, prior_logits

    def obs_step(self, ctx, z, action, embed):
        """Posterior step (with observation). Returns (ctx_new, prior_logits, posterior_logits)."""
        h = self._transformer_h(ctx) if ctx.shape[1] > 0 else self._zero_h(ctx.shape[0], ctx.device)
        prior_logits = self.prior_head(h).view(-1, self.stoch_size, self.stoch_classes)
        posterior_logits = self.posterior_head(
            torch.cat([h, embed], dim=-1)
        ).view(-1, self.stoch_size, self.stoch_classes)
        ctx_new = self._append_token(ctx, z, action)
        return ctx_new, prior_logits, posterior_logits

    def observe(self, embeds, actions):
        """Process a full observed sequence. Returns (feats, prior_lgs, post_lgs)."""
        B, T, _ = embeds.shape
        ctx, z = self.initial_state(B, embeds.device)

        feats, prior_lgs, post_lgs = [], [], []
        for t in range(T):
            ctx, prior_lg, post_lg = self.obs_step(ctx, z, actions[:, t], embeds[:, t])
            z = straight_through_sample(post_lg)
            h = self._transformer_h(ctx)
            feats.append(self._feat(h, z))
            prior_lgs.append(prior_lg)
            post_lgs.append(post_lg)

        return (
            torch.stack(feats, dim=1),
            torch.stack(prior_lgs, dim=1),
            torch.stack(post_lgs, dim=1),
        )

    def imagine(self, init_feat, actor, horizon):
        """Unroll in imagination. Seeds with a single token from init_feat."""
        B = init_feat.shape[0]
        z = init_feat[:, self.h_dim:].view(B, self.stoch_size, self.stoch_classes)
        # Seed context with (z_seed, zero_action)
        zero_a = torch.zeros(B, self.action_dim, device=init_feat.device)
        ctx = self._append_token(
            torch.zeros(B, 0, self.stoch_flat + self.action_dim, device=init_feat.device),
            z, zero_a,
        )

        feats, acts = [], []
        for _ in range(horizon):
            h = self._transformer_h(ctx)
            feat = self._feat(h, z)
            action, _, _ = actor(feat)
            ctx, prior_logits = self.img_step(ctx, z, action)
            z = straight_through_sample(prior_logits)
            feats.append(feat)
            acts.append(action)

        terminal_feat = self._feat(self._transformer_h(ctx), z)
        return (
            torch.stack(feats, dim=1),
            torch.stack(acts, dim=1),
            terminal_feat,
        )
