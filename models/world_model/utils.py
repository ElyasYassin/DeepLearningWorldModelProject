import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x.abs().exp() - 1)


def straight_through_sample(logits: torch.Tensor) -> torch.Tensor:
    """Straight-through categorical sample.

    logits: (*, stoch_size, stoch_classes)
    Returns one-hot tensor with same shape; gradient flows through soft probs.
    """
    probs = F.softmax(logits, dim=-1)
    indices = probs.argmax(dim=-1)
    one_hot = F.one_hot(indices, num_classes=logits.shape[-1]).to(probs.dtype)
    return one_hot + probs - probs.detach()


def make_bins(n_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    return torch.linspace(low, high, n_bins)


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar x as a two-hot distribution over bins.

    x: (*), bins: (n_bins,) on same device → (*, n_bins)
    """
    n = bins.shape[0]
    x_cl = x.clamp(bins[0].item(), bins[-1].item())
    below = torch.bucketize(x_cl.contiguous(), bins.contiguous()) - 1
    below = below.clamp(0, n - 2)
    above = below + 1

    b_lo = bins[below]
    b_hi = bins[above]
    w_hi = ((x_cl - b_lo) / (b_hi - b_lo).clamp(min=1e-8)).clamp(0.0, 1.0)
    w_lo = 1.0 - w_hi

    target = torch.zeros(*x.shape, n, device=x.device, dtype=x.dtype)
    target.scatter_(-1, below.unsqueeze(-1), w_lo.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_hi.unsqueeze(-1))
    return target


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode two-hot logits to scalar expectation.

    logits: (*, n_bins), bins: (n_bins,) → (*)
    """
    return (F.softmax(logits, dim=-1) * bins.to(logits.device)).sum(-1)


def kl_loss(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_bits: float = 1.0,
    balance: float = 0.8,
) -> torch.Tensor:
    """KL divergence loss between posterior and prior categorical distributions.

    Args:
        posterior_logits: (B, T, stoch_size, stoch_classes)
        prior_logits:     (B, T, stoch_size, stoch_classes)
        free_bits: minimum KL per categorical variable (nats)
        balance: weight for the posterior-gradient term (vs prior-gradient term)

    Returns scalar loss.
    """
    def _kl(p_logits, q_logits):
        # Use log_softmax directly — avoids softmax→log which produces -Inf for small probs
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # (B, T, stoch_size)
        return kl.clamp(min=free_bits).sum(-1)          # (B, T)

    # Gradient through posterior only (stop prior gradient)
    lhs = _kl(posterior_logits, prior_logits.detach())
    # Gradient through prior only (stop posterior gradient)
    rhs = _kl(posterior_logits.detach(), prior_logits)

    return (balance * lhs + (1.0 - balance) * rhs).mean()


def lambda_returns(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute lambda-return targets.

    rewards:   (B, H)
    continues: (B, H) — per-step continuation probability in [0, 1]
    values:    (B, H+1) — critic estimates; values[:, H] is the bootstrap
    Returns:   (B, H)
    """
    B, H = rewards.shape
    targets = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(H)):
        bootstrap = (1.0 - lam) * values[:, t + 1] + lam * last
        last = rewards[:, t] + gamma * continues[:, t] * bootstrap
        targets[:, t] = last
    return targets
