"""
Open-loop dynamics verification.

Runs one real episode, then replays the same actions through the dynamics
model starting from z_0 only (no real frames after step 0). Decodes both
real and predicted latents and saves a comparison grid + latent MSE plot.

Usage:
    python scripts/verify_dynamics.py configs/default.yaml
    python scripts/verify_dynamics.py configs/default.yaml --steps 30 --save-dir paper_figures/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

from models.encoder.vae import VAE
from models.dynamics.transformer import LatentDynamicsModel
from sim.env import RoboticArmEnv


# ---------------------------------------------------------------------------

def _encode_frame(obs_dict: dict, vae: VAE, device) -> torch.Tensor:
    img = (
        torch.from_numpy(obs_dict["image"])
        .permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)
    )
    with torch.no_grad():
        return vae.encode(img)  # (1, latent_dim)


def _decode_latent(z: torch.Tensor, vae: VAE) -> np.ndarray:
    with torch.no_grad():
        recon = vae.decoder(z)  # (1, 3, 64, 64)
    return recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ---------------------------------------------------------------------------

def verify(config: dict, n_steps: int, save_dir: str, moving_target: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = config["encoder"]
    dyn_cfg = config["dynamics"]
    latent_dim = enc_cfg["latent_dim"]

    # Load checkpoints
    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load("checkpoints/encoder.pt", map_location=device))
    vae.eval()

    use_moving = moving_target or config["env"].get("moving_target", False)
    env = RoboticArmEnv(config, moving_target=use_moving)
    action_dim = env.action_space.shape[0]

    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        d_model=dyn_cfg["d_model"],
        nhead=dyn_cfg["nhead"],
        num_layers=dyn_cfg["num_layers"],
        max_seq_len=dyn_cfg["max_seq_len"],
    ).to(device)
    dynamics.load_state_dict(torch.load("checkpoints/dynamics.pt", map_location=device))
    dynamics.eval()

    # --- Collect real trajectory ---
    obs_dict, _ = env.reset()
    real_frames, real_latents, actions = [], [], []

    for _ in range(n_steps):
        z = _encode_frame(obs_dict, vae, device)
        img_decoded = _decode_latent(z, vae)

        action = env.action_space.sample()
        obs_dict, _, terminated, truncated, _ = env.step(action)

        real_frames.append(img_decoded)
        real_latents.append(z.squeeze(0))
        actions.append(torch.from_numpy(action.astype(np.float32)).to(device))

        if terminated or truncated:
            break

    env.close()
    T = len(real_frames)
    print(f"Collected {T} real steps.")

    # --- Open-loop dynamics rollout ---
    # Start from z_0, predict forward using only the real actions (no real frames)
    ctx_z = real_latents[0].unsqueeze(0).unsqueeze(0)       # (1, 1, latent_dim)
    ctx_a = torch.zeros(1, 1, action_dim, device=device)

    pred_latents = [real_latents[0]]  # step 0 is real by definition
    pred_frames  = [real_frames[0]]

    z_t = real_latents[0].unsqueeze(0)                      # (1, latent_dim)
    for t in range(T - 1):
        a_t = actions[t].unsqueeze(0).unsqueeze(0)          # (1, 1, action_dim)

        ctx_z = torch.cat([ctx_z, z_t.unsqueeze(1)], dim=1)
        ctx_a = torch.cat([ctx_a, a_t], dim=1)

        with torch.no_grad():
            z_next = dynamics.step(ctx_z, ctx_a)            # (1, latent_dim)

        pred_latents.append(z_next.squeeze(0))
        pred_frames.append(_decode_latent(z_next, vae))
        z_t = z_next

    # --- Latent MSE over time ---
    mse_per_step = []
    for t in range(T):
        diff = (real_latents[t] - pred_latents[t]).pow(2).mean().item()
        mse_per_step.append(diff)

    # --- Build figure ---
    os.makedirs(save_dir, exist_ok=True)

    # Pick evenly spaced display steps
    display_steps = np.linspace(0, T - 1, min(10, T), dtype=int).tolist()
    n_cols = len(display_steps)

    fig = plt.figure(figsize=(n_cols * 1.8, 6))
    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.08, wspace=0.04)

    for col, t in enumerate(display_steps):
        # Real frame
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(real_frames[t])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Real", fontsize=9, labelpad=4)
        ax.set_title(f"t={t}", fontsize=7)

        # Predicted frame
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pred_frames[t])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Predicted", fontsize=9, labelpad=4)

    # MSE plot spanning full width
    ax_mse = fig.add_subplot(gs[2, :])
    ax_mse.plot(mse_per_step, color="steelblue", linewidth=1.5)
    ax_mse.set_xlabel("Step", fontsize=9)
    ax_mse.set_ylabel("Latent MSE", fontsize=9)
    ax_mse.set_title("Open-loop prediction error over time", fontsize=9)
    ax_mse.tick_params(labelsize=8)
    ax_mse.grid(True, alpha=0.3)
    # Mark the display steps
    for t in display_steps:
        ax_mse.axvline(t, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    out_path = os.path.join(save_dir, "dynamics_verification.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print summary
    print(f"\nLatent MSE summary:")
    print(f"  Step  0 (t=0):    {mse_per_step[0]:.5f}  (reference)")
    mid = T // 2
    print(f"  Step {mid:2d} (t={mid}): {mse_per_step[mid]:.5f}")
    print(f"  Step {T-1:2d} (t={T-1}): {mse_per_step[-1]:.5f}")
    print(f"  Mean MSE:          {np.mean(mse_per_step):.5f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=25,
                        help="Number of steps to roll out (default: 25)")
    parser.add_argument("--save-dir", default="paper_figures",
                        help="Output directory (default: paper_figures/)")
    parser.add_argument("--moving-target", action="store_true",
                        help="Enable moving target (dynamic tracking phase)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    verify(config, args.steps, args.save_dir, moving_target=args.moving_target)
