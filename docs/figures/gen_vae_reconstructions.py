"""
Generates Figure 2: VAE reconstruction examples.
Output: docs/figures/vae_reconstructions.pdf

Run from project root:
    python docs/figures/gen_vae_reconstructions.py configs/default.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.encoder.vae import VAE
from sim.env import RoboticArmEnv

OUT = os.path.join(os.path.dirname(__file__), "vae_reconstructions.pdf")
N_FRAMES = 3   # number of columns in the grid


def collect_diverse_frames(env, n=N_FRAMES, max_episodes=10):
    """Collect N frames spread across episode time to get visual diversity."""
    frames = []
    for _ in range(max_episodes):
        obs, _ = env.reset()
        done = False
        ep_frames = []
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_frames.append(obs["image"].copy())   # (64, 64, 3) uint8
        # Sample frames spread across the episode
        if len(ep_frames) >= n:
            idxs = np.linspace(0, len(ep_frames) - 1, n, dtype=int)
            frames.extend([ep_frames[i] for i in idxs])
        if len(frames) >= n:
            break
    return frames[:n]


def main(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    latent_dim = config["encoder"]["latent_dim"]
    ckpt = "checkpoints/encoder.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt}")

    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load(ckpt, map_location=device))
    vae.eval()

    env = RoboticArmEnv(config)
    print("Collecting frames from environment...")
    raw_frames = collect_diverse_frames(env, n=N_FRAMES)
    env.close()

    # Encode then decode each frame
    reconstructions = []
    with torch.no_grad():
        for frame in raw_frames:
            img = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            ).to(device)
            z = vae.encode(img)                  # (1, latent_dim) — deterministic mu
            recon = vae.decoder(z).squeeze(0)    # (3, 64, 64) in [0, 1]
            reconstructions.append(recon.permute(1, 2, 0).cpu().numpy())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, N_FRAMES,
        figsize=(N_FRAMES * 1.8, 2 * 1.8 + 0.4),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )

    row_labels = ["Input", "Reconstruction"]
    all_imgs = [raw_frames, reconstructions]

    for row, (label, imgs) in enumerate(zip(row_labels, all_imgs)):
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            if row == 0:
                # uint8 input
                ax.imshow(img)
            else:
                # float [0,1] reconstruction
                ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold",
                              rotation=0, labelpad=48, va="center")

    fig.suptitle("VAE Reconstruction Examples (wrist-camera frames)",
                 fontsize=9, y=1.02)

    plt.savefig(OUT, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)
