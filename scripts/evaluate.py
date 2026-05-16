"""
Environment evaluation for the DreamerV3 world model agent.

Runs N episodes in the real robosuite TargetTracking environment,
measuring task performance: episodic return, success rate,
mean EEF-to-target distance, and mean EEF jerk.

Jerk is computed as the L2 norm of the third finite difference of the
EEF position trajectory, normalised by dt³ (control_freq = 20 Hz → dt = 0.05 s).
Units: m/s³.

Usage:
    python scripts/evaluate.py configs/default.yaml
    python scripts/evaluate.py configs/default.yaml --episodes 20 --save-dir paper_figures/
    python scripts/evaluate.py configs/default.yaml --moving-target
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from models.world_model.world_model import WorldModel
from models.controller.actor_critic import DreamerActor
from models.world_model.rssm import TransformerRSSM
from models.world_model.utils import straight_through_sample
from sim.env import RoboticArmEnv

CONTROL_FREQ = 20  # Hz — must match env construction in sim/env.py
DT = 1.0 / CONTROL_FREQ


def _img_to_tensor(obs: dict, device) -> torch.Tensor:
    """obs['image'] (H,W,3 uint8) → (1,3,H,W) float in [0,1]."""
    return (
        torch.from_numpy(obs["image"])
        .permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)
    )


def _episode_jerk(eef_positions: list[np.ndarray]) -> float:
    """Mean L2 jerk (m/s³) over one episode.

    Requires ≥ 4 positions. Returns 0.0 for shorter episodes.
    """
    if len(eef_positions) < 4:
        return 0.0
    pos = np.stack(eef_positions)              # (T, 3)
    vel = np.diff(pos, axis=0) / DT           # (T-1, 3)
    acc = np.diff(vel, axis=0) / DT           # (T-2, 3)
    jerk = np.diff(acc, axis=0) / DT          # (T-3, 3)
    return float(np.linalg.norm(jerk, axis=1).mean())


def evaluate(config: dict, n_episodes: int, save_dir: str, moving_target: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dreamer_cfg = config["dreamer"]
    env_cfg = config["env"]

    use_moving = moving_target or env_cfg.get("moving_target", False)
    env = RoboticArmEnv(config, moving_target=use_moving)
    action_dim = env.action_space.shape[0]

    # ---- World model (auto-detect encoder type from checkpoint) ----
    ckpt = torch.load("checkpoints/dreamer_world_model.pt", map_location=device)
    encoder_type = "cnn" if any(k.startswith("encoder.cnn.") for k in ckpt) else "resnet"
    dynamics_type = dreamer_cfg.get("dynamics_type", "gru")
    print(f"Checkpoint encoder: {encoder_type}, dynamics: {dynamics_type}")

    world_model = WorldModel(
        action_dim=action_dim,
        image_size=env_cfg["image_size"],
        depth=dreamer_cfg["depth"],
        embed_dim=dreamer_cfg["embed_dim"],
        h_dim=dreamer_cfg["h_dim"],
        stoch_size=dreamer_cfg["stoch_size"],
        stoch_classes=dreamer_cfg["stoch_classes"],
        hidden_dim=dreamer_cfg["hidden_dim"],
        kl_scale=dreamer_cfg["kl_scale"],
        free_bits=dreamer_cfg["free_bits"],
        kl_balance=dreamer_cfg["kl_balance"],
        encoder_type=encoder_type,
        pretrained_encoder=dreamer_cfg.get("pretrained_encoder", True),
        freeze_encoder=dreamer_cfg.get("freeze_encoder", False),
        dynamics_type=dynamics_type,
        d_model=dreamer_cfg.get("d_model", 512),
        nhead=dreamer_cfg.get("nhead", 8),
        num_layers=dreamer_cfg.get("num_layers", 4),
        max_ctx_len=dreamer_cfg.get("max_ctx_len", 64),
    ).to(device)
    world_model.load_state_dict(ckpt)
    world_model.eval()

    feat_dim = world_model.feat_dim
    rssm = world_model.rssm
    is_transformer = isinstance(rssm, TransformerRSSM)

    # ---- Actor ----
    actor = DreamerActor(
        feat_dim=feat_dim,
        action_dim=action_dim,
        hidden_dim=dreamer_cfg["hidden_dim"],
    ).to(device)
    actor.load_state_dict(
        torch.load("checkpoints/dreamer_actor.pt", map_location=device)
    )
    actor.eval()

    # ---- Evaluation loop ----
    ep_returns = []
    ep_successes = []
    ep_eef_dists = []
    ep_jerks = []
    ep_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = rssm.initial_state(1, device)   # (h,z) or (ctx,z)
        prev_action = torch.zeros(1, action_dim, device=device)

        ep_return = 0.0
        step_eef_dist = []
        step_eef_pos = []
        ep_success = False

        while True:
            with torch.no_grad():
                img = _img_to_tensor(obs, device)
                embed = world_model.encoder(img)

                if is_transformer:
                    ctx, z = state
                    ctx, _prior, post_lgs = rssm.obs_step(ctx, z, prev_action, embed)
                    z = straight_through_sample(post_lgs)
                    h = rssm._transformer_h(ctx)
                    state = (ctx, z)
                else:
                    h, z = state
                    h, _prior, post_lgs = rssm.obs_step(h, z, prev_action, embed)
                    z = straight_through_sample(post_lgs)
                    state = (h, z)

                feat = rssm._feat(h, z)
                action_t = torch.tanh(actor.mu_head(actor.trunk(feat)))

            action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action_np)

            ep_return += reward
            step_eef_dist.append(info["eef_to_target"])
            step_eef_pos.append(info["eef_pos"])
            if info["success"]:
                ep_success = True

            prev_action = torch.from_numpy(action_np).unsqueeze(0).to(device)

            if terminated or truncated:
                break

        ep_jerk = _episode_jerk(step_eef_pos)
        ep_returns.append(ep_return)
        ep_successes.append(float(ep_success))
        ep_eef_dists.append(float(np.mean(step_eef_dist)))
        ep_jerks.append(ep_jerk)
        ep_lengths.append(len(step_eef_dist))

        print(
            f"Ep {ep+1:3d}/{n_episodes} | "
            f"return={ep_return:8.2f} | "
            f"success={str(ep_success):5s} | "
            f"eef={np.mean(step_eef_dist):.4f} m | "
            f"jerk={ep_jerk:.3f} m/s³ | "
            f"steps={len(step_eef_dist)}"
        )

    env.close()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  DreamerV3 Evaluation  ({n_episodes} episodes, moving={use_moving})")
    print(f"{'='*60}")
    print(f"  Mean return     : {np.mean(ep_returns):8.2f}  ±{np.std(ep_returns):.2f}")
    print(f"  Success rate    : {100*np.mean(ep_successes):5.1f}%  "
          f"({int(sum(ep_successes))}/{n_episodes})")
    print(f"  Mean EEF dist   : {np.mean(ep_eef_dists):.4f} m  ±{np.std(ep_eef_dists):.4f}")
    print(f"  Mean jerk       : {np.mean(ep_jerks):.3f} m/s³  ±{np.std(ep_jerks):.3f}")
    print(f"  Mean ep length  : {np.mean(ep_lengths):.1f} steps")
    print(f"{'='*60}\n")

    # ---- Plot ----
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))

    def _line_panel(ax, data, color, ylabel, title):
        ax.plot(data, marker="o", markersize=4, linewidth=1.2, color=color)
        ax.axhline(
            np.mean(data), color=color, linestyle="--", linewidth=1.2,
            label=f"mean = {np.mean(data):.3g}",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _line_panel(axes[0], ep_returns,  "steelblue", "Return",           "Episodic Return")
    _line_panel(axes[1], ep_eef_dists, "coral",    "EEF-to-Target (m)", "Mean EEF Distance")
    _line_panel(axes[2], ep_jerks,    "mediumpurple", "Jerk (m/s³)",   "Mean EEF Jerk")

    # Success / Failure bar
    success_pct = 100.0 * np.mean(ep_successes)
    fail_pct = 100.0 - success_pct
    bars = axes[3].bar(
        ["Success", "Failure"],
        [success_pct, fail_pct],
        color=["seagreen", "tomato"],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[3].set_ylabel("Episodes (%)")
    axes[3].set_title(f"Success Rate: {success_pct:.1f}%")
    axes[3].set_ylim(0, 105)
    for bar, val in zip(bars, [success_pct, fail_pct]):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", fontsize=10,
        )
    axes[3].grid(True, alpha=0.3, axis="y")

    condition = "moving target" if use_moving else "static target"
    fig.suptitle(
        f"DreamerV3 World Model — Environment Evaluation ({condition})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out_path = os.path.join(save_dir, "env_evaluation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--save-dir", default="paper_figures",
                        help="Output directory (default: paper_figures/)")
    parser.add_argument("--moving-target", action="store_true",
                        help="Enable moving target")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, args.episodes, args.save_dir, moving_target=args.moving_target)
