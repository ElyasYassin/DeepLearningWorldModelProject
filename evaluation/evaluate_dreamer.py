"""
Evaluation script for the trained DreamerV3 world model + actor.

Loads checkpoints/dreamer_world_model.pt and checkpoints/dreamer_actor.pt,
runs N episodes in the real environment, and reports the same metrics as
evaluate.py so results are directly comparable to the PPO baselines.

Usage:
    python -m evaluation.evaluate_dreamer configs/default.yaml
    python -m evaluation.evaluate_dreamer configs/default.yaml --episodes 20 --render
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2

from models.world_model.world_model import WorldModel
from models.controller.actor_critic import DreamerActor
from models.world_model.utils import straight_through_sample
from sim.env import RoboticArmEnv
from evaluation.metrics import tracking_accuracy, fluidity
from utils.config import load_config


RENDER_CAMERAS = {
    ord("1"): "agentview",
    ord("2"): "birdview",
    ord("3"): "sideview",
    ord("4"): "robot0_eye_in_hand",
}
RENDER_SIZE = 512


def _get_render_frame(env: RoboticArmEnv, camera: str) -> np.ndarray:
    rgb = env._env.sim.render(camera_name=camera, height=RENDER_SIZE, width=RENDER_SIZE)[::-1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _deterministic_action(actor: DreamerActor, feat: torch.Tensor) -> torch.Tensor:
    """Return tanh(mean) — no sampling, for stable evaluation."""
    h = actor.trunk(feat)
    mu = actor.mu_head(h)
    return torch.tanh(mu)


def evaluate(config: dict, checkpoint_dir: str, n_episodes: int, render: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr_cfg = config["dreamer"]

    env = RoboticArmEnv(config)
    action_dim = env.action_space.shape[0]

    wm = WorldModel(
        action_dim=action_dim,
        image_size=config["env"]["image_size"],
        depth=dr_cfg["depth"],
        embed_dim=dr_cfg["embed_dim"],
        h_dim=dr_cfg["h_dim"],
        stoch_size=dr_cfg["stoch_size"],
        stoch_classes=dr_cfg["stoch_classes"],
        hidden_dim=dr_cfg["hidden_dim"],
        kl_scale=dr_cfg["kl_scale"],
        free_bits=dr_cfg["free_bits"],
        kl_balance=dr_cfg["kl_balance"],
        encoder_type=dr_cfg.get("encoder_type", "cnn"),
        pretrained_encoder=dr_cfg.get("pretrained_encoder", False),
        freeze_encoder=dr_cfg.get("freeze_encoder", False),
        dynamics_type=dr_cfg.get("dynamics_type", "gru"),
        d_model=dr_cfg.get("d_model", 512),
        nhead=dr_cfg.get("nhead", 8),
        num_layers=dr_cfg.get("num_layers", 4),
        max_ctx_len=dr_cfg.get("max_ctx_len", 64),
    ).to(device)
    wm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "world_model.pt"), map_location=device))
    wm.eval()

    actor = DreamerActor(wm.feat_dim, action_dim, dr_cfg["hidden_dim"]).to(device)
    actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, "actor.pt"), map_location=device))
    actor.eval()

    ep_rewards, ep_distances, ep_fluidity, ep_successes = [], [], [], []
    active_camera = "agentview"
    quit_early = False

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        state, z = wm.rssm.initial_state(1, device)
        prev_action = torch.zeros(1, action_dim, device=device)

        done = False
        total_reward = 0.0
        distances = []
        joint_positions = []

        while not done:
            img = (
                torch.from_numpy(obs_dict["image"])
                .permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)
            )

            with torch.no_grad():
                embed = wm.encoder(img)
                state, _, post_lg = wm.rssm.obs_step(state, z, prev_action, embed)
                z = straight_through_sample(post_lg)
                h = wm.rssm._transformer_h(state) if hasattr(wm.rssm, "_transformer_h") else state
                feat = wm.rssm._feat(h, z)
                action = _deterministic_action(actor, feat)  # (1, action_dim)

            action_np = action.squeeze(0).cpu().numpy()
            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_reward += reward
            distances.append(info["eef_to_target"])
            joint_positions.append(info["joint_positions"])
            prev_action = action.detach()

            if render:
                frame = _get_render_frame(env, active_camera)
                cv2.putText(
                    frame,
                    f"[{active_camera}]  1=agentview  2=birdview  3=sideview  4=wrist  q=quit",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow("DreamerV3 Eval", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    quit_early = True
                    done = True
                elif key in RENDER_CAMERAS:
                    active_camera = RENDER_CAMERAS[key]

        ep_rewards.append(total_reward)
        ep_distances.append(tracking_accuracy(distances))
        ep_successes.append(int(info["success"]))

        joint_traj = np.stack(joint_positions)
        if joint_traj.shape[0] > 3:
            ep_fluidity.append(fluidity(joint_traj, control_freq=20.0))

        print(
            f"Episode {ep + 1:>3}/{n_episodes} | "
            f"reward: {total_reward:7.2f} | "
            f"mean dist: {ep_distances[-1]:.4f} m | "
            f"success: {bool(info['success'])}"
        )

        if quit_early:
            break

    env.close()
    if render:
        cv2.destroyAllWindows()

    print("\n--- DreamerV3 Results ---")
    print(f"Episodes         : {len(ep_rewards)}")
    print(f"Mean reward      : {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")
    print(f"Success rate     : {np.mean(ep_successes) * 100:.1f}%")
    print(f"Tracking accuracy: {np.mean(ep_distances):.4f} m (mean dist to target)")
    if ep_fluidity:
        print(f"Fluidity (jerk²) : {np.mean(ep_fluidity):.6f}")

    return {
        "episodes": len(ep_rewards),
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "success_rate": float(np.mean(ep_successes)),
        "mean_distance": float(np.mean(ep_distances)),
        "mean_fluidity": float(np.mean(ep_fluidity)) if ep_fluidity else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--checkpoint", default="checkpoints/dreamer_1", help="Checkpoint directory")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.checkpoint, args.episodes, args.render)
