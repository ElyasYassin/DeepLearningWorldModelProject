"""
Visualization script for paper figures and presentation videos.

Renders the Panda arm in a high-resolution third-person view alongside the
wrist camera that the policy actually sees. Optionally loads a trained policy.

Usage:
    # Random actions (works before any training)
    python scripts/visualize.py configs/default.yaml

    # Trained policy
    python scripts/visualize.py configs/default.yaml --policy checkpoints/policy.pt

    # More episodes, custom output directory
    python scripts/visualize.py configs/default.yaml --episodes 3 --save-dir paper_figures/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

import robosuite

THIRD_PERSON_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
RENDER_SIZE = 512       # px per camera panel
GAP = 20                # px between panels
CAPTION_H = 44          # px for caption bar below panels
FPS = 20

# Camera pull-back: scale agentview position by this factor and widen FOV
CAM_DISTANCE_SCALE = 2.2   # 1.0 = default robosuite distance, >1 = further away
CAM_FOV_DEG = 65.0          # default agentview FOV is ~45°; wider = more scene visible


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _make_vis_env(config: dict):
    env_cfg = config["env"]
    return robosuite.make(
        "TargetTracking",
        robots=env_cfg["robot"],
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=[THIRD_PERSON_CAM, WRIST_CAM],
        camera_heights=RENDER_SIZE,
        camera_widths=RENDER_SIZE,
        camera_depths=False,
        moving_target=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        horizon=env_cfg["max_episode_steps"],
        reward_shaping=True,
        hard_reset=False,
    )


# ---------------------------------------------------------------------------
# Camera adjustment
# ---------------------------------------------------------------------------

def _adjust_camera(env, cam_name: str):
    """Pull the agentview camera back and widen its FOV for a more spacious view."""
    try:
        model = env.sim.model
        cam_id = model.camera_name2id(cam_name)
        # Scale position outward from the scene origin — preserves the angle
        model.cam_pos[cam_id] *= CAM_DISTANCE_SCALE
        model.cam_fovy[cam_id] = CAM_FOV_DEG
        print(f"Camera '{cam_name}' repositioned (scale={CAM_DISTANCE_SCALE}, fov={CAM_FOV_DEG}°)")
    except Exception as e:
        print(f"Warning: could not adjust camera — {e}")


# ---------------------------------------------------------------------------
# Policy — SB3 (.zip) or custom Dreamer (.pt)
# ---------------------------------------------------------------------------

def _is_sb3(path: str) -> bool:
    return path.endswith(".zip")


def _load_sb3(path: str):
    from stable_baselines3 import PPO
    model = PPO.load(path)
    print(f"Loaded SB3 PPO model from {path}")
    return model


def _sb3_obs(obs_dict: dict) -> dict:
    """Build the gym-style obs dict the SB3 model was trained with.

    The model was trained with RoboticArmEnv which exposes:
      "image"  — (64, 64, 3) uint8 wrist-cam
      "proprio" — float32 vector
    We resize the 512×512 wrist image back to 64×64 to match training.
    """
    from PIL import Image as _Image
    wrist = obs_dict[f"{WRIST_CAM}_image"]                        # (512, 512, 3) uint8
    wrist_64 = np.array(
        _Image.fromarray(wrist).resize((64, 64), _Image.BILINEAR)
    )
    parts = [obs_dict["robot0_proprio-state"].astype(np.float32)]
    if "target_pos" in obs_dict:
        parts.append(obs_dict["target_pos"].astype(np.float32))
    proprio = np.concatenate(parts)
    return {"image": wrist_64, "proprio": proprio}


def _sb3_action(obs_dict: dict, model) -> np.ndarray:
    obs = _sb3_obs(obs_dict)
    action, _ = model.predict(obs, deterministic=True)
    return action.astype(np.float32)


def _load_dreamer_policy(policy_path: str, config: dict, action_dim: int, device):
    from models.encoder.vae import VAE
    from models.controller.policy import Policy

    latent_dim = config["encoder"]["latent_dim"]
    hidden_dim = config["controller"]["hidden_dim"]

    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load("checkpoints/encoder.pt", map_location=device))
    vae.eval()

    policy = Policy(latent_dim, action_dim, hidden_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    return vae, policy


def _dreamer_action(obs_dict, vae, policy, device) -> np.ndarray:
    img = obs_dict[f"{WRIST_CAM}_image"].astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z = vae.encode(img_t)
        action = policy(z).squeeze(0).cpu().numpy()
    return action


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color=(220, 220, 220)):
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)


def _compose_panel(
    third_person: np.ndarray,
    wrist: np.ndarray,
    step: int,
    reward: float,
    dist: float,
) -> Image.Image:
    canvas_w = RENDER_SIZE * 2 + GAP
    canvas_h = RENDER_SIZE + CAPTION_H
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))

    canvas.paste(Image.fromarray(third_person), (0, 0))
    canvas.paste(Image.fromarray(wrist), (RENDER_SIZE + GAP, 0))

    draw = ImageDraw.Draw(canvas)

    # Panel labels
    _label(draw, 10, RENDER_SIZE + 10, "Third-person view")
    _label(draw, RENDER_SIZE + GAP + 10, RENDER_SIZE + 10, "Wrist camera (policy input)")

    # Step / reward / distance — right-aligned
    status = f"step {step:4d}   reward {reward:+.3f}   dist {dist:.4f} m"
    _label(draw, 10, RENDER_SIZE + 26, status, color=(160, 210, 160))

    # Thin separator line between panels
    draw.rectangle(
        [RENDER_SIZE, 0, RENDER_SIZE + GAP - 1, RENDER_SIZE - 1],
        fill=(18, 18, 18),
    )

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    config: dict,
    policy_path: Optional[str],
    n_episodes: int,
    save_dir: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_dir = os.path.join(save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Creating environment ({RENDER_SIZE}×{RENDER_SIZE} per panel)...")
    env = _make_vis_env(config)
    env.reset()                          # initialise sim so model is populated
    _adjust_camera(env, THIRD_PERSON_CAM)
    low, high = env.action_spec
    action_dim = low.shape[0]

    sb3_model = None
    vae, dreamer_policy = None, None

    if policy_path:
        print(f"Loading policy from {policy_path}...")
        if _is_sb3(policy_path):
            sb3_model = _load_sb3(policy_path)
        else:
            vae, dreamer_policy = _load_dreamer_policy(policy_path, config, action_dim, device)
            print("Using Dreamer policy.")
    else:
        print("No policy checkpoint — using random actions.")

    # Video writer
    video_path = os.path.join(save_dir, "visualization.mp4")
    canvas_w = RENDER_SIZE * 2 + GAP
    canvas_h = RENDER_SIZE + CAPTION_H
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, float(FPS), (canvas_w, canvas_h))
        has_video = True
    except ImportError:
        video_writer = None
        has_video = False
        print("cv2 not available — saving frames only (pip install opencv-python for video).")

    all_panels: list = []
    best_panel: Optional[Image.Image] = None   # clearest single frame for paper

    for ep in range(n_episodes):
        obs_dict = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        ep_panels: list = []

        print(f"\nEpisode {ep + 1}/{n_episodes}")

        while not done:
            if sb3_model is not None:
                action = _sb3_action(obs_dict, sb3_model)
            elif dreamer_policy is not None:
                action = _dreamer_action(obs_dict, vae, dreamer_policy, device)
            else:
                action = np.random.uniform(low, high).astype(np.float32)

            obs_dict, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            done = done or step >= config["env"]["max_episode_steps"]

            dist = float(np.linalg.norm(env._eef_to_target())) if hasattr(env, "_eef_to_target") else 0.0

            # MuJoCo offscreen renders are flipped vertically
            third_person = np.flipud(obs_dict[f"{THIRD_PERSON_CAM}_image"])
            wrist = np.flipud(obs_dict[f"{WRIST_CAM}_image"])

            panel = _compose_panel(third_person, wrist, step, reward, dist)
            ep_panels.append(panel)

            # Save one PNG every 10 steps for paper figures
            if step % 10 == 0:
                out = os.path.join(frames_dir, f"ep{ep+1:02d}_step{step:04d}.png")
                panel.save(out)

            if step % 50 == 0:
                print(f"  step {step:4d}  cumulative_r={total_reward:+.2f}  dist={dist:.4f}")

        print(f"  Episode {ep+1} finished — {step} steps, total reward {total_reward:.2f}")
        all_panels.extend(ep_panels)

        # Pick the midpoint frame from episode 1 as the poster
        if ep == 0:
            best_panel = ep_panels[len(ep_panels) // 2]

    # Write video
    if has_video and all_panels:
        import cv2
        for panel in all_panels:
            frame_bgr = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"\nVideo saved → {video_path}")

    # Clean poster frame: just the third-person view, no caption bar
    if best_panel is not None:
        poster_path = os.path.join(save_dir, "poster_frame.png")
        third_only = best_panel.crop((0, 0, RENDER_SIZE, RENDER_SIZE))
        third_only.save(poster_path)
        print(f"Poster frame saved → {poster_path}")

        side_by_side_path = os.path.join(save_dir, "side_by_side.png")
        best_panel.save(side_by_side_path)
        print(f"Side-by-side panel saved → {side_by_side_path}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the arm for paper figures / video.")
    parser.add_argument("config", help="Path to YAML config (e.g. configs/default.yaml)")
    parser.add_argument("--policy", default=None, metavar="CKPT",
                        help="Path to policy checkpoint (optional; random actions if omitted)")
    parser.add_argument("--episodes", type=int, default=1, metavar="N")
    parser.add_argument("--save-dir", default="paper_figures", metavar="DIR")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(config, args.policy, args.episodes, args.save_dir)
