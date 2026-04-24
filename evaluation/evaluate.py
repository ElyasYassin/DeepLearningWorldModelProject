"""
Evaluation script for trained RL policies.

Loads a saved model, runs N episodes in the environment, and reports
tracking accuracy, fluidity, episode reward, and success rate.

Usage:
    python evaluation/evaluate.py configs/default.yaml --model ppo_target_tracking
    python evaluation/evaluate.py configs/default.yaml --model ppo_target_tracking --episodes 20 --render

Render controls (when --render is active):
    1  →  agentview          (front-diagonal, full workspace)
    2  →  birdview           (overhead)
    3  →  sideview           (side angle)
    4  →  robot0_eye_in_hand (wrist cam — what the policy sees)
    q  →  quit early
"""

import argparse
import yaml
import numpy as np
import cv2

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from sim.env import RoboticArmEnv
from evaluation.metrics import tracking_accuracy, fluidity


def load_model(model_path: str, env):
    """Load a PPO or RecurrentPPO model, detecting type from the zip."""
    try:
        return RecurrentPPO.load(model_path, env=env), True
    except Exception:
        return PPO.load(model_path, env=env), False


RENDER_CAMERAS = {
    ord("1"): "agentview",
    ord("2"): "birdview",
    ord("3"): "sideview",
    ord("4"): "robot0_eye_in_hand",
}
RENDER_SIZE = 512


def _get_render_frame(env: RoboticArmEnv, camera: str) -> np.ndarray:
    """Fetch a BGR frame from any robosuite camera for display."""
    # robosuite renders upside-down → flip vertically
    rgb = env._env.sim.render(
        camera_name=camera, height=RENDER_SIZE, width=RENDER_SIZE
    )[::-1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def evaluate(
    config: dict, model_path: str | None, n_episodes: int, render: bool = False
):
    env = RoboticArmEnv(config)
    is_recurrent = False
    if model_path is None:
        model = None
    else:
        model, is_recurrent = load_model(model_path, env)

    ep_rewards = []
    ep_distances = []
    ep_fluidity = []
    ep_successes = []

    active_camera = "agentview"  # default render view
    quit_early = False

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        distances = []
        joint_positions = []
        lstm_states = None  # only used by RecurrentPPO
        episode_start = True

        while not done:
            if model is None:
                action = env.action_space.sample()
            elif is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            distances.append(info["eef_to_target"])
            joint_positions.append(info["joint_positions"])

            if render:
                frame = _get_render_frame(env, active_camera)
                cv2.putText(
                    frame,
                    f"[{active_camera}]  1=agentview  2=birdview  3=sideview  4=wrist  q=quit",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("TargetTracking", frame)
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

    print("\n--- Results ---")
    print(f"Episodes         : {len(ep_rewards)}")
    print(f"Mean reward      : {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")
    print(f"Success rate     : {np.mean(ep_successes) * 100:.1f}%")
    print(f"Tracking accuracy: {np.mean(ep_distances):.4f} m (mean dist to target)")
    if ep_fluidity:
        print(f"Fluidity (jerk²) : {np.mean(ep_fluidity):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--model", default=None, help="Path to saved model. Omit to use random policy."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, args.model, args.episodes, args.render)
