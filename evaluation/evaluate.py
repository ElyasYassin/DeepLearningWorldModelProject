"""
Evaluation script for trained RL policies.

Loads a saved model, runs N episodes in the environment, and reports
tracking accuracy, fluidity, episode reward, and success rate.

Usage:
    python evaluation/evaluate.py configs/default.yaml --model ppo_target_tracking
    python evaluation/evaluate.py configs/default.yaml --model ppo_target_tracking --episodes 20 --render
"""

import argparse
import sys
import yaml
import numpy as np
import cv2

from stable_baselines3 import PPO

from sim.env import RoboticArmEnv
from evaluation.metrics import tracking_accuracy, fluidity


def evaluate(config: dict, model_path: str | None, n_episodes: int, render: bool = False):
    env = RoboticArmEnv(config, render=render)
    model = None if model_path is None else PPO.load(model_path, env=env)

    ep_rewards = []
    ep_distances = []    # tracking accuracy per episode
    ep_fluidity = []     # fluidity per episode
    ep_successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        distances = []
        joint_positions = []

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            distances.append(info["eef_to_target"])
            joint_positions.append(info["joint_positions"])

            if render:
                # Upscale the 64×64 wrist-cam image for visibility and display
                frame = cv2.resize(obs["image"], (512, 512), interpolation=cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("TargetTracking — wrist cam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        ep_rewards.append(total_reward)
        ep_distances.append(tracking_accuracy(distances))
        ep_successes.append(int(info["success"]))

        joint_traj = np.stack(joint_positions)  # (T, 7)
        if joint_traj.shape[0] > 3:             # need at least 4 steps for jerk
            ep_fluidity.append(fluidity(joint_traj, control_freq=20.0))

        print(
            f"Episode {ep + 1:>3}/{n_episodes} | "
            f"reward: {total_reward:7.2f} | "
            f"mean dist: {ep_distances[-1]:.4f} m | "
            f"success: {bool(info['success'])}"
        )

    env.close()
    if render:
        cv2.destroyAllWindows()

    print("\n--- Results ---")
    print(f"Episodes         : {n_episodes}")
    print(f"Mean reward      : {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")
    print(f"Success rate     : {np.mean(ep_successes) * 100:.1f}%")
    print(f"Tracking accuracy: {np.mean(ep_distances):.4f} m (mean dist to target)")
    if ep_fluidity:
        print(f"Fluidity (jerk²) : {np.mean(ep_fluidity):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--model", default=None, help="Path to saved model (no .zip). Omit to use random policy.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, args.model, args.episodes, args.render)
