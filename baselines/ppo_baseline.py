"""
PPO baseline — model-free RL on raw pixel observations.

Uses stable-baselines3 PPO with CnnPolicy (no world model).
Serves as a comparison target against the Dreamer-style world model controller.

Usage:
    python baselines/ppo_baseline.py configs/default.yaml
"""

import sys
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecMonitor

from sim.env import RoboticArmEnv


def train(config: dict):
    ppo_cfg = config["baselines"]["ppo"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        return RoboticArmEnv(config, moving_target=False)

    # VecTransposeImage handles (H,W,C) → (C,H,W) for the image branch of MultiInputPolicy
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    model = PPO(
        "MultiInputPolicy",  # CNN for image branch + MLP for proprio branch
        env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        tb_log_name="ppo",
    )
    model.save("trained_models/ppo_target_tracking")
    env.close()


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
