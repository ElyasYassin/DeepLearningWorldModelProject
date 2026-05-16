"""
PPO baseline — model-free RL on raw pixel observations.

Uses stable-baselines3 PPO with CnnPolicy (no world model).
Serves as a comparison target against the Dreamer-style world model controller.

Usage:
    python -m baselines.ppo_baseline configs/experiments/ppo.yaml
"""

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecTransposeImage,
)

from sim.env import RoboticArmEnv
from utils.config import load_config


def train(config: dict):
    ppo_cfg = config["baselines"]["ppo"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        def _init():
            return RoboticArmEnv(config, moving_target=False)

        return _init

    n_envs = ppo_cfg.get("n_envs", 1)

    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])

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
        device=ppo_cfg.get("device", "auto"),
    )

    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        tb_log_name=ppo_cfg["tb_log_name"],
    )
    model.save(ppo_cfg["save_path"])

    env.close()


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    train(config)
