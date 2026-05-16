"""
- Trains PPO using:
    image -> pretrained ResNet-18
    proprio -> direct input
    [visual features ; proprio] -> PPO
Usage:
    python baselines/ppo_resnet18_baseline.py configs/default.yaml
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
from models.resnet_proprio_extractor import ResnetProprioExtractor
from utils.config import load_config


def train(config: dict):
    ppo_cfg = config["baselines"]["ppo_resnet18"]
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

    policy_kwargs = dict(
        features_extractor_class=ResnetProprioExtractor,
        features_extractor_kwargs=dict(
            visual_dim=ppo_cfg["visual_dim"],
            freeze_backbone=ppo_cfg["freeze_backbone"],
        ),
        net_arch=dict(
            pi=ppo_cfg["policy_hidden_sizes"],
            vf=ppo_cfg["value_hidden_sizes"],
        ),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        policy_kwargs=policy_kwargs,
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
