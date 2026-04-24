"""
- Trains PPO using:
    image -> pretrained ResNet-18
    proprio -> direct input
    [visual features ; proprio] -> PPO
Usage:
    python baselines/ppo_resnet18_baseline.py configs/default.yaml
"""

import sys
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from sim.env import RoboticArmEnv
from models.resnet_proprio_extractor_ft import ResnetProprioExtractorFT


def train(config: dict):
    ppo_cfg = config["baselines"]["ppo_resnet18_ft"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        return RoboticArmEnv(config, moving_target=False)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(
        env
    )  # converts image branch to channel-first for SB3 handling
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=ResnetProprioExtractorFT,
        features_extractor_kwargs=dict(
            visual_dim=ppo_cfg["visual_dim"],
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
    )

    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        tb_log_name="resnet18_ppo_ft",
    )
    model.save("trained_models/ppo_resnet18_target_tracking_ft")
    env.close()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)
    train(config)
