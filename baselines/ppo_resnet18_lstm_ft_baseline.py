"""
Recurrent PPO baseline with a fine-tuned ResNet-18 visual backbone.

This combines:
    image -> pretrained ResNet-18 with selected layers unfrozen
    proprio -> direct input
    [visual features ; proprio] -> LSTM policy -> action

Usage:
    python baselines/ppo_resnet18_lstm_ft_baseline.py configs/default.yaml
"""

import sys

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from models.resnet_proprio_extractor_ft import ResnetProprioExtractorFT
from sim.env import RoboticArmEnv
from utils.config import load_config


def train(config: dict):
    cfg = config["baselines"]["ppo_resnet18_lstm_ft"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        return RoboticArmEnv(config, moving_target=False)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=ResnetProprioExtractorFT,
        features_extractor_kwargs=dict(
            visual_dim=cfg["visual_dim"],
            unfreeze_layers=cfg["unfreeze_layers"],
        ),
        lstm_hidden_size=cfg["lstm_hidden_size"],
        n_lstm_layers=cfg["n_lstm_layers"],
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(
            pi=cfg["policy_hidden_sizes"],
            vf=cfg["value_hidden_sizes"],
        ),
    )

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        tb_log_name=cfg["tb_log_name"],
    )
    model.save(cfg["save_path"])

    env.close()


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    train(config)
