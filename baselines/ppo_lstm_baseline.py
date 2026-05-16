"""
Recurrent PPO baseline — LSTM policy on image + proprio observations.

Uses sb3-contrib RecurrentPPO with MultiInputLstmPolicy. The LSTM hidden state
persists across timesteps within an episode, allowing the policy to remember
where the target was last seen — key for the wrist-camera partial observability problem.

Compare against ppo_baseline.py (stateless CnnPolicy) to measure the benefit of memory.

Usage:
    python -m baselines.ppo_lstm_baseline configs/experiments/ppo_lstm.yaml
"""

import sys

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecTransposeImage,
)

from sim.env import RoboticArmEnv
from utils.config import load_config


def train(config: dict):
    lstm_cfg = config["baselines"]["ppo_lstm"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        def _init():
            return RoboticArmEnv(config, moving_target=False)

        return _init

    n_envs = lstm_cfg.get("n_envs", 1)

    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])

    env = VecTransposeImage(env)
    env = VecMonitor(env)

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        learning_rate=lstm_cfg["learning_rate"],
        n_steps=lstm_cfg["n_steps"],
        batch_size=lstm_cfg["batch_size"],
        n_epochs=lstm_cfg["n_epochs"],
        gamma=0.99,
        policy_kwargs=dict(
            lstm_hidden_size=lstm_cfg["lstm_hidden_size"],
            n_lstm_layers=lstm_cfg["n_lstm_layers"],
            shared_lstm=False,  # separate LSTMs for policy and value heads
            enable_critic_lstm=True,
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device=lstm_cfg.get("device", "auto"),
    )

    model.learn(
        total_timesteps=lstm_cfg["total_timesteps"],
        tb_log_name=lstm_cfg["tb_log_name"],
    )
    model.save(lstm_cfg["save_path"])
    env.close()


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    train(config)
