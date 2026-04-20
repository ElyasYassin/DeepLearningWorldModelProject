"""
Recurrent PPO baseline — LSTM policy on image + proprio observations.

Uses sb3-contrib RecurrentPPO with MultiInputLstmPolicy. The LSTM hidden state
persists across timesteps within an episode, allowing the policy to remember
where the target was last seen — key for the wrist-camera partial observability problem.

Compare against ppo_baseline.py (stateless CnnPolicy) to measure the benefit of memory.

Usage:
    python baselines/ppo_lstm_baseline.py configs/default.yaml
"""

import sys
import yaml

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from sim.env import RoboticArmEnv


def train(config: dict):
    lstm_cfg = config["baselines"]["ppo_lstm"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        return RoboticArmEnv(config, moving_target=False)

    # VecTransposeImage: (H,W,C) → (C,H,W) for the CNN image branch
    env = DummyVecEnv([make_env])
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
    )

    model.learn(
        total_timesteps=lstm_cfg["total_timesteps"],
        tb_log_name="lstm_ppo",
    )
    model.save("trained_models/ppo_lstm_target_tracking")
    env.close()


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
