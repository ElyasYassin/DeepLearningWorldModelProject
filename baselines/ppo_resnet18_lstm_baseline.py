import sys
from sb3_contrib import RecurrentPPO
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
    cfg = config["baselines"]["ppo_resnet18_lstm"]
    log_dir = config["evaluation"]["log_dir"]

    def make_env():
        def _init():
            return RoboticArmEnv(config, moving_target=False)

        return _init

    n_envs = cfg.get("n_envs", 1)

    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])

    env = VecTransposeImage(env)
    env = VecMonitor(env)
    policy_kwargs = dict(
        features_extractor_class=ResnetProprioExtractor,
        features_extractor_kwargs=dict(
            visual_dim=cfg["visual_dim"],
            freeze_backbone=cfg["freeze_backbone"],
        ),
        lstm_hidden_size=cfg["lstm_hidden_size"],
        n_lstm_layers=cfg["n_lstm_layers"],
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
        device=cfg.get("device", "auto"),
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
