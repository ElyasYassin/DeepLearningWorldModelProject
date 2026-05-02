"""
SAC baseline using stable-baselines3.

Trains a Soft Actor-Critic agent directly on raw observations
(no world model) for comparison against the latent world model approach.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    from utils.config import load_config

    config = load_config(sys.argv[1])
    train(config)
