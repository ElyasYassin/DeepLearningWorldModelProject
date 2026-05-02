"""
Training script for the latent dynamics model.

Uses sequences of (z, a, z') tuples collected from the environment to
train the transformer dynamics model to predict future latent states.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    from utils.config import load_config

    config = load_config(sys.argv[1])
    train(config)
