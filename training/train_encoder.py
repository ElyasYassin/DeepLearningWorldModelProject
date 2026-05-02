"""
Training script for the VAE vision encoder.

Collects image observations from the simulation environment and trains the
VAE to reconstruct them, learning a compact latent representation.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import sys

    from utils.config import load_config

    config = load_config(sys.argv[1])
    train(config)
