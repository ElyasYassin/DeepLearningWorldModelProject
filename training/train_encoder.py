"""
Training script for the VAE vision encoder.

Collects image observations from the simulation environment and trains the
VAE to reconstruct them, learning a compact latent representation.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import yaml, sys
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
