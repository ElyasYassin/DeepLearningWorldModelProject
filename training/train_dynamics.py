"""
Training script for the latent dynamics model.

Uses sequences of (z, a, z') tuples collected from the environment to
train the transformer dynamics model to predict future latent states.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import yaml, sys
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
