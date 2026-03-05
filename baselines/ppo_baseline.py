"""
PPO baseline using stable-baselines3.

Trains a Proximal Policy Optimization agent directly on raw observations
(no world model) for comparison against the latent world model approach.
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    import yaml, sys
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
