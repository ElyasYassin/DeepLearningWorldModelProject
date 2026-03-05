"""
Simulation environment wrapper.

Wraps the chosen physics engine (Isaac Sim / MuJoCo / Gazebo) behind a
Gym-compatible interface so the rest of the codebase stays engine-agnostic.
"""


class RoboticArmEnv:
    """Gym-style wrapper for the robotic arm tracking environment."""

    def __init__(self, config: dict):
        raise NotImplementedError

    def reset(self):
        """Reset the environment and return the initial RGB observation."""
        raise NotImplementedError

    def step(self, action):
        """Apply action; return (obs, reward, done, info)."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
