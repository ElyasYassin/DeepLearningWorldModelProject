"""
Simulation environment wrapper.

Wraps robosuite's TargetTracking environment (MuJoCo) behind a
gymnasium.Env-compatible interface so the rest of the codebase stays
engine-agnostic.

Observation is a dict with two keys:
  "image"  — (H, W, 3) uint8 wrist-camera RGB frame
  "proprio" — 1-D float32 vector: robot proprio-state + target position
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite
from robosuite.environments.manipulation.target_tracking import TargetTracking


class RoboticArmEnv(gym.Env):
    """
    Gym-style wrapper for the robosuite TargetTracking task.

    Observation (dict):
        "image":   (image_size, image_size, 3) uint8  — wrist-cam RGB
        "proprio": (N,) float32                       — joint states + EEF pose + target pos

    Action: continuous joint/EEF control from robosuite action_spec.

    Args:
        config (dict): Parsed default.yaml. Reads keys under ``env``.
        moving_target (bool): If True, target moves each step. Default False.
        render (bool): If True, display camera feed via cv2 when render() is called.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, moving_target: bool = False, render: bool = False):
        super().__init__()

        env_cfg = config["env"]
        image_size = env_cfg["image_size"]
        camera_name = env_cfg["camera_name"]

        self._camera_key = f"{camera_name}_image"
        self._render = render

        self._env: TargetTracking = robosuite.make(
            "TargetTracking",
            robots=env_cfg["robot"],
            use_camera_obs=True,
            use_object_obs=True,        # needed for target_pos in proprio
            camera_names=camera_name,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=False,
            moving_target=moving_target,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=env_cfg["max_episode_steps"],
            reward_shaping=True,
            hard_reset=False,
        )

        # Determine proprio dimension from a live reset
        _obs = self._env.reset()
        proprio = self._build_proprio(_obs)
        proprio_dim = proprio.shape[0]

        # Observation space: dict of image + proprio
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(proprio_dim,), dtype=np.float32),
        })

        # Action space: continuous, bounds from robosuite
        low, high = self._env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs_dict = self._env.reset()
        return self._extract_obs(obs_dict), {}

    def step(self, action):
        obs_dict, reward, terminated, info = self._env.step(action)
        obs = self._extract_obs(obs_dict)

        # Metric data for evaluation script
        info["eef_to_target"] = float(np.linalg.norm(self._env._eef_to_target()))
        info["joint_positions"] = self._env.sim.data.qpos[:7].copy()
        info["success"] = bool(self._env._check_success())

        return obs, float(reward), bool(terminated), False, info

    def render(self):
        pass  # rendering handled externally via obs["image"] in evaluate.py

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_proprio(self, obs_dict: dict) -> np.ndarray:
        """
        Build the proprioceptive vector from a robosuite obs dict.

        Concatenates:
          - robot0_proprio-state  (joint pos, vel, EEF pos, EEF quat, ...)
          - target_pos            (3D position of the target)
        """
        parts = [obs_dict["robot0_proprio-state"].astype(np.float32)]
        if "target_pos" in obs_dict:
            parts.append(obs_dict["target_pos"].astype(np.float32))
        return np.concatenate(parts)

    def _extract_obs(self, obs_dict: dict) -> dict:
        """Return the dict observation expected by the policy."""
        return {
            "image": obs_dict[self._camera_key].astype(np.uint8),
            "proprio": self._build_proprio(obs_dict),
        }
