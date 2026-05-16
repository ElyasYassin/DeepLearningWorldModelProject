"""
Evaluation metrics for the tracking task.

Tracking Accuracy : average end-effector distance to target over an episode
Response Time      : latency between target movement and arm correction
Fluidity           : smoothness of joint trajectories (jerk minimization)
"""

import numpy as np


def tracking_accuracy(distances: list[float]) -> float:
    """
    Mean Euclidean distance between end-effector and target across a trajectory.

    Args:
        distances: Per-step scalar distances (meters) from info["eef_to_target"].

    Returns:
        Mean distance in meters. Lower is better.
    """
    return float(np.mean(distances))


def response_time(target_events: list[float], correction_events: list[float]) -> float:
    """
    Average time (seconds) between a target move event and the arm's first correction.
    Only meaningful for the dynamic target task (stage 2+).

    Args:
        target_events:     Timestamps (seconds) when the target changed position.
        correction_events: Timestamps (seconds) of the arm's first measurable response
                           after each target move.

    Returns:
        Mean response latency in seconds. Lower is better.
    """
    assert len(target_events) == len(correction_events), (
        "Each target event must have a corresponding correction event."
    )
    latencies = [c - t for t, c in zip(target_events, correction_events)]
    return float(np.mean(latencies))


def fluidity(joint_trajectories: np.ndarray, control_freq: float = 20.0) -> float:
    """
    Mean squared jerk (third derivative of position) across all joints.

    Args:
        joint_trajectories: Array of shape (T, n_joints) — joint positions per timestep.
        control_freq:       Control frequency in Hz (default 20).

    Returns:
        Mean squared jerk across all joints and timesteps. Lower is better.
    """
    dt = 1.0 / control_freq
    velocity     = np.diff(joint_trajectories, n=1, axis=0) / dt
    acceleration = np.diff(velocity,           n=1, axis=0) / dt
    jerk         = np.diff(acceleration,       n=1, axis=0) / dt
    return float(np.mean(jerk ** 2))
