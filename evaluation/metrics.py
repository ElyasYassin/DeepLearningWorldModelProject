"""
Evaluation metrics for the tracking task.

Tracking Accuracy : average end-effector distance to target over an episode
Response Time      : latency between target movement and arm correction
Fluidity           : smoothness of joint trajectories (jerk minimization)
"""


def tracking_accuracy(end_effector_positions, target_positions) -> float:
    """Mean Euclidean distance between end effector and target across a trajectory."""
    raise NotImplementedError


def response_time(target_events, correction_events) -> float:
    """Average time (seconds) between a target move event and the arm's first correction."""
    raise NotImplementedError


def fluidity(joint_trajectories) -> float:
    """Mean squared jerk (third derivative of position) across all joints."""
    raise NotImplementedError
