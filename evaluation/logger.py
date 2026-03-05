"""
Experiment logger.

Records per-step and per-episode metrics to disk for later statistical
analysis and comparison between the world model and baseline agents.
"""


class Logger:
    def __init__(self, log_dir: str, run_name: str):
        raise NotImplementedError

    def log_step(self, step: int, data: dict):
        raise NotImplementedError

    def log_episode(self, episode: int, data: dict):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
