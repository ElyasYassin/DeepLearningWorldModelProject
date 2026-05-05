import argparse
from copy import deepcopy
from pathlib import Path

import optuna

from baselines.ppo_resnet18_lstm_ft_baseline import train
from evaluation.evaluate import evaluate
from utils.config import load_config

BASELINE_KEY = "ppo_resnet18_lstm_ft"


def _suggest_config(trial: optuna.Trial, config: dict) -> dict:
    trial_config = deepcopy(config)
    optuna_cfg = trial_config["optuna"]
    search_space = optuna_cfg["search_space"]
    baseline_cfg = trial_config["baselines"][BASELINE_KEY]

    lr_cfg = search_space["learning_rate"]
    learning_rate = trial.suggest_float(
        "learning_rate",
        lr_cfg["low"],
        lr_cfg["high"],
        log=lr_cfg.get("log", False),
    )
    n_steps = trial.suggest_categorical("n_steps", search_space["n_steps"])
    lstm_hidden_size = trial.suggest_categorical(
        "lstm_hidden_size", search_space["lstm_hidden_size"]
    )
    unfreeze_name = trial.suggest_categorical(
        "unfreeze_layers", list(search_space["unfreeze_layers"].keys())
    )
    unfreeze_layers = search_space["unfreeze_layers"][unfreeze_name]

    baseline_cfg["learning_rate"] = learning_rate
    baseline_cfg["n_steps"] = n_steps
    baseline_cfg["lstm_hidden_size"] = lstm_hidden_size
    baseline_cfg["unfreeze_layers"] = unfreeze_layers
    baseline_cfg["total_timesteps"] = optuna_cfg["train_timesteps"]

    trial_name = (
        f"trial{trial.number:03d}_"
        f"lr{learning_rate:.2e}_"
        f"n{n_steps}_"
        f"lstm{lstm_hidden_size}_"
        f"{unfreeze_name}"
    ).replace("+", "")

    trial_name = trial_name.replace(".", "p")

    baseline_cfg["tb_log_name"] = f"optuna_resnet18_lstm_ft_{trial_name}"
    baseline_cfg["save_path"] = f"trained_models/optuna_resnet18_lstm_ft_{trial_name}"
    return trial_config


def _model_zip_path(save_path: str) -> str:
    if save_path.endswith(".zip"):
        return save_path
    return f"{save_path}.zip"


def _score(metrics: dict, objective_metric: str) -> float:
    if objective_metric == "mean_reward":
        return metrics["mean_reward"]
    if objective_metric == "success_rate":
        return metrics["success_rate"]
    if objective_metric == "negative_mean_distance":
        return -metrics["mean_distance"]
    raise ValueError(f"Unsupported objective metric: {objective_metric}")


def objective(trial: optuna.Trial, config: dict) -> float:
    trial_config = _suggest_config(trial, config)
    optuna_cfg = trial_config["optuna"]
    baseline_cfg = trial_config["baselines"][BASELINE_KEY]

    train(trial_config)
    metrics = evaluate(
        trial_config,
        model_path=_model_zip_path(baseline_cfg["save_path"]),
        n_episodes=optuna_cfg["eval_episodes"],
        render=False,
    )

    for key, value in metrics.items():
        if value is not None:
            trial.set_user_attr(key, value)

    return _score(metrics, optuna_cfg["objective_metric"])


def main(config_path: str):
    config = load_config(config_path)
    optuna_cfg = config["optuna"]

    storage = optuna_cfg.get("storage")
    if storage and storage.startswith("sqlite:///"):
        db_path = Path(storage.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=optuna_cfg["study_name"],
        storage=storage,
        direction=optuna_cfg["direction"],
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, config), n_trials=optuna_cfg["n_trials"]
    )

    print("\n--- Best Trial ---")
    print(f"Value : {study.best_trial.value}")
    print(f"Params: {study.best_trial.params}")
    print(f"Attrs : {study.best_trial.user_attrs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        default="configs/tuning/optuna_lstm_ft.yaml",
        nargs="?",
        help="Path to Optuna YAML config",
    )
    args = parser.parse_args()
    main(args.config)
