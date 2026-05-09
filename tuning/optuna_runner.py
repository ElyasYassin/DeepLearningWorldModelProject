import argparse
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

import optuna

from evaluation.evaluate import evaluate
from utils.config import load_config


def _suggest_value(trial: optuna.Trial, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    if isinstance(spec, dict):
        if {"low", "high"}.issubset(spec):
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        return trial.suggest_categorical(name, list(spec.keys()))

    raise TypeError(f"Unsupported search space for {name}: {spec!r}")


def _resolve_value(name: str, selected: Any, spec: Any) -> Any:
    if isinstance(spec, dict) and "low" not in spec and "high" not in spec:
        return spec[selected]
    return selected


def _format_param(value: Any) -> str:
    if isinstance(value, float):
        value = f"{value:.2e}"
    if isinstance(value, list):
        value = "_".join(str(item) for item in value)
    return str(value).replace("+", "").replace(".", "p").replace(" ", "")


def _trial_name(trial: optuna.Trial, selected_params: dict[str, Any]) -> str:
    parts = [f"trial{trial.number:03d}"]
    for key, value in selected_params.items():
        parts.append(f"{key}{_format_param(value)}")
    return "_".join(parts)


def _model_zip_path(save_path: str) -> str:
    if save_path.endswith(".zip"):
        return save_path
    return f"{save_path}.zip"


def _score(metrics: dict) -> tuple[float, float]:
    if metrics["mean_fluidity"] is None:
        raise ValueError("mean_fluidity is required for multi-objective tuning")
    return metrics["mean_reward"], metrics["mean_fluidity"]


def _load_train_function(train_module: str):
    module = import_module(train_module)
    return module.train


def _suggest_config(trial: optuna.Trial, config: dict) -> dict:
    trial_config = deepcopy(config)
    optuna_cfg = trial_config["optuna"]
    baseline_key = optuna_cfg["baseline_key"]
    search_space = optuna_cfg["search_space"]
    baseline_cfg = trial_config["baselines"][baseline_key]
    selected_params = {}

    for name, spec in search_space.items():
        selected = _suggest_value(trial, name, spec)
        value = _resolve_value(name, selected, spec)
        baseline_cfg[name] = value
        selected_params[name] = selected

    baseline_cfg["total_timesteps"] = optuna_cfg["train_timesteps"]

    prefix = optuna_cfg.get("artifact_prefix", optuna_cfg["study_name"])
    trial_name = _trial_name(trial, selected_params)
    baseline_cfg["tb_log_name"] = f"{prefix}_{trial_name}"
    baseline_cfg["save_path"] = f"trained_models/{prefix}_{trial_name}"
    return trial_config


def objective(trial: optuna.Trial, config: dict) -> tuple[float, float]:
    trial_config = _suggest_config(trial, config)
    optuna_cfg = trial_config["optuna"]
    baseline_key = optuna_cfg["baseline_key"]
    baseline_cfg = trial_config["baselines"][baseline_key]
    train = _load_train_function(optuna_cfg["train_module"])

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

    return _score(metrics)


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
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, config), n_trials=optuna_cfg["n_trials"]
    )

    print("\n--- Pareto Front ---")
    for best_trial in study.best_trials:
        print(f"\nTrial {best_trial.number}")
        print(f"Values: {best_trial.values}")
        print(f"Params: {best_trial.params}")
        print(f"Attrs : {best_trial.user_attrs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to Optuna YAML config")
    args = parser.parse_args()
    main(args.config)
