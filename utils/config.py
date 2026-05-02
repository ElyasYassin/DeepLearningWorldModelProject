from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    base_config = config.pop("base_config", None)
    if base_config is None:
        return config

    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path

    return _deep_merge(load_config(base_path), config)
