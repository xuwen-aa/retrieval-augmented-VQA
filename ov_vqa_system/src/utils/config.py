"""
Configuration utilities for OV-VQA experiments.

This module loads a default YAML config and merges it with
dataset-specific, baseline-specific, or ablation-specific configs.

Example:
    cfg = load_config(
        config_path="configs/okvqa.yaml",
        default_path="configs/default.yaml",
        cli_overrides={
            "experiment.output_dir": "outputs/okvqa/ours_full",
            "data.limit": 50,
        },
    )
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ConfigDict = Dict[str, Any]


def load_yaml(path: str | os.PathLike) -> ConfigDict:
    """
    Load a YAML file as a Python dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file is empty or invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML dictionary: {path}")

    return data


def deep_update(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """
    Recursively update a nested dictionary.

    Values in override will replace values in base.
    Nested dictionaries are merged recursively.

    Args:
        base: Base dictionary.
        override: Dictionary containing override values.

    Returns:
        Updated dictionary.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def set_by_dot_key(cfg: ConfigDict, dot_key: str, value: Any) -> None:
    """
    Set a nested config value using a dot-separated key.

    Example:
        set_by_dot_key(cfg, "detector.conf_threshold", 0.3)

    Args:
        cfg: Config dictionary to update in place.
        dot_key: Dot-separated key.
        value: New value.
    """
    keys = dot_key.split(".")
    current = cfg

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def get_by_dot_key(cfg: ConfigDict, dot_key: str, default: Any = None) -> Any:
    """
    Get a nested config value using a dot-separated key.

    Example:
        value = get_by_dot_key(cfg, "detector.model_path")

    Args:
        cfg: Config dictionary.
        dot_key: Dot-separated key.
        default: Returned when the key does not exist.

    Returns:
        Config value or default.
    """
    keys = dot_key.split(".")
    current: Any = cfg

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def parse_override_value(value: str) -> Any:
    """
    Convert a command-line override string to a Python value.

    Examples:
        "true" -> True
        "false" -> False
        "null" -> None
        "42" -> 42
        "0.25" -> 0.25
        "abc" -> "abc"

    Args:
        value: String value from command line.

    Returns:
        Parsed Python value.
    """
    lower = value.lower()

    if lower in {"true", "yes"}:
        return True

    if lower in {"false", "no"}:
        return False

    if lower in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def parse_cli_overrides(overrides: Optional[list[str]]) -> ConfigDict:
    """
    Parse command-line overrides.

    Expected format:
        ["detector.conf_threshold=0.3", "retrieval.enabled=false"]

    Args:
        overrides: List of key=value strings.

    Returns:
        Flat dictionary where keys are dot-separated config paths.
    """
    parsed: ConfigDict = {}

    if not overrides:
        return parsed

    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override: {item}. Expected format: key=value"
            )

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid override key in: {item}")

        parsed[key] = parse_override_value(value)

    return parsed


def apply_cli_overrides(cfg: ConfigDict, overrides: Optional[Dict[str, Any]]) -> ConfigDict:
    """
    Apply dot-key overrides to a config dictionary.

    Args:
        cfg: Original config.
        overrides: Mapping from dot-key to value.

    Returns:
        New config dictionary.
    """
    result = copy.deepcopy(cfg)

    if not overrides:
        return result

    for dot_key, value in overrides.items():
        set_by_dot_key(result, dot_key, value)

    return result


def validate_config(cfg: ConfigDict) -> None:
    """
    Validate critical config fields.

    This function only checks fields that are necessary
    for the main experiment to run.

    Args:
        cfg: Final merged config.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required_keys = [
        "experiment.name",
        "experiment.output_dir",
        "data.dataset_name",
        "data.image_root",
        "detector.model_path",
        "llm.provider",
        "llm.model_name",
    ]

    missing = []

    for key in required_keys:
        value = get_by_dot_key(cfg, key)
        if value is None:
            missing.append(key)

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if get_by_dot_key(cfg, "detector.max_prompts", 0) <= 0:
        raise ValueError("detector.max_prompts must be positive.")

    if get_by_dot_key(cfg, "retrieval.top_k", 1) <= 0:
        raise ValueError("retrieval.top_k must be positive.")

    if get_by_dot_key(cfg, "reasoning.max_answer_words", 1) <= 0:
        raise ValueError("reasoning.max_answer_words must be positive.")


def load_config(
    config_path: Optional[str | os.PathLike] = None,
    default_path: str | os.PathLike = "configs/default.yaml",
    cli_overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> ConfigDict:
    """
    Load and merge experiment configuration.

    Merge priority:
        1. configs/default.yaml
        2. config_path
        3. cli_overrides

    Args:
        config_path: Dataset, baseline, or ablation config path.
        default_path: Default config path.
        cli_overrides: Dot-key overrides.
        validate: Whether to validate required fields.

    Returns:
        Final merged config dictionary.
    """
    cfg = load_yaml(default_path)

    if config_path is not None:
        specific_cfg = load_yaml(config_path)
        cfg = deep_update(cfg, specific_cfg)

    cfg = apply_cli_overrides(cfg, cli_overrides)

    if validate:
        validate_config(cfg)

    return cfg


def save_config(cfg: ConfigDict, output_path: str | os.PathLike) -> None:
    """
    Save config dictionary to a YAML file.

    Args:
        cfg: Config dictionary.
        output_path: Output YAML path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )