"""Shared helpers for loading the central YAML configuration."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the repository-wide YAML config.

    Args:
        path: Optional override path. Defaults to ``config/config.yaml``.

    Returns:
        The parsed configuration mapping.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
