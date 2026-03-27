"""
config.py — YAML + .env configuration loader with stage-specific overrides.

Usage:
    from utils.config import load_config
    cfg = load_config("stage_c/config.yaml")
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def load_config(config_path: str) -> dict:
    """
    Load a YAML config file and resolve environment variable placeholders.
    
    Supports ${ENV_VAR} syntax in YAML values for secret injection.
    
    Args:
        config_path: Path to the YAML config file (relative to project root or absolute).
        
    Returns:
        dict: Parsed configuration with env vars resolved.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return _resolve_env_vars(config)


def _resolve_env_vars(obj):
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_key = obj[2:-1]
        value = os.environ.get(env_key)
        if value is None:
            raise EnvironmentError(
                f"Environment variable '{env_key}' not set. "
                f"Check your .env file or export it."
            )
        return value
    return obj


def get_env(key: str, default: str = None) -> str:
    """Get an environment variable with an optional default."""
    value = os.environ.get(key, default)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{key}' not set and no default provided."
        )
    return value


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return _PROJECT_ROOT
