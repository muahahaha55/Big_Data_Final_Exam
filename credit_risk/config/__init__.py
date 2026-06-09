"""Configuration loader with hierarchical merge.

Pattern:
    1. Load `conf/base/config.yaml` (defaults, version-controlled)
    2. Overlay `conf/{env}/config.yaml` if exists (env-specific)
    3. Overlay `conf/local/config.yaml` if exists (developer overrides, gitignored)
    4. Apply environment variable overrides via $CRP_* prefix

Usage:
    from credit_risk.config import get_config
    cfg = get_config()
    print(cfg["models"]["default_model"])
"""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ENV_VAR = "CRP_ENV"
ENV_PREFIX = "CRP_"


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until we find pyproject.toml."""
    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate project root (no pyproject.toml found upward)")


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into `base`. Overlay wins on conflicts."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply env var overrides. Pattern: CRP_models__default_model=logistic
    becomes cfg['models']['default_model'] = 'logistic'."""
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key == ENV_VAR:
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce_value(value)
    return cfg


def _coerce_value(value: str) -> Any:
    """Best-effort type coercion for env var strings."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


@lru_cache(maxsize=1)
def get_config(env: str | None = None) -> dict[str, Any]:
    """Load and merge configuration.

    Args:
        env: Override environment (default: $CRP_ENV or 'base').

    Returns:
        Dict containing the merged configuration.
    """
    root = _find_project_root()
    env = env or os.environ.get(ENV_VAR, "base")
    conf_dir = root / "conf"

    base_cfg = _load_yaml(conf_dir / "base" / "config.yaml")

    if env != "base":
        env_cfg = _load_yaml(conf_dir / env / "config.yaml")
        base_cfg = _deep_merge(base_cfg, env_cfg)

    local_cfg = _load_yaml(conf_dir / "local" / "config.yaml")
    if local_cfg:
        base_cfg = _deep_merge(base_cfg, local_cfg)

    base_cfg = _apply_env_overrides(base_cfg)
    base_cfg["_meta"] = {"env": env, "project_root": str(root)}
    return base_cfg


def project_root() -> Path:
    """Convenience accessor for the project root."""
    return _find_project_root()


def reset_cache() -> None:
    """Clear the LRU cache — useful for tests."""
    get_config.cache_clear()
