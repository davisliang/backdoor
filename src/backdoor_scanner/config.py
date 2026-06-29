# Copyright (c) 2026 Davis Liang. MIT License.
"""Configuration loading and merging for the backdoor scanner.

A single YAML file (see ``configs/default.yaml``) holds every tunable. The
:func:`load_config` helper deep-merges, in order: the packaged defaults, an
optional user YAML, and explicit keyword overrides (typically from the CLI).
The result is a plain nested ``dict`` consumed throughout the package.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Packaged default config. When installed as a wheel it lives under
# ``backdoor_scanner/_data/default.yaml``; in a source checkout it is at
# ``<repo>/configs/default.yaml``.
_PKG_DEFAULT = Path(__file__).parent / "_data" / "default.yaml"
_REPO_DEFAULT = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def default_config_path() -> Path:
    if _PKG_DEFAULT.exists():
        return _PKG_DEFAULT
    return _REPO_DEFAULT


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if val is None:
            # Treat None as "leave the base value alone" so partial CLI overrides
            # don't wipe defaults. Use an explicit value to override.
            if key not in out:
                out[key] = val
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(
    user_config: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load defaults, merge a user YAML, then apply keyword overrides."""
    with open(default_config_path(), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if user_config is not None:
        with open(user_config, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg


def dump_config(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
