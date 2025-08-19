from __future__ import annotations

import os
from pathlib import Path

import yaml


def _resolve_config_path(path: str | None, profile: str | None) -> Path:
    if path:
        return Path(path)

    # env overrides first
    env_path = os.getenv("REGIME_CONFIG")
    if env_path:
        return Path(env_path)

    env_profile = os.getenv("REGIME_PROFILE")
    prof = profile or env_profile
    if prof:
        p = Path(f"config/profiles/{prof}.yaml")
        if p.exists():
            return p
        # backward-compat fallback if someone kept old layout
        p2 = Path(f"config/{prof}.yaml")
        if p2.exists():
            return p2

    return Path("config/default.yaml")


def load_config(path: str | None = None, profile: str | None = None) -> dict:
    p = _resolve_config_path(path, profile)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data
