from __future__ import annotations

import os
from pathlib import Path

import yaml


def _project_root() -> Path:
    # core/ is one level under repo root
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(path: str | None, profile: str | None) -> Path:
    """
    Resolve config path robustly:
      1) explicit --config
      2) $REGIME_CONFIG
      3) profile (CLI or $REGIME_PROFILE) under config/profiles/<profile>.yaml
         or config/<profile>.yaml
      4) default: config/default.yaml
    All paths are resolved relative to repo root, not CWD.
    """
    root = _project_root()

    if path:
        p = Path(path)
        return p if p.is_absolute() else (root / p)

    # env overrides first
    env_path = os.getenv("REGIME_CONFIG")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else (root / p)

    env_profile = os.getenv("REGIME_PROFILE")
    prof = profile or env_profile
    if prof:
        cand = root / f"config/profiles/{prof}.yaml"
        if cand.exists():
            return cand
        # backward-compat fallback if someone kept old layout
        cand2 = root / f"config/{prof}.yaml"
        if cand2.exists():
            return cand2

    return root / "config/default.yaml"


def load_config(path: str | None = None, profile: str | None = None) -> dict:
    p = _resolve_config_path(path, profile)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data
