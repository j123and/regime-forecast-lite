# core/config.py
from __future__ import annotations
from pathlib import Path
import os, yaml
from typing import Any, Dict

def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    # shallow is enough for this project; upgrade to deep-merge if you need
    out = dict(a)
    out.update(b)
    return out

def load_config(config: str | os.PathLike | None = None,
                profile: str | None = None,
                **kwargs) -> Dict[str, Any]:
    """
    Resolve config by priority:
      1) explicit path (--config)
      2) $REGIME_CONFIG
      3) profile (--profile or $REGIME_PROFILE) -> config/profiles/<profile>.yaml
      4) fallback: config/default.yaml
    Accepts legacy 'path=' via **kwargs for backwards compatibility.
    """
    # legacy alias support
    if config is None and "path" in kwargs and kwargs["path"]:
        config = kwargs["path"]

    repo_root = Path(__file__).resolve().parents[1]  # project root
    cfg: Dict[str, Any] = {}

    # 1) explicit path
    if config:
        p = Path(config)
        if not p.is_file():
            raise FileNotFoundError(f"--config not found: {p}")
        cfg = _merge(cfg, _read_yaml(p))
        return cfg  # explicit path wins outright

    # 2) env path
    env_path = os.getenv("REGIME_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            cfg = _merge(cfg, _read_yaml(p))
            return cfg

    # 3) profile
    prof = profile or os.getenv("REGIME_PROFILE")
    if prof:
        p = repo_root / "config" / "profiles" / f"{prof}.yaml"
        if not p.is_file():
            raise FileNotFoundError(f"profile not found: {p}")
        cfg = _merge(cfg, _read_yaml(p))
        return cfg

    # 4) fallback
    fallback = repo_root / "config" / "default.yaml"
    if fallback.is_file():
        cfg = _merge(cfg, _read_yaml(fallback))
        return cfg

    # nothing found → empty config
    return {}
