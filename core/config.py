# core/config.py
from __future__ import annotations

from pathlib import Path
import os
import yaml
from typing import Any, Dict


def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update(b)
    return out


def _postprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light normalization so the rest of the code sees flat keys it actually reads.
    """
    det = cfg.get("detector")
    if isinstance(det, dict):
        # Map detector.vol_threshold to the flat key if not already present
        if "regime_vol_threshold" not in cfg and "vol_threshold" in det:
            try:
                cfg["regime_vol_threshold"] = float(det["vol_threshold"])
            except Exception:
                pass
    return cfg


def load_config(
    config: str | os.PathLike | None = None,
    profile: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Resolve config with the following rules:

      A) If an explicit file is provided, it wins outright:
         1) --config (or legacy path=)
         2) $REGIME_CONFIG

      B) Otherwise, layer files:
         3) config/default.yaml (if present)
         4) config/profiles/<profile>.yaml when --profile or $REGIME_PROFILE is set
            (profile overlays default)

    Returns a (shallow) merged dict; also normalizes a few nested keys.
    """
    if config is None and "path" in kwargs and kwargs["path"]:
        config = kwargs["path"]

    repo_root = Path(__file__).resolve().parents[1]  

    if config:
        p = Path(config)
        if not p.is_file():
            raise FileNotFoundError(f"--config not found: {p}")
        return _postprocess(_read_yaml(p))

    env_path = os.getenv("REGIME_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return _postprocess(_read_yaml(p))


    cfg: Dict[str, Any] = {}

    fallback = repo_root / "config" / "default.yaml"
    if fallback.is_file():
        cfg = _merge(cfg, _read_yaml(fallback))

    prof = profile or os.getenv("REGIME_PROFILE")
    if prof:
        p = repo_root / "config" / "profiles" / f"{prof}.yaml"
        if not p.is_file():
            raise FileNotFoundError(f"profile not found: {p}")
        cfg = _merge(cfg, _read_yaml(p))

    return _postprocess(cfg)
