from pathlib import Path
import yaml

def load_config(path: str = "config/default.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        return yaml.safe_load(f) or {}
