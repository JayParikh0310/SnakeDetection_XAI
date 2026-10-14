"""SnakeGuard: explainable snake vs. look-alike detection."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = {0: "snake", 1: "snake_like_object"}


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("weights", "data_root", "data_yaml", "results_dir"):
        cfg[k] = str((PROJECT_ROOT / cfg[k]).resolve())
    return cfg
