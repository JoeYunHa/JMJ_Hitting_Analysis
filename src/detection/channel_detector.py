import cv2
import numpy as np
from pathlib import Path
import json

CHANNEL_CONFIGS_DIR = Path("configs/channels")


def load_all_configs() -> dict:
    """Load all channel JSON configs recursively, pre-loading logo templates."""
    configs = {}
    for f in CHANNEL_CONFIGS_DIR.rglob("*.json"):
        with open(f) as fp:
            cfg = json.load(fp)
        tmpl_path = cfg.get("logo_template")
        if tmpl_path and Path(tmpl_path).exists():
            tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            cfg["_template_img"] = tmpl if tmpl is not None else None
        else:
            cfg["_template_img"] = None
        configs[cfg["config_id"]] = cfg
    return configs


def detect_channel(
    frame: np.ndarray, configs: dict, threshold: float = 0.8
) -> str | None:
    """Match pre-loaded logo template against frame; return config_id or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_id, best_val = None, 0.0
    for cfg_id, cfg in configs.items():
        tmpl = cfg.get("_template_img")
        if tmpl is None:
            continue
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        val = float(res.max())
        if val > best_val:
            best_val, best_id = val, cfg_id
    return best_id if best_val >= threshold else None
