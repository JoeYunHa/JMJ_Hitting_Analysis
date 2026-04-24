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


def detect_channel_from_video(
    video_path: str,
    configs: dict,
    threshold: float = 0.8,
    sample_interval_sec: float = 5.0,
    max_sample_sec: float = 60.0,
) -> str | None:
    """Sample frames from first max_sample_sec seconds and return best-matching config_id."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(sample_interval_sec * fps))
    limit = min(total, int(max_sample_sec * fps))

    best_id, best_val = None, 0.0
    for idx in range(0, limit, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for cfg_id, cfg in configs.items():
            tmpl = cfg.get("_template_img")
            if tmpl is None:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            val = float(res.max())
            if val > best_val:
                best_val, best_id = val, cfg_id
    cap.release()
    return best_id if best_val >= threshold else None
