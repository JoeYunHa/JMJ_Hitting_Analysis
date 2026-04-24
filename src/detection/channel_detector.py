import logging
import json
import cv2
import numpy as np
from pathlib import Path

from src.exceptions import ConfigError, VideoProcessingError
from src.utils.video import open_video
from src.config.params import ChannelDetectionParams

logger = logging.getLogger(__name__)

CHANNEL_CONFIGS_DIR = Path("configs/channels")


def load_all_configs() -> dict:
    """Load all channel JSON configs recursively, pre-loading logo templates.

    Raises ConfigError if the configs directory does not exist.
    """
    if not CHANNEL_CONFIGS_DIR.exists():
        raise ConfigError(
            f"Channel configs directory not found: {CHANNEL_CONFIGS_DIR}. "
            "Create channel JSON configs under configs/channels/ first."
        )

    configs = {}
    for f in CHANNEL_CONFIGS_DIR.rglob("*.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                cfg = json.load(fp)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping malformed config {f}: {e}")
            continue

        tmpl_path = cfg.get("logo_template")
        if tmpl_path and Path(tmpl_path).exists():
            tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            cfg["_template_img"] = tmpl if tmpl is not None else None
        else:
            cfg["_template_img"] = None
            if tmpl_path:
                logger.warning(f"Logo template not found: {tmpl_path} (config: {f})")

        configs[cfg["config_id"]] = cfg

    logger.info(f"Loaded {len(configs)} channel configs.")
    return configs


def detect_channel(
    frame: np.ndarray,
    configs: dict,
    params: ChannelDetectionParams | None = None,
) -> str | None:
    """Match pre-loaded logo template against frame; return config_id or None."""
    p = params or ChannelDetectionParams()
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
        if best_val >= p.early_exit_confidence:
            logger.debug(f"Early exit: {best_id} confidence={best_val:.3f}")
            break

    return best_id if best_val >= p.threshold else None


def detect_channel_from_video(
    video_path: str,
    configs: dict,
    params: ChannelDetectionParams | None = None,
) -> str | None:
    """Sample frames from first max_sample_sec seconds and return best-matching config_id."""
    p = params or ChannelDetectionParams()

    try:
        with open_video(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(p.sample_interval_sec * fps))
            limit = min(total, int(p.max_sample_sec * fps))

            best_id, best_val = None, 0.0
            for idx in range(0, limit, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    logger.debug(f"Failed to read frame {idx} from {video_path}")
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

                if best_val >= p.early_exit_confidence:
                    logger.debug(
                        f"Early exit at frame {idx}: {best_id} confidence={best_val:.3f}"
                    )
                    break

    except VideoProcessingError as e:
        logger.error(f"Cannot open video for channel detection: {e}")
        return None

    result = best_id if best_val >= p.threshold else None
    logger.info(
        f"Channel detection: {result} (confidence={best_val:.3f}) for {video_path}"
    )
    return result
