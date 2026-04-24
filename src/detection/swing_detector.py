import logging
import cv2
import numpy as np
from dataclasses import dataclass

from src.utils.video import open_video, validate_frame_range
from src.config.params import SwingDetectionParams

logger = logging.getLogger(__name__)


@dataclass
class FlowConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2


def detect_swing_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    params: SwingDetectionParams | None = None,
    flow_config: FlowConfig | None = None,
) -> list[int]:
    """
    Use Optical Flow magnitude spike to find swing moments within a segment.
    Returns list of frame indices where swing is detected.
    """
    p = params or SwingDetectionParams()
    with open_video(video_path) as cap:
        validate_frame_range(cap, start_frame, end_frame, video_path)
        return _detect_swings(cap, start_frame, end_frame, p, flow_config or FlowConfig())


def _detect_swings(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    params: SwingDetectionParams,
    flow_config: FlowConfig,
) -> list[int]:
    """Core swing detection on an already-open VideoCapture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev = cap.read()
    if not ret:
        logger.warning(f"Cannot read start frame {start_frame} for swing detection.")
        return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    swing_frames: list[int] = []
    cooldown = 0
    frame_idx = start_frame + 1

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Frame read failed at {frame_idx}, stopping swing detection.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=flow_config.pyr_scale,
            levels=flow_config.levels,
            winsize=flow_config.winsize,
            iterations=flow_config.iterations,
            poly_n=flow_config.poly_n,
            poly_sigma=flow_config.poly_sigma,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        if mag > params.flow_threshold and cooldown == 0:
            swing_frames.append(frame_idx)
            cooldown = params.cooldown_frames
            logger.debug(f"Swing detected at frame {frame_idx} (mag={mag:.2f})")
        cooldown = max(0, cooldown - 1)
        prev_gray = gray
        frame_idx += 1

    return swing_frames
