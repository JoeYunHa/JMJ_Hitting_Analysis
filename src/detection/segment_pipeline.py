import json
import logging
from pathlib import Path

from configs.settings import TARGET_NAME
from src.utils.video import open_video
from src.exceptions import PipelineError
from src.config.params import SwingDetectionParams
from .channel_detector import load_all_configs, detect_channel_from_video
from .ocr_extractor import extract_batter_segments
from .swing_detector import _detect_swings, FlowConfig

logger = logging.getLogger(__name__)


def run_phase3(
    video_path: str,
    output_dir: str = "outputs",
    target_name: str = TARGET_NAME,
    swing_params: SwingDetectionParams | None = None,
    skip_if_exists: bool = False,
) -> str:
    """
    Full Phase 3 pipeline:
    1. Detect channel from first frame
    2. Extract batter segments via OCR (with delay compensation)
    3. Detect swing moments via Optical Flow (single VideoCapture pass)
    4. Save results as JSON
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_segments.json"

    if skip_if_exists and out_path.exists():
        logger.info(f"[Phase3] Skipping {stem}, output already exists.")
        return str(out_path)

    configs = load_all_configs()

    # Step 1: detect channel
    config_id = detect_channel_from_video(video_path, configs)
    if config_id is None:
        raise ValueError(
            f"Channel not recognized for video: {video_path}. "
            "Verify logo templates exist under configs/channels/templates/."
        )

    cfg = configs[config_id]
    channel_id = cfg["channel"]
    logger.info(f"[Phase3] Channel: {channel_id}")

    # Step 2: OCR-based batter segment extraction
    segments = extract_batter_segments(video_path, cfg, target_name=target_name)
    logger.info(f"[Phase3] Segments found: {len(segments)}")

    # Step 3: swing detection — single VideoCapture reused across all segments
    sp = swing_params or SwingDetectionParams()
    with open_video(video_path) as cap:
        for seg in segments:
            seg.swing_frames = _detect_swings(
                cap, seg.start_frame, seg.end_frame, sp, FlowConfig()
            )
            logger.info(
                f"  Segment {seg.start_sec:.1f}s~{seg.end_sec:.1f}s "
                f"-> {len(seg.swing_frames)} swings"
            )

    # Step 4: save
    payload = {
        "channel": channel_id,
        "target": target_name,
        "segments": [seg.to_dict() for seg in segments],
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise PipelineError(f"Failed to write phase3 output {out_path}: {e}") from e

    logger.info(f"[Phase3] Saved: {out_path}")
    return str(out_path)
