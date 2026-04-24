import json
from pathlib import Path
import cv2
from configs.settings import TARGET_NAME
from src.utils.video import open_video
from .channel_detector import load_all_configs, detect_channel
from .ocr_extractor import extract_batter_segments
from .swing_detector import _detect_swings, FlowConfig


def run_phase3(
    video_path: str,
    output_dir: str = "outputs",
    target_name: str = TARGET_NAME,
) -> str:
    """
    Full Phase 3 pipeline:
    1. Detect channel from first frame
    2. Extract batter segments via OCR (with delay compensation)
    3. Detect swing moments via Optical Flow (single VideoCapture pass)
    4. Save results as JSON
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    configs = load_all_configs()

    # Step 1: detect channel
    with open_video(video_path) as cap:
        ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")

    config_id = detect_channel(first_frame, configs)
    if config_id is None:
        raise ValueError(
            f"Channel not recognized for video: {video_path}. "
            "Verify logo templates exist under configs/channels/templates/."
        )

    cfg = configs[config_id]
    channel_id = cfg["channel"]
    print(f"[Phase3] Channel: {channel_id}")

    # Step 2: OCR-based batter segment extraction (delay-compensated)
    segments = extract_batter_segments(video_path, cfg, target_name=target_name)
    print(f"[Phase3] Segments found: {len(segments)}")

    # Step 3: swing detection — single VideoCapture reused across all segments
    with open_video(video_path) as cap:
        for seg in segments:
            seg.swing_frames = _detect_swings(
                cap, seg.start_frame, seg.end_frame,
                flow_threshold=4.0, cooldown_frames=15, flow_config=FlowConfig(),
            )
            print(
                f"  Segment {seg.start_sec:.1f}s~{seg.end_sec:.1f}s "
                f"→ {len(seg.swing_frames)} swings"
            )

    # Step 4: save
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_segments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channel": channel_id,
                "target": target_name,
                "segments": [seg.to_dict() for seg in segments],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Phase3] Saved: {out_path}")
    return str(out_path)
