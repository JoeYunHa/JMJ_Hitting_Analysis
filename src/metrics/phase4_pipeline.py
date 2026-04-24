import json
import logging
import re
from pathlib import Path

from configs.settings import TARGET_NAME
from src.detection.channel_detector import load_all_configs
from src.pose.pose_estimator import get_estimator
from src.metrics.swing_metrics import compute_segment_metrics
from src.metrics.db_client import get_player_code, get_woba_stats
from src.metrics.performance_tagger import tag_performance
from src.exceptions import PipelineError
from src.config.params import Phase4Params

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})")


def _parse_game_date(stem: str) -> str | None:
    m = _DATE_RE.match(stem)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _frame_window(swing_frames: list[int], seg_start: int, seg_end: int, window: int) -> list[int]:
    frames = set()
    for sf in swing_frames:
        frames.update(range(max(seg_start, sf - window), min(seg_end, sf + window) + 1))
    if not frames:
        step = max(1, (seg_end - seg_start) // 10)
        frames.update(range(seg_start, seg_end + 1, step))
    return sorted(frames)


def run_phase4(
    phase3_json: str,
    video_path: str,
    output_dir: str = "outputs",
    target_name: str = TARGET_NAME,
    params: Phase4Params | None = None,
    skip_if_exists: bool = False,
) -> str:
    """
    Full Phase 4 pipeline:
    1. Load Phase 3 JSON
    2. Parse game_date from filename
    3. Fetch rolling wOBA from DB -> performance_tag
    4. Extract keypoints (YOLO) for swing_frames +/- window per segment
    5. Compute swing metrics per segment
    6. Save outputs/{stem}_phase4.json
    """
    p = params or Phase4Params()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_phase4.json"

    if skip_if_exists and out_path.exists():
        logger.info(f"[Phase4] Skipping {stem}, output already exists.")
        return str(out_path)

    try:
        with open(phase3_json, encoding="utf-8") as f:
            phase3 = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise PipelineError(f"Failed to load phase3 JSON '{phase3_json}': {e}") from e

    game_date = _parse_game_date(stem)
    channel_id = phase3.get("channel", "unknown")
    segments = phase3.get("segments", [])

    logger.info(f"[Phase4] video={stem}  date={game_date}  segments={len(segments)}")

    # --- DB: player_code + wOBA stats ---
    player_code = get_player_code(target_name)
    if player_code is None:
        logger.warning(f"[Phase4] player_code not found for '{target_name}', DB stats skipped.")

    woba_stats = get_woba_stats(game_date, player_code) if (game_date and player_code) else None
    perf_result = tag_performance(woba_stats)
    logger.info(
        f"[Phase4] player_code={player_code}  "
        f"woba={perf_result.get('woba')}  woba_source={perf_result.get('woba_source')}  "
        f"lg_woba={woba_stats.get('lg_woba') if woba_stats else None}  "
        f"tag={perf_result['tag']}"
    )

    # --- batter_zone_roi from channel config ---
    all_configs = load_all_configs()
    cfg = next(
        (c for c in all_configs.values() if c.get("channel") == channel_id),
        None,
    )
    batter_zone_roi = cfg.get("batter_zone_roi") if cfg else None
    if batter_zone_roi is None:
        logger.warning(
            f"[Phase4] batter_zone_roi not found for channel '{channel_id}', "
            "using largest-bbox fallback."
        )

    # --- YOLO keypoint extraction ---
    estimator = get_estimator()
    enriched_segments = []

    for i, seg in enumerate(segments):
        seg_start = seg["start_frame"]
        seg_end = seg["end_frame"]
        swing_frames = seg.get("swing_frames", [])

        target_frames = _frame_window(swing_frames, seg_start, seg_end, p.swing_window)
        kp_map = estimator.extract_keypoints(video_path, target_frames, batter_zone_roi)

        kp_serializable = {str(fidx): kp.tolist() for fidx, kp in kp_map.items()}
        metrics = compute_segment_metrics(kp_map, swing_frames, seg_start)

        logger.info(
            f"  seg[{i}] {seg['start_sec']:.1f}s~{seg['end_sec']:.1f}s  "
            f"frames={len(target_frames)}  kp_extracted={len(kp_map)}  "
            f"shoulder={metrics['shoulder_rotation_deg']}deg"
        )

        enriched_segments.append({**seg, "keypoints": kp_serializable, "metrics": metrics})

    # --- Save ---
    out = {
        "video": stem,
        "game_date": game_date,
        "channel": channel_id,
        "target": target_name,
        "player_code": player_code,
        "performance_tag": perf_result["tag"],
        "woba_source": perf_result["woba_source"],
        "woba": perf_result["woba"],
        "woba_cumul": woba_stats.get("woba_cumul") if woba_stats else None,
        "lg_woba": woba_stats.get("lg_woba") if woba_stats else None,
        "early_season": woba_stats.get("early_season") if woba_stats else None,
        "segments": enriched_segments,
    }

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise PipelineError(f"Failed to write phase4 output '{out_path}': {e}") from e

    logger.info(f"[Phase4] Saved: {out_path}")
    return str(out_path)
