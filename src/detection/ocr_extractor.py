import logging
import cv2
import numpy as np

from configs.settings import TARGET_NAME
from src.utils.video import open_video, get_validated_fps
from src.utils.types import Segment
from src.models.model_manager import get_ocr
from src.config.params import OcrExtractionParams

logger = logging.getLogger(__name__)


def roi_to_xyxy(roi: dict) -> tuple[int, int, int, int]:
    """Convert {x,y,w,h} roi dict to (x1,y1,x2,y2)."""
    return roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]


def crop_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    """Crop frame using {x,y,w,h} roi dict."""
    x1, y1, x2, y2 = roi_to_xyxy(roi)
    return frame[y1:y2, x1:x2]


def ocr_contains_name(
    frame: np.ndarray, roi: dict, target: str = TARGET_NAME
) -> bool:
    """Return True if target name detected in batter_name_roi."""
    crop = crop_roi(frame, roi)
    try:
        results = get_ocr().predict(crop)
    except Exception as e:
        logger.warning(f"OCR predict failed: {e}")
        return False

    if not results:
        return False

    first = results[0]
    if not isinstance(first, dict) or "rec_texts" not in first:
        logger.warning(f"Unexpected OCR result structure: {type(first)}")
        return False

    return any(target in t for t in first["rec_texts"])


def extract_batter_segments(
    video_path: str,
    cfg: dict,
    target_name: str = TARGET_NAME,
    params: OcrExtractionParams | None = None,
) -> list[Segment]:
    """
    Scan video; return segments where target batter is active.
    Seeks directly to each sample frame to avoid decoding unused frames.
    ocr_delay_sec: rewinds segment start to compensate for batter appearing
                   before the name overlay renders.
    """
    p = params or OcrExtractionParams()

    with open_video(video_path) as cap:
        video_fps = get_validated_fps(cap, video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi = cfg["batter_name_roi"]
        delay_frames = int(p.ocr_delay_sec * video_fps)
        step = max(1, int(video_fps / p.sample_fps))
        gap_frames = int(p.gap_tolerance_sec * video_fps)

        segments: list[Segment] = []
        in_segment = False
        seg_start = 0
        last_seen_frame = -1

        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                break

            found = ocr_contains_name(frame, roi, target_name)
            if found:
                if not in_segment:
                    seg_start = max(0, frame_idx - delay_frames)
                    in_segment = True
                    logger.debug(f"Segment start at frame {seg_start}")
                last_seen_frame = frame_idx
            elif in_segment and (frame_idx - last_seen_frame) > gap_frames:
                segments.append(
                    Segment(
                        start_frame=seg_start,
                        end_frame=last_seen_frame,
                        start_sec=seg_start / video_fps,
                        end_sec=last_seen_frame / video_fps,
                    )
                )
                logger.debug(
                    f"Segment closed: {seg_start/video_fps:.1f}s~{last_seen_frame/video_fps:.1f}s"
                )
                in_segment = False

            frame_idx += step

        if in_segment:
            segments.append(
                Segment(
                    start_frame=seg_start,
                    end_frame=last_seen_frame,
                    start_sec=seg_start / video_fps,
                    end_sec=last_seen_frame / video_fps,
                )
            )

    logger.info(f"OCR extraction: {len(segments)} segments found in {video_path}")
    return segments
