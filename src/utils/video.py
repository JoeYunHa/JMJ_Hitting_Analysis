from contextlib import contextmanager
from typing import Generator
import cv2

from src.exceptions import VideoProcessingError


@contextmanager
def open_video(path: str) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager for cv2.VideoCapture — always releases on exit.

    Raises VideoProcessingError if the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(f"Cannot open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def get_validated_fps(cap: cv2.VideoCapture, path: str) -> float:
    """Read FPS from VideoCapture and raise VideoProcessingError if invalid."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise VideoProcessingError(f"Invalid FPS ({fps}) in video: {path}")
    return fps


def validate_frame_range(
    cap: cv2.VideoCapture, start_frame: int, end_frame: int, path: str = ""
) -> None:
    """Raise VideoProcessingError if [start_frame, end_frame] is out of bounds."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or end_frame >= total or start_frame > end_frame:
        raise VideoProcessingError(
            f"Invalid frame range [{start_frame}, {end_frame}] for video "
            f"'{path}' with {total} total frames."
        )
