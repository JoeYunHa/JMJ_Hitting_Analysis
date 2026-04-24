from contextlib import contextmanager
from typing import Generator
import cv2


@contextmanager
def open_video(path: str) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager for cv2.VideoCapture — always releases on exit."""
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()
