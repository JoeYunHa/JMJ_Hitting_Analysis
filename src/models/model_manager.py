"""
Thread-safe singleton manager for heavy ML models (OCR, YOLO).

Usage:
    from src.models.model_manager import get_ocr, get_pose_estimator, cleanup
"""

import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_ocr = None
_estimator = None


def get_ocr():
    """Return thread-safe PaddleOCR singleton, initializing on first call."""
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                import os
                os.environ["FLAGS_use_mkldnn"] = "0"
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR model...")
                _ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang="korean",
                    enable_mkldnn=False,
                )
                logger.info("PaddleOCR ready.")
    return _ocr


def get_pose_estimator():
    """Return thread-safe YoloPoseEstimator singleton, initializing on first call."""
    global _estimator
    if _estimator is None:
        with _lock:
            if _estimator is None:
                from src.pose.pose_estimator import YoloPoseEstimator
                from src.config.params import PoseEstimationParams
                p = PoseEstimationParams()
                logger.info(f"Initializing YOLO pose estimator ({p.model_name})...")
                _estimator = YoloPoseEstimator(model_name=p.model_name, conf=p.conf)
                logger.info("YOLO pose estimator ready.")
    return _estimator


def cleanup() -> None:
    """Release all model references (frees GPU/CPU memory on next GC cycle)."""
    global _ocr, _estimator
    with _lock:
        _ocr = None
        _estimator = None
    logger.info("Model manager: all resources released.")
