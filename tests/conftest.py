"""
Global fixtures — env vars must be set before any module import.
"""
import os
os.environ.setdefault("YOUTUBE_API_KEY", "test_api_key_for_testing")

from pathlib import Path
import pytest
import cv2
import numpy as np

# ── 프로젝트 루트 기준 실제 데이터 경로 ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR   = PROJECT_ROOT / "data" / "frames"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"


# ── 채널별 샘플 프레임 픽스처 ─────────────────────────────────────────

@pytest.fixture(scope="session")
def kbsn_frame():
    path = FRAMES_DIR / "kbsn_sample.png"
    frame = cv2.imread(str(path))
    assert frame is not None, f"샘플 프레임 없음: {path}"
    return frame


@pytest.fixture(scope="session")
def spotv_frame():
    path = FRAMES_DIR / "spotv_sample.png"
    frame = cv2.imread(str(path))
    assert frame is not None, f"샘플 프레임 없음: {path}"
    return frame


@pytest.fixture(scope="session")
def mbcsports_frame():
    path = FRAMES_DIR / "mbcsports_sample.png"
    frame = cv2.imread(str(path))
    assert frame is not None, f"샘플 프레임 없음: {path}"
    return frame


# ── 실제 raw_data 영상 경로 ───────────────────────────────────────────

@pytest.fixture(scope="session")
def first_video_path():
    """raw_data 디렉토리에서 날짜순 첫 번째 영상 경로."""
    videos = sorted(RAW_DATA_DIR.glob("*.webm"))
    if not videos:
        pytest.skip("raw_data 디렉토리에 영상 파일 없음")
    return str(videos[0])


# ── 실제 channel configs (load_all_configs 결과) ─────────────────────

@pytest.fixture(scope="session")
def real_configs():
    from src.detection.channel_detector import load_all_configs
    return load_all_configs()
