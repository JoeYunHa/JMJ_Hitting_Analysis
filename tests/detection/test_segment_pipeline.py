"""
Tests for src/detection/segment_pipeline.py

run_phase3() 전체 파이프라인 통합 테스트.

채널 탐지(channel_detector)는 로고 템플릿이 없으므로 mock 처리,
OCR/스윙 탐지는 실제 영상으로 실행.
실행: pytest -m slow  (영상 전체 스캔 → 수 분 소요)
"""
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.detection.segment_pipeline import run_phase3


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

KBSN_CONFIG_ID = "kbo_kbsn_2026_game_highlight"


def _first_kbsn_video() -> str | None:
    """raw_data 중 KBSN 방송사로 추정되는 첫 번째 영상 경로."""
    raw = Path(__file__).parent.parent.parent / "data" / "raw_data"
    videos = sorted(raw.glob("*.webm"))
    return str(videos[0]) if videos else None


# ---------------------------------------------------------------------------
# 에러 처리 테스트 (빠른 테스트 — 영상 전체를 읽지 않음)
# ---------------------------------------------------------------------------

class TestRunPhase3ErrorHandling:

    def test_raises_on_nonexistent_video(self, tmp_path):
        """존재하지 않는 영상 경로는 ValueError를 발생시킨다."""
        with pytest.raises((ValueError, Exception)):
            run_phase3(str(tmp_path / "nonexistent.webm"), output_dir=str(tmp_path))

    def test_raises_when_channel_not_recognized(self, first_video_path, tmp_path):
        """채널 탐지 결과가 None이면 ValueError를 발생시킨다."""
        with patch("src.detection.segment_pipeline.detect_channel", return_value=None):
            with pytest.raises(ValueError, match="Channel not recognized"):
                run_phase3(first_video_path, output_dir=str(tmp_path))

    def test_output_dir_created_if_missing(self, first_video_path, tmp_path):
        """output_dir이 없어도 자동 생성한다 (채널 탐지 mock)."""
        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()

        with patch("src.detection.segment_pipeline.detect_channel", return_value=None):
            try:
                run_phase3(first_video_path, output_dir=str(new_dir))
            except ValueError:
                pass  # channel not recognized — 이후 로직은 검증 불필요

        assert new_dir.exists()


# ---------------------------------------------------------------------------
# 통합 테스트 — 실제 영상 + mocked channel detection
# 실행: pytest -m slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunPhase3Integration:

    @pytest.fixture
    def kbsn_video(self):
        path = _first_kbsn_video()
        if path is None:
            pytest.skip("raw_data에 영상 파일 없음")
        return path

    def test_output_json_created(self, kbsn_video, tmp_path, real_configs):
        """run_phase3 실행 후 *_segments.json 파일이 생성된다."""
        with patch("src.detection.segment_pipeline.detect_channel",
                   return_value=KBSN_CONFIG_ID):
            out_path = run_phase3(kbsn_video, output_dir=str(tmp_path))

        assert Path(out_path).exists(), f"출력 파일 없음: {out_path}"

    def test_output_json_structure(self, kbsn_video, tmp_path, real_configs):
        """출력 JSON이 channel / target / segments 키를 포함한다."""
        with patch("src.detection.segment_pipeline.detect_channel",
                   return_value=KBSN_CONFIG_ID):
            out_path = run_phase3(kbsn_video, output_dir=str(tmp_path))

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "channel" in data
        assert "target" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)

    def test_output_channel_matches_config(self, kbsn_video, tmp_path, real_configs):
        """출력 JSON의 channel 값이 config의 channel 필드와 일치한다."""
        with patch("src.detection.segment_pipeline.detect_channel",
                   return_value=KBSN_CONFIG_ID):
            out_path = run_phase3(kbsn_video, output_dir=str(tmp_path))

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["channel"] == real_configs[KBSN_CONFIG_ID]["channel"]

    def test_segment_fields_valid(self, kbsn_video, tmp_path, real_configs):
        """각 세그먼트가 start_frame / end_frame / start_sec / end_sec / swing_frames 필드를 갖는다."""
        with patch("src.detection.segment_pipeline.detect_channel",
                   return_value=KBSN_CONFIG_ID):
            out_path = run_phase3(kbsn_video, output_dir=str(tmp_path))

        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)

        required = {"start_frame", "end_frame", "start_sec", "end_sec", "swing_frames"}
        for seg in data["segments"]:
            assert required.issubset(seg.keys()), f"세그먼트 필드 누락: {seg}"
            assert seg["start_frame"] <= seg["end_frame"]
            assert seg["start_sec"] <= seg["end_sec"]
            assert isinstance(seg["swing_frames"], list)

    def test_output_filename_matches_video_stem(self, kbsn_video, tmp_path, real_configs):
        """출력 파일명이 영상 파일명 stem + _segments.json 형태이다."""
        with patch("src.detection.segment_pipeline.detect_channel",
                   return_value=KBSN_CONFIG_ID):
            out_path = run_phase3(kbsn_video, output_dir=str(tmp_path))

        stem = Path(kbsn_video).stem
        expected = tmp_path / f"{stem}_segments.json"
        assert Path(out_path) == expected
