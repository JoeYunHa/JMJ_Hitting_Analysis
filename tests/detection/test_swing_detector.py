"""
Tests for src/detection/swing_detector.py

_detect_swings / detect_swing_frames — 실제 영상 파일의 짧은 구간 사용
FlowConfig dataclass 검증 포함
"""
import cv2
import numpy as np
import pytest

from src.detection.swing_detector import FlowConfig, detect_swing_frames, _detect_swings
from src.utils.video import open_video


# ---------------------------------------------------------------------------
# FlowConfig
# ---------------------------------------------------------------------------

class TestFlowConfig:

    def test_default_values(self):
        cfg = FlowConfig()
        assert cfg.pyr_scale == 0.5
        assert cfg.levels == 3
        assert cfg.winsize == 15
        assert cfg.iterations == 3
        assert cfg.poly_n == 5
        assert cfg.poly_sigma == 1.2

    def test_custom_values(self):
        cfg = FlowConfig(pyr_scale=0.3, levels=5, winsize=21)
        assert cfg.pyr_scale == 0.3
        assert cfg.levels == 5
        assert cfg.winsize == 21


# ---------------------------------------------------------------------------
# detect_swing_frames — 실제 영상 사용
# ---------------------------------------------------------------------------

class TestDetectSwingFramesWithRealVideo:

    def test_returns_list(self, first_video_path):
        """detect_swing_frames가 list를 반환한다."""
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 영상 앞부분 100프레임 구간만 검사
        end = min(100, total - 1)
        result = detect_swing_frames(first_video_path, start_frame=0, end_frame=end)
        assert isinstance(result, list)

    def test_result_frame_indices_within_range(self, first_video_path):
        """반환된 스윙 프레임 인덱스가 start_frame ~ end_frame 범위 안에 있다."""
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start, end = 0, min(100, total - 1)
        result = detect_swing_frames(first_video_path, start_frame=start, end_frame=end)
        for idx in result:
            assert start <= idx <= end, f"범위 밖 프레임 인덱스: {idx}"

    def test_result_indices_are_sorted(self, first_video_path):
        """스윙 프레임 인덱스가 오름차순으로 정렬되어 있다."""
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        end = min(200, total - 1)
        result = detect_swing_frames(first_video_path, start_frame=0, end_frame=end)
        assert result == sorted(result)

    def test_cooldown_enforced(self, first_video_path):
        """cooldown_frames=30이면 연속 스윙 탐지 간격이 30프레임 이상이다."""
        cooldown = 30
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        end = min(300, total - 1)
        result = detect_swing_frames(
            first_video_path, start_frame=0, end_frame=end,
            cooldown_frames=cooldown,
        )
        for i in range(1, len(result)):
            gap = result[i] - result[i - 1]
            assert gap >= cooldown, (
                f"쿨다운 위반: 프레임 {result[i-1]}~{result[i]} 간격={gap} < {cooldown}"
            )

    def test_high_threshold_detects_fewer_swings(self, first_video_path):
        """threshold가 높을수록 탐지되는 스윙 수가 적거나 같다."""
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        end = min(300, total - 1)
        low  = detect_swing_frames(first_video_path, 0, end, flow_threshold=2.0)
        high = detect_swing_frames(first_video_path, 0, end, flow_threshold=20.0)
        assert len(high) <= len(low)

    def test_empty_segment_returns_empty(self, first_video_path):
        """start_frame == end_frame이면 빈 리스트를 반환한다."""
        result = detect_swing_frames(first_video_path, start_frame=10, end_frame=10)
        assert result == []

    def test_invalid_start_returns_empty(self, first_video_path):
        """읽을 수 없는 프레임 위치(영상 끝 초과)에서 빈 리스트를 반환한다."""
        with open_video(first_video_path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        result = detect_swing_frames(
            first_video_path, start_frame=total + 100, end_frame=total + 200
        )
        assert result == []


# ---------------------------------------------------------------------------
# _detect_swings — 합성 프레임으로 광학흐름 크기 직접 검증
# ---------------------------------------------------------------------------

class TestDetectSwingsWithSyntheticFrames:
    """
    실제 영상 없이도 광학흐름 로직을 검증할 수 있는 합성(numpy) 프레임 기반 테스트.
    크게 이동한 패턴 프레임 → 높은 magnitude → 스윙 탐지됨
    정지 프레임들만 → 낮은 magnitude → 탐지 안 됨
    """

    def _make_mock_cap(self, frames: list[np.ndarray]) -> cv2.VideoCapture:
        """프레임 리스트를 순서대로 read()하는 가짜 VideoCapture."""
        import unittest.mock as mock
        cap = mock.MagicMock(spec=cv2.VideoCapture)

        read_returns = [(True, f) for f in frames] + [(False, None)]
        cap.read.side_effect = read_returns

        def set_pos(prop, val):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                # start_frame 이후부터 읽도록 side_effect 재설정
                remaining = [(True, f) for f in frames[int(val):]] + [(False, None)]
                cap.read.side_effect = remaining
        cap.set.side_effect = set_pos

        return cap

    def _solid_frame(self, value: int = 100) -> np.ndarray:
        """단색 BGR 프레임."""
        return np.full((480, 640, 3), value, dtype=np.uint8)

    def test_stationary_frames_no_swing(self):
        """동일한 프레임이 연속되면 스윙 탐지되지 않는다."""
        still = self._solid_frame(100)
        frames = [still] * 10
        cap = self._make_mock_cap(frames)

        result = _detect_swings(cap, 0, 9, flow_threshold=4.0, cooldown_frames=15,
                                flow_config=FlowConfig())
        assert result == []

    def test_large_motion_triggers_swing(self):
        """큰 변화가 있는 프레임 전환은 높은 magnitude를 만들어 스윙으로 탐지된다."""
        dark  = self._solid_frame(0)
        bright = self._solid_frame(200)
        # dark → bright → dark → bright ... (강한 전환 반복)
        frames = [dark, bright, dark, bright, dark, bright]
        cap = self._make_mock_cap(frames)

        result = _detect_swings(cap, 0, 5, flow_threshold=0.1, cooldown_frames=0,
                                flow_config=FlowConfig())
        assert len(result) > 0

    def test_cooldown_limits_detections(self):
        """cooldown_frames이 크면 연속 탐지가 억제된다."""
        dark  = self._solid_frame(0)
        bright = self._solid_frame(200)
        frames = [dark, bright, dark, bright, dark, bright, dark, bright]
        cap = self._make_mock_cap(frames)

        result_no_cd  = _detect_swings(cap, 0, 7, flow_threshold=0.1, cooldown_frames=0,
                                        flow_config=FlowConfig())
        cap2 = self._make_mock_cap(frames)
        result_with_cd = _detect_swings(cap2, 0, 7, flow_threshold=0.1, cooldown_frames=10,
                                         flow_config=FlowConfig())

        assert len(result_with_cd) <= len(result_no_cd)
