"""
Tests for src/detection/ocr_extractor.py

roi_to_xyxy / crop_roi — 실제 config ROI + 실제 샘플 프레임
ocr_contains_name      — 실제 PaddleOCR 실행 (requires paddleocr)
extract_batter_segments — 실제 영상 파일 (느린 통합 테스트, --slow 옵션으로 실행)
"""
import numpy as np
import pytest
import cv2

from src.detection.ocr_extractor import roi_to_xyxy, crop_roi, ocr_contains_name
from src.detection.channel_detector import load_all_configs


# ---------------------------------------------------------------------------
# roi_to_xyxy
# ---------------------------------------------------------------------------

class TestRoiToXyxy:

    def test_kbsn_batter_roi_conversion(self, real_configs):
        """KBSN batter_name_roi {x,y,w,h} → (x1,y1,x2,y2) 변환이 정확하다."""
        roi = real_configs["kbo_kbsn_2026_game_highlight"]["batter_name_roi"]
        x1, y1, x2, y2 = roi_to_xyxy(roi)
        assert x1 == roi["x"]
        assert y1 == roi["y"]
        assert x2 == roi["x"] + roi["w"]
        assert y2 == roi["y"] + roi["h"]

    def test_spotv_batter_roi_conversion(self, real_configs):
        roi = real_configs["kbo_spotv_2026_game_highlight"]["batter_name_roi"]
        x1, y1, x2, y2 = roi_to_xyxy(roi)
        assert x2 - x1 == roi["w"]
        assert y2 - y1 == roi["h"]

    def test_mbcsports_batter_roi_conversion(self, real_configs):
        roi = real_configs["kbo_mbcsports_2026_game_highlight"]["batter_name_roi"]
        x1, y1, x2, y2 = roi_to_xyxy(roi)
        assert x2 > x1
        assert y2 > y1

    @pytest.mark.parametrize("cfg_id", [
        "kbo_kbsn_2026_game_highlight",
        "kbo_spotv_2026_game_highlight",
        "kbo_mbcsports_2026_game_highlight",
        "tving_2026_game_highlight",
    ])
    def test_all_configs_roi_positive_area(self, cfg_id, real_configs):
        """모든 채널의 batter_name_roi 변환 결과가 양의 넓이를 가진다."""
        roi = real_configs[cfg_id]["batter_name_roi"]
        x1, y1, x2, y2 = roi_to_xyxy(roi)
        assert (x2 - x1) * (y2 - y1) > 0


# ---------------------------------------------------------------------------
# crop_roi — 실제 샘플 프레임 크롭
# ---------------------------------------------------------------------------

class TestCropRoi:

    def test_kbsn_crop_shape_matches_roi(self, kbsn_frame, real_configs):
        """KBSN 프레임에서 batter_name_roi 크롭 결과 크기가 ROI w×h와 일치한다."""
        roi = real_configs["kbo_kbsn_2026_game_highlight"]["batter_name_roi"]
        cropped = crop_roi(kbsn_frame, roi)
        assert cropped.shape[0] == roi["h"], f"높이 불일치: {cropped.shape[0]} != {roi['h']}"
        assert cropped.shape[1] == roi["w"], f"너비 불일치: {cropped.shape[1]} != {roi['w']}"

    def test_spotv_crop_shape_matches_roi(self, spotv_frame, real_configs):
        roi = real_configs["kbo_spotv_2026_game_highlight"]["batter_name_roi"]
        cropped = crop_roi(spotv_frame, roi)
        assert cropped.shape[0] == roi["h"]
        assert cropped.shape[1] == roi["w"]

    def test_mbcsports_crop_shape_matches_roi(self, mbcsports_frame, real_configs):
        roi = real_configs["kbo_mbcsports_2026_game_highlight"]["batter_name_roi"]
        cropped = crop_roi(mbcsports_frame, roi)
        assert cropped.shape[0] == roi["h"]
        assert cropped.shape[1] == roi["w"]

    def test_crop_returns_numpy_array(self, kbsn_frame, real_configs):
        roi = real_configs["kbo_kbsn_2026_game_highlight"]["batter_name_roi"]
        result = crop_roi(kbsn_frame, roi)
        assert isinstance(result, np.ndarray)

    def test_crop_is_not_empty(self, kbsn_frame, real_configs):
        """크롭된 영역이 비어 있지 않다 (픽셀 값이 존재한다)."""
        roi = real_configs["kbo_kbsn_2026_game_highlight"]["batter_name_roi"]
        cropped = crop_roi(kbsn_frame, roi)
        assert cropped.size > 0

    def test_crop_does_not_modify_original_frame(self, kbsn_frame, real_configs):
        """crop_roi가 원본 프레임을 변경하지 않는다."""
        original_copy = kbsn_frame.copy()
        roi = real_configs["kbo_kbsn_2026_game_highlight"]["batter_name_roi"]
        crop_roi(kbsn_frame, roi)
        assert np.array_equal(kbsn_frame, original_copy)


# ---------------------------------------------------------------------------
# ocr_contains_name — 실제 PaddleOCR 실행
# (PaddleOCR 설치 필요, 첫 실행 시 모델 다운로드 발생)
# ---------------------------------------------------------------------------

@pytest.mark.ocr
class TestOcrContainsName:

    def test_kbsn_frame_does_not_crash(self, kbsn_frame, real_configs):
        """KBSN 샘플 프레임에서 OCR이 예외 없이 실행된다."""
        cfg = real_configs["kbo_kbsn_2026_game_highlight"]
        try:
            ocr_contains_name(kbsn_frame, cfg["batter_name_roi"])
        except Exception as e:
            pytest.fail(f"OCR 예외: {e}")

    def test_spotv_frame_does_not_crash(self, spotv_frame, real_configs):
        cfg = real_configs["kbo_spotv_2026_game_highlight"]
        try:
            ocr_contains_name(spotv_frame, cfg["batter_name_roi"])
        except Exception as e:
            pytest.fail(f"OCR 예외: {e}")

    def test_mbcsports_frame_does_not_crash(self, mbcsports_frame, real_configs):
        cfg = real_configs["kbo_mbcsports_2026_game_highlight"]
        try:
            ocr_contains_name(mbcsports_frame, cfg["batter_name_roi"])
        except Exception as e:
            pytest.fail(f"OCR 예외: {e}")

    def test_returns_bool(self, kbsn_frame, real_configs):
        """ocr_contains_name의 반환값이 bool이다."""
        cfg = real_configs["kbo_kbsn_2026_game_highlight"]
        result = ocr_contains_name(kbsn_frame, cfg["batter_name_roi"])
        assert isinstance(result, bool)

    def test_custom_target_not_in_random_region(self, kbsn_frame):
        """존재하지 않는 이름(가짜 문자열)은 False를 반환해야 한다."""
        roi = {"x": 0, "y": 0, "w": 10, "h": 10}  # 거의 빈 구석 크롭
        result = ocr_contains_name(kbsn_frame, roi, target="ZZZZNONEXISTENT")
        assert result is False


# ---------------------------------------------------------------------------
# extract_batter_segments — 실제 영상 파일 통합 테스트
# 실행: pytest -m slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestExtractBatterSegmentsIntegration:

    def test_returns_list(self, first_video_path, real_configs):
        """extract_batter_segments가 list를 반환한다."""
        from src.detection.ocr_extractor import extract_batter_segments

        # kbsn config 사용 (첫 영상이 kbsn 방송사 가정)
        cfg = real_configs["kbo_kbsn_2026_game_highlight"]
        result = extract_batter_segments(first_video_path, cfg)
        assert isinstance(result, list)

    def test_segments_have_valid_frame_range(self, first_video_path, real_configs):
        """각 세그먼트의 start_frame <= end_frame이다."""
        from src.detection.ocr_extractor import extract_batter_segments

        cfg = real_configs["kbo_kbsn_2026_game_highlight"]
        segments = extract_batter_segments(first_video_path, cfg)
        for seg in segments:
            assert seg.start_frame <= seg.end_frame, (
                f"비정상 구간: {seg.start_frame} > {seg.end_frame}"
            )

    def test_segments_time_consistent_with_frames(self, first_video_path, real_configs):
        """start_sec < end_sec 이고 프레임과 시간이 순서가 일치한다."""
        from src.detection.ocr_extractor import extract_batter_segments

        cfg = real_configs["kbo_kbsn_2026_game_highlight"]
        segments = extract_batter_segments(first_video_path, cfg)
        for seg in segments:
            assert seg.start_sec <= seg.end_sec
