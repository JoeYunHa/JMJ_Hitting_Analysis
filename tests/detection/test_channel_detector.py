"""
Tests for src/detection/channel_detector.py

load_all_configs() — 실제 configs/channels/ JSON 파일 로드
detect_channel()   — 실제 샘플 프레임으로 템플릿 매칭 실행
                     (로고 템플릿이 없으므로 _template_img=None → None 반환이 정상)
"""
import numpy as np
import pytest

from src.detection.channel_detector import load_all_configs, detect_channel

# 등록된 config_id 목록 (configs/channels/ 실제 파일 기준)
EXPECTED_CONFIG_IDS = {
    "kbo_kbsn_2026_game_highlight",
    "kbo_spotv_2026_game_highlight",
    "kbo_mbcsports_2026_game_highlight",
    "tving_2026_game_highlight",
}

REQUIRED_ROI_KEYS = {"batter_name_roi"}
REQUIRED_ROI_FIELDS = {"x", "y", "w", "h"}


# ---------------------------------------------------------------------------
# load_all_configs
# ---------------------------------------------------------------------------

class TestLoadAllConfigs:

    def test_all_expected_configs_loaded(self, real_configs):
        """4개 채널 config가 모두 로드된다."""
        assert EXPECTED_CONFIG_IDS.issubset(real_configs.keys()), (
            f"누락된 config: {EXPECTED_CONFIG_IDS - real_configs.keys()}"
        )

    def test_config_id_key_matches_filename_field(self, real_configs):
        """dict 키와 JSON 내 config_id 값이 일치한다."""
        for key, cfg in real_configs.items():
            assert cfg["config_id"] == key

    def test_batter_name_roi_shape(self, real_configs):
        """batter_name_roi가 x/y/w/h 필드를 모두 갖는다."""
        for cfg_id, cfg in real_configs.items():
            roi = cfg.get("batter_name_roi")
            assert roi is not None, f"{cfg_id}: batter_name_roi 없음"
            assert REQUIRED_ROI_FIELDS.issubset(roi.keys()), (
                f"{cfg_id}: batter_name_roi 필드 누락 — {roi}"
            )

    def test_roi_values_are_positive(self, real_configs):
        """ROI 좌표와 크기가 모두 양수이다."""
        for cfg_id, cfg in real_configs.items():
            roi = cfg["batter_name_roi"]
            for field in REQUIRED_ROI_FIELDS:
                assert roi[field] >= 0, f"{cfg_id}.batter_name_roi.{field} < 0"
            assert roi["w"] > 0, f"{cfg_id}: batter_name_roi.w == 0"
            assert roi["h"] > 0, f"{cfg_id}: batter_name_roi.h == 0"

    def test_roi_within_frame_resolution(self, real_configs):
        """batter_name_roi 영역이 frame_resolution 범위를 벗어나지 않는다."""
        for cfg_id, cfg in real_configs.items():
            res = cfg.get("frame_resolution")
            if res is None:
                continue
            roi = cfg["batter_name_roi"]
            assert roi["x"] + roi["w"] <= res["w"], (
                f"{cfg_id}: batter_name_roi 가로 범위 초과"
            )
            assert roi["y"] + roi["h"] <= res["h"], (
                f"{cfg_id}: batter_name_roi 세로 범위 초과"
            )

    def test_template_img_loaded_when_file_present(self, real_configs):
        """로고 템플릿 파일이 존재하면 _template_img가 numpy array이다."""
        import numpy as np
        from pathlib import Path
        for cfg_id, cfg in real_configs.items():
            tmpl_path = cfg.get("logo_template")
            tmpl_img = cfg.get("_template_img")
            if tmpl_path and Path(tmpl_path).exists():
                assert isinstance(tmpl_img, np.ndarray), (
                    f"{cfg_id}: 템플릿 파일이 있는데 _template_img가 ndarray가 아님"
                )
            else:
                assert tmpl_img is None, (
                    f"{cfg_id}: 템플릿 파일이 없는데 _template_img가 None이 아님"
                )


# ---------------------------------------------------------------------------
# detect_channel — 실제 샘플 프레임 사용
# ---------------------------------------------------------------------------

class TestDetectChannelWithRealFrames:

    def test_detects_kbsn_from_kbsn_frame(self, kbsn_frame, real_configs):
        """KBSN 샘플 프레임에서 kbo_kbsn_2026_game_highlight가 검출된다."""
        result = detect_channel(kbsn_frame, real_configs, threshold=0.5)
        assert result == "kbo_kbsn_2026_game_highlight", (
            f"예상 채널 kbo_kbsn_2026_game_highlight, 실제: {result}"
        )

    def test_accepts_any_bgr_frame_without_crash(self, spotv_frame, real_configs):
        """BGR 프레임 입력 시 예외 없이 실행된다."""
        try:
            detect_channel(spotv_frame, real_configs)
        except Exception as e:
            pytest.fail(f"detect_channel 예외 발생: {e}")

    def test_handles_empty_configs(self, kbsn_frame):
        """configs가 빈 dict이면 None을 반환한다."""
        assert detect_channel(kbsn_frame, {}) is None

    def test_threshold_too_high_returns_none(self, kbsn_frame, real_configs):
        """threshold=1.0이면 완전 일치가 아닌 한 None이다."""
        assert detect_channel(kbsn_frame, real_configs, threshold=1.0) is None

    def test_threshold_zero_returns_best_match(self, kbsn_frame, real_configs):
        """threshold=0.0이면 템플릿 있을 때 best match가 반환된다."""
        result = detect_channel(kbsn_frame, real_configs, threshold=0.0)
        assert result is not None
