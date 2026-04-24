from dataclasses import dataclass


@dataclass
class ChannelDetectionParams:
    threshold: float = 0.8
    sample_interval_sec: float = 5.0
    max_sample_sec: float = 60.0
    early_exit_confidence: float = 0.95


@dataclass
class SwingDetectionParams:
    flow_threshold: float = 4.0
    cooldown_frames: int = 15


@dataclass
class OcrExtractionParams:
    sample_fps: int = 2
    ocr_delay_sec: float = 3.0
    gap_tolerance_sec: float = 5.0


@dataclass
class PoseEstimationParams:
    model_name: str = "yolov8m-pose.pt"
    conf: float = 0.5
    kp_conf_threshold: float = 0.3
    overlap_threshold: float = 0.3


@dataclass
class Phase4Params:
    swing_window: int = 30
