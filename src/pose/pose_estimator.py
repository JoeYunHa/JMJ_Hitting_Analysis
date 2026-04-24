import cv2
import numpy as np
from src.utils.video import open_video

_estimator = None

# Keypoints with visibility score below this are zeroed out.
_KP_CONF_THRESHOLD = 0.3
# Minimum overlap ratio (intersection / box area) to consider a person "in zone".
_OVERLAP_THRESHOLD = 0.3


def get_estimator() -> "YoloPoseEstimator":
    global _estimator
    if _estimator is None:
        _estimator = YoloPoseEstimator()
    return _estimator


class YoloPoseEstimator:
    def __init__(self, model_name: str = "yolov8m-pose.pt", conf: float = 0.5):
        from ultralytics import YOLO
        self._model = YOLO(model_name)
        self._conf = conf

    def extract_keypoints(
        self,
        video_path: str,
        frame_indices: list[int],
        batter_zone_roi: dict | None = None,
    ) -> dict[int, np.ndarray]:
        """
        Run pose estimation on specified frames only.
        Returns {frame_idx: ndarray(17, 3)} where columns are (x, y, confidence).
        Keypoints with confidence below _KP_CONF_THRESHOLD are zeroed out.
        """
        if not frame_indices:
            return {}

        targets = sorted(set(frame_indices))
        result: dict[int, np.ndarray] = {}

        with open_video(video_path) as cap:
            for idx in targets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                kp = self._infer_frame(frame, batter_zone_roi)
                if kp is not None:
                    result[idx] = kp

        return result

    def _infer_frame(
        self, frame: np.ndarray, zone_roi: dict | None
    ) -> np.ndarray | None:
        results = self._model(frame, conf=self._conf, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        kp_data = results[0].keypoints.data  # (N, 17, 3)
        boxes = results[0].boxes

        if len(kp_data) == 0:
            return None

        if len(kp_data) == 1:
            chosen = 0
        else:
            chosen = _select_batter(boxes, zone_roi)

        kp = kp_data[chosen].cpu().numpy()  # (17, 3)
        if kp.shape != (17, 3):
            return None

        # Zero out low-confidence keypoints so downstream code can skip them.
        kp[kp[:, 2] < _KP_CONF_THRESHOLD] = 0.0
        return kp


def _overlap_ratio(xyxy: np.ndarray, zx1: float, zy1: float, zx2: float, zy2: float) -> np.ndarray:
    """Intersection area / bounding-box area for each detection."""
    ix1 = np.maximum(xyxy[:, 0], zx1)
    iy1 = np.maximum(xyxy[:, 1], zy1)
    ix2 = np.minimum(xyxy[:, 2], zx2)
    iy2 = np.minimum(xyxy[:, 3], zy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    box_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return np.where(box_area > 0, inter / box_area, 0.0)


def _select_batter(boxes, zone_roi: dict | None) -> int:
    """Return index of the batter among YOLO detections.

    Strategy:
      1. If zone_roi given, keep only detections whose bounding box overlaps
         the zone by at least _OVERLAP_THRESHOLD.
      2. Among candidates, pick the largest bounding box — the batter is
         closest to the camera and therefore largest (catcher/umpire are
         farther away or partially occluded).
      3. If no detection passes the overlap threshold, fall back to the
         largest box across all detections.
    """
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    if zone_roi is not None:
        zx1, zy1 = float(zone_roi["x"]), float(zone_roi["y"])
        zx2, zy2 = zx1 + zone_roi["w"], zy1 + zone_roi["h"]
        overlaps = _overlap_ratio(xyxy, zx1, zy1, zx2, zy2)
        in_zone = np.where(overlaps >= _OVERLAP_THRESHOLD)[0]
        if len(in_zone) > 0:
            return int(in_zone[areas[in_zone].argmax()])

    return int(areas.argmax())
