import logging
import cv2
import numpy as np

from src.utils.video import open_video
from src.config.params import PoseEstimationParams

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = PoseEstimationParams()


def get_estimator() -> "YoloPoseEstimator":
    """Return the thread-safe YoloPoseEstimator singleton via model_manager."""
    from src.models.model_manager import get_pose_estimator
    return get_pose_estimator()


class YoloPoseEstimator:
    def __init__(
        self,
        model_name: str = _DEFAULT_PARAMS.model_name,
        conf: float = _DEFAULT_PARAMS.conf,
        params: PoseEstimationParams | None = None,
    ):
        from ultralytics import YOLO
        self._params = params or PoseEstimationParams(model_name=model_name, conf=conf)
        logger.info(f"Loading YOLO model: {self._params.model_name}")
        self._model = YOLO(self._params.model_name)
        self._conf = self._params.conf

    def extract_keypoints(
        self,
        video_path: str,
        frame_indices: list[int],
        batter_zone_roi: dict | None = None,
    ) -> dict[int, np.ndarray]:
        """
        Run pose estimation on specified frames only.
        Returns {frame_idx: ndarray(17, 3)} where columns are (x, y, confidence).
        Keypoints with confidence below kp_conf_threshold are zeroed out.
        """
        if not frame_indices:
            return {}

        targets = sorted(set(frame_indices))
        result: dict[int, np.ndarray] = {}

        with open_video(video_path) as cap:
            for idx in targets:
                if idx < 0:
                    logger.warning(f"Skipping negative frame index {idx}")
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Cannot read frame {idx} from {video_path}")
                    continue
                try:
                    kp = self._infer_frame(frame, batter_zone_roi)
                except Exception as e:
                    logger.error(f"Inference failed at frame {idx}: {e}")
                    continue
                if kp is not None:
                    result[idx] = kp

        logger.info(
            f"Keypoint extraction: {len(result)}/{len(targets)} frames succeeded "
            f"for {video_path}"
        )
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

        chosen = 0 if len(kp_data) == 1 else _select_batter(
            boxes, zone_roi, self._params.overlap_threshold
        )

        kp = kp_data[chosen].cpu().numpy()  # (17, 3)
        if kp.shape != (17, 3):
            return None

        kp[kp[:, 2] < self._params.kp_conf_threshold] = 0.0
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


def _select_batter(boxes, zone_roi: dict | None, overlap_threshold: float) -> int:
    """Return index of the batter among YOLO detections.

    Strategy:
      1. If zone_roi given, keep only detections whose bounding box overlaps
         the zone by at least overlap_threshold.
      2. Among candidates, pick the largest bounding box.
      3. If no detection passes the overlap threshold, fall back to the
         largest box across all detections.
    """
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    if zone_roi is not None:
        zx1, zy1 = float(zone_roi["x"]), float(zone_roi["y"])
        zx2, zy2 = zx1 + zone_roi["w"], zy1 + zone_roi["h"]
        overlaps = _overlap_ratio(xyxy, zx1, zy1, zx2, zy2)
        in_zone = np.where(overlaps >= overlap_threshold)[0]
        if len(in_zone) > 0:
            return int(in_zone[areas[in_zone].argmax()])

    return int(areas.argmax())
