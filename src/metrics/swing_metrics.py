import numpy as np
from src.pose.keypoint_schema import (
    NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST,
)

_CONF_THRESHOLD = 0.3


def _valid(kp: np.ndarray, idx: int) -> bool:
    return kp[idx, 2] >= _CONF_THRESHOLD


def _angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Horizontal angle (degrees) of the vector from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def calc_shoulder_rotation(kp: np.ndarray) -> float | None:
    """Angle of shoulder line relative to horizontal (degrees)."""
    if not (_valid(kp, L_SHOULDER) and _valid(kp, R_SHOULDER)):
        return None
    return _angle_deg(kp[R_SHOULDER, :2], kp[L_SHOULDER, :2])


def calc_hip_rotation(kp: np.ndarray) -> float | None:
    """Angle of hip line relative to horizontal (degrees)."""
    if not (_valid(kp, L_HIP) and _valid(kp, R_HIP)):
        return None
    return _angle_deg(kp[R_HIP, :2], kp[L_HIP, :2])


def calc_head_stability(kp_seq: list[np.ndarray]) -> float | None:
    """Std-dev of nose position (pixels) across a sequence of keypoint frames."""
    positions = [
        kp[NOSE, :2] for kp in kp_seq if _valid(kp, NOSE)
    ]
    if len(positions) < 2:
        return None
    arr = np.array(positions)
    return float(np.sqrt(np.var(arr[:, 0]) + np.var(arr[:, 1])))


def calc_wrist_trajectory(kp_seq: list[np.ndarray]) -> list[list[float]]:
    """
    (x, y) positions of the dominant wrist across frames.
    Picks whichever of L/R wrist has higher mean confidence in the sequence.
    """
    l_confs = [kp[L_WRIST, 2] for kp in kp_seq]
    r_confs = [kp[R_WRIST, 2] for kp in kp_seq]
    idx = L_WRIST if np.mean(l_confs) >= np.mean(r_confs) else R_WRIST
    return [
        kp[idx, :2].tolist()
        for kp in kp_seq
        if kp[idx, 2] >= _CONF_THRESHOLD
    ]


def calc_swing_timing(swing_frames: list[int], seg_start: int) -> int:
    """Frame offset of first swing relative to segment start. -1 if no swing."""
    if not swing_frames:
        return -1
    return swing_frames[0] - seg_start


def compute_segment_metrics(
    kp_map: dict[int, np.ndarray],
    swing_frames: list[int],
    seg_start: int,
) -> dict:
    """
    Compute all metrics for a single segment.
    kp_map: {frame_idx: ndarray(17,3)}
    """
    kp_seq = [kp_map[f] for f in sorted(kp_map)]

    # Use keypoint at first swing frame for rotation angles; fallback to first available
    anchor_frame = swing_frames[0] if swing_frames else (sorted(kp_map)[0] if kp_map else None)
    anchor_kp = kp_map.get(anchor_frame) if anchor_frame is not None else None

    return {
        "shoulder_rotation_deg": calc_shoulder_rotation(anchor_kp) if anchor_kp is not None else None,
        "hip_rotation_deg": calc_hip_rotation(anchor_kp) if anchor_kp is not None else None,
        "head_stability_px": calc_head_stability(kp_seq),
        "swing_timing_frame": calc_swing_timing(swing_frames, seg_start),
        "wrist_trajectory": calc_wrist_trajectory(kp_seq),
    }
