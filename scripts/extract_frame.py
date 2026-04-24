# scripts/extract_frame.py
# Extract a specific frame from video by timestamp

import cv2
import argparse
from pathlib import Path


def extract_frame(video_path: str, timestamp: float, out_path: str) -> None:
    """
    timestamp: seconds (e.g. 63.5 = 1min 3.5sec)
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    print(
        f"FPS: {fps:.2f} | Duration: {duration:.2f}s | Resolution: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    target_frame = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    assert ret, f"Failed to read frame at {timestamp}s"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Saved -> {out_path}")
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument(
        "--timestamp", required=True, type=float, help="Timestamp in seconds"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output image path (e.g. data/sample_tving_2025.png)",
    )
    args = parser.parse_args()

    extract_frame(args.video, args.timestamp, args.out)


if __name__ == "__main__":
    main()
