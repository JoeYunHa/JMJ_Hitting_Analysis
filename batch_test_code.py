# scripts/run_phase3_4.py
from pathlib import Path
from src.detection.segment_pipeline import run_phase3
from src.metrics.phase4_pipeline import run_phase4

VIDEO_DIR = Path("data/raw_data")
OUTPUT_DIR = Path("outputs/json")

videos = sorted(VIDEO_DIR.glob("*.webm"))
print(f"총 {len(videos)}개 영상 처리 시작\n")

for i, video in enumerate(videos, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(videos)}] {video.name}")
    try:
        p3_json = run_phase3(str(video), output_dir=str(OUTPUT_DIR))
        run_phase4(
            phase3_json=p3_json, video_path=str(video), output_dir=str(OUTPUT_DIR)
        )
    except Exception as e:
        print(f"  [ERROR] {video.name}: {e}")
        continue
