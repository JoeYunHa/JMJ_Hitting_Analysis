from src.metrics.phase4_pipeline import run_phase4

run_phase4(
    phase3_json="outputs/json/20260328_[롯데자이언츠 vs 삼성라이온즈] 3.28(토) 야구 하이라이트｜2026 KBO리그｜KB_segments.json",
    video_path="data/raw_data/20260328_[롯데자이언츠 vs 삼성라이온즈] 3.28(토) 야구 하이라이트｜2026 KBO 리그｜KB.webm",
    output_dir="outputs/json",
)
