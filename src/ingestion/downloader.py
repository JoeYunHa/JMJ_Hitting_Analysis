import subprocess
from pathlib import Path
from configs.settings import DATA_DIR
from src.ingestion.tagger import tag_video_filename

DATA_DIR.mkdir(exist_ok=True)


def download_video(url: str, date: str | None = None) -> Path | None:
    """Try yt-dlp download; skip if already exists; fall back to manual path on failure."""
    output_template = str(DATA_DIR / "%(upload_date)s_%(title).50s.%(ext)s")

    # Skip if already downloaded
    result_check = subprocess.run(
        ["yt-dlp", "--no-playlist", "--print", "filename", "-o", output_template, url],
        capture_output=True,
        text=True,
    )
    expected = Path(result_check.stdout.strip())
    if expected.exists():
        print(f"[skip] {expected.name}")
        return expected

    result = subprocess.run(
        ["yt-dlp", "--no-playlist", "-o", output_template, url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[yt-dlp failed] {result.stderr.strip()}")
        return _manual_fallback(date)

    files = sorted(DATA_DIR.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None

    downloaded = files[0]
    tagged = tag_video_filename(downloaded, date)
    print(f"[downloaded] {tagged}")
    return tagged


def _manual_fallback(date: str | None = None) -> Path | None:
    """Prompt user to provide a local video path."""
    path_str = input("Enter local video file path: ").strip()
    src = Path(path_str)

    if not src.exists():
        print(f"[error] File not found: {src}")
        return None

    tagged = tag_video_filename(src, date, copy_to_data=True)
    print(f"[manual] {tagged}")
    return tagged
