import shutil
import re
from pathlib import Path
from datetime import datetime
from configs.settings import DATA_DIR

DATE_PREFIX_RE = re.compile(r"^\d{8}_")  # YYYYMMDD_ already prefixed


def tag_video_filename(
    path: Path, date: str | None = None, copy_to_data: bool = False
) -> Path:
    """
    Ensure filename starts with YYYYMMDD_.
    date: override string 'YYYYMMDD'; if None, parse from filename or use today.
    """
    dest_dir = DATA_DIR
    dest_dir.mkdir(exist_ok=True)

    name = path.stem
    ext = path.suffix

    if DATE_PREFIX_RE.match(name):
        tagged_name = name  # already tagged
    else:
        if date:
            d = date  # user-provided 'YYYYMMDD'
        elif len(name) >= 8 and name[:8].isdigit():
            d = name[:8]  # yt-dlp already bakes upload_date into stem
        else:
            d = datetime.today().strftime("%Y%m%d")
        tagged_name = f"{d}_{name}"

    dest = dest_dir / f"{tagged_name}{ext}"

    if copy_to_data and path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
    elif path.resolve() != dest.resolve():
        path.rename(dest)

    return dest
