import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]

SEARCH_CONFIG = {
    "playlist_id": "PLuY-NTS_5Ipz7qggSm4O0aTp4RQjNCLCl",
    "max_results": 50,
    "filter_keyword": "롯데",  # filter by title
}

DATA_DIR = Path("data")
TARGET_NAME = "전민재"
GAP_TOLERANCE_SEC = 5.0
