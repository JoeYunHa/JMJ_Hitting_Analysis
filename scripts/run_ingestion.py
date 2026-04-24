from src.ingestion import download_video
from src.ingestion.youtube_search import search_videos

for v in search_videos():
    path = download_video(url=v["url"], date=v["date"])
    print(path)
