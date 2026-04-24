from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from configs.settings import YOUTUBE_API_KEY, SEARCH_CONFIG


def search_videos(
    playlist_id: str = SEARCH_CONFIG["playlist_id"],
    filter_keyword: str = SEARCH_CONFIG["filter_keyword"],
) -> list[dict]:
    """Fetch all playlist videos with pagination, filtered by keyword in title."""
    yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    results = []
    next_page_token = None
    seen_ids = set()

    while True:
        try:
            resp = (
                yt.playlistItems()
                .list(
                    playlistId=playlist_id,
                    part="snippet",
                    maxResults=50,
                    pageToken=next_page_token,
                )
                .execute()
            )
        except HttpError as e:
            raise RuntimeError(
                f"YouTube API error (status {e.resp.status}): {e}"
            ) from e

        for item in resp.get("items", []):
            video_id = item["snippet"]["resourceId"]["videoId"]
            title = item["snippet"]["title"]
            if filter_keyword in title and video_id not in seen_ids:
                seen_ids.add(video_id)
                results.append(
                    {
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "title": title,
                        "date": item["snippet"]["publishedAt"][:10].replace("-", ""),
                    }
                )

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return results
