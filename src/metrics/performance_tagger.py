import logging

logger = logging.getLogger(__name__)


def tag_performance(stats: dict | None) -> dict:
    """
    방안 B: rolling wOBA 10경기 vs 리그 평균 wOBA 비교.
    woba_roll10이 없으면 woba_cumul(시즌 누적)로 fallback하되, 어느 값을 사용했는지 명시.

    Returns:
        {
            "tag": "good_period" | "bad_period" | "unknown",
            "woba_source": "rolling_10" | "cumulative_season" | None,
            "woba": float | None,
        }

    추후 방안 D 확장 시 이 함수만 수정:
        hot         woba >= lg_woba * 1.15
        good_period woba >= lg_woba
        bad_period  woba >= lg_woba * 0.85
        slump       woba <  lg_woba * 0.85
    """
    if stats is None:
        return {"tag": "unknown", "woba_source": None, "woba": None}

    woba_roll10 = stats.get("woba_roll10")
    woba_cumul = stats.get("woba_cumul")

    if woba_roll10 is not None:
        woba = woba_roll10
        source = "rolling_10"
    elif woba_cumul is not None:
        woba = woba_cumul
        source = "cumulative_season"
        logger.info("woba_roll10 unavailable; falling back to cumulative season wOBA.")
    else:
        return {"tag": "unknown", "woba_source": None, "woba": None}

    lg_woba = stats["lg_woba"]
    tag = "good_period" if woba >= lg_woba else "bad_period"
    return {"tag": tag, "woba_source": source, "woba": woba}
