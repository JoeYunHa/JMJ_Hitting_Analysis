def tag_performance(stats: dict | None) -> str:
    """
    방안 B: rolling wOBA 10경기 vs 리그 평균 wOBA 비교.
    woba_roll10이 없으면 woba_cumul(시즌 누적)로 fallback.

    추후 방안 D 확장 시 이 함수만 수정:
        hot        woba >= lg_woba * 1.15
        good_period woba >= lg_woba
        bad_period  woba >= lg_woba * 0.85
        slump       woba <  lg_woba * 0.85
    """
    if stats is None:
        return "unknown"

    woba = stats.get("woba_roll10") or stats.get("woba_cumul")
    if woba is None:
        return "unknown"

    lg_woba = stats["lg_woba"]
    return "good_period" if woba >= lg_woba else "bad_period"
