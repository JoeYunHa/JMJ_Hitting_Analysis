import os
import contextlib
from dotenv import load_dotenv

load_dotenv()

_DB_URL = os.environ.get("DB_URL")

_WOBA_QUERY = """
SELECT
    pwg.woba_roll10,
    pwg.woba_cumul,
    pwg.pa_game,
    ww.lg_woba,
    (
        SELECT COUNT(*)
        FROM player_woba_game pwg2
        JOIN game g2 ON g2.game_id = pwg2.game_id
        WHERE pwg2.player_code = pwg.player_code
          AND g2.season = g.season
          AND g2.game_date <= g.game_date
    ) AS game_seq
FROM player_woba_game pwg
JOIN game g ON g.game_id = pwg.game_id
JOIN woba_weights ww
    ON ww.season = g.season
   AND ww.league = pwg.league
   AND ww.method = 'hybrid'
WHERE pwg.player_code = %s
  AND g.game_date = %s
LIMIT 1;
"""

_PLAYER_CODE_QUERY = """
SELECT player_code FROM player WHERE player_name = %s LIMIT 1;
"""


@contextlib.contextmanager
def _connect():
    import psycopg2
    conn = psycopg2.connect(_DB_URL)
    try:
        yield conn
    finally:
        conn.close()


def get_player_code(player_name: str) -> str | None:
    if not _DB_URL:
        return None
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_PLAYER_CODE_QUERY, (player_name,))
                row = cur.fetchone()
                return str(row[0]) if row else None
    except Exception as e:
        print(f"[db_client] player_code lookup failed: {e}")
        return None


def get_woba_stats(game_date: str, player_code: str) -> dict | None:
    """
    Returns:
        woba_roll10: rolling 10-game wOBA (None if < 10 games played)
        woba_cumul:  season-to-date cumulative wOBA
        lg_woba:     league average wOBA for the season
        pa_game:     plate appearances on game_date
        early_season: True if fewer than 10 games played before this date
    """
    if not _DB_URL:
        return None
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_WOBA_QUERY, (player_code, game_date))
                row = cur.fetchone()
                if row is None:
                    return None
                woba_roll10, woba_cumul, pa_game, lg_woba, game_seq = row
                return {
                    "woba_roll10": float(woba_roll10) if woba_roll10 is not None else None,
                    "woba_cumul": float(woba_cumul) if woba_cumul is not None else None,
                    "pa_game": int(pa_game) if pa_game is not None else 0,
                    "lg_woba": float(lg_woba),
                    "early_season": int(game_seq) < 10,
                }
    except Exception as e:
        print(f"[db_client] woba_stats query failed: {e}")
        return None
