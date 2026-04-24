import os
import re
import threading
import logging
import contextlib
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_DB_URL = os.environ.get("DB_URL")
_pool = None
_pool_lock = threading.Lock()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                import psycopg2.pool
                logger.info("Initializing DB connection pool...")
                _pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=5,
                    dsn=_DB_URL,
                    connect_timeout=10,
                )
                logger.info("DB connection pool ready.")
    return _pool


@contextlib.contextmanager
def _connect():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


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
        logger.error(f"player_code lookup failed for '{player_name}': {e}")
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
    if not _DATE_RE.match(game_date):
        logger.error(f"Invalid game_date format '{game_date}', expected YYYY-MM-DD")
        return None
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_WOBA_QUERY, (player_code, game_date))
                row = cur.fetchone()
                if row is None:
                    logger.info(f"No wOBA stats for player={player_code} date={game_date}")
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
        logger.error(f"woba_stats query failed for player={player_code} date={game_date}: {e}")
        return None
