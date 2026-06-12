import aiomysql
import logging
from typing import Optional

from app.config import Config

logger = logging.getLogger("projection")

source_pool: Optional[aiomysql.Pool] = None

async def source_init_db_pool():
    global source_pool
    if source_pool is None or source_pool._closed:
        source_pool = await aiomysql.create_pool(
            host=Config.SOURCE_DB_HOST,
            port=Config.SOURCE_DB_PORT,
            user=Config.SOURCE_DB_USER,
            password=Config.SOURCE_DB_PASSWORD,
            db=Config.SOURCE_DB_NAME,
            minsize=1,
            maxsize=3,
            autocommit=False,
            connect_timeout=10,
            pool_recycle=300,  # recycle connections older than 5 minutes
            # Pin every pooled connection to UTC. fixtures.kickoff_datetime
            # (and all stored datetimes) are UTC, but the DB host runs on
            # SYSTEM time = Europe/London, so a bare NOW() drifts +1h during
            # BST. That made `kickoff_datetime > NOW()` drop fixtures from the
            # "upcoming" projection window a full hour before kickoff (and the
            # mirror `< NOW()` lookback include them an hour early). Forcing the
            # session TZ to UTC makes NOW()/CURRENT_TIMESTAMP agree with the
            # stored UTC values year-round (GMT *and* BST).
            init_command="SET time_zone = '+00:00'",
        )

async def get_source_connection():
    if source_pool is None or source_pool._closed:
        await source_init_db_pool()
    conn = await source_pool.acquire()
    return conn

def release_source_connection(conn):
    """Safely release a connection back to the pool."""
    if conn is not None and source_pool is not None and not source_pool._closed:
        try:
            source_pool.release(conn)
        except Exception:
            pass

async def close_source_db_pool():
    global source_pool
    if source_pool:
        source_pool.close()
        await source_pool.wait_closed()
        source_pool = None

async def check_source_connection(conn):
    try:
        await conn.execute("SELECT 1")
        return True
    except Exception:
        return False
