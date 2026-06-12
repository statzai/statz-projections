import logging
import os
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.config import Config
from app.database import init_db_pool, close_db_pool, get_connection
from app.source_database import (
    source_init_db_pool,
    close_source_db_pool,
    get_source_connection,
    release_source_connection,
)

# Dual logging: stdout (for `docker compose logs`) AND a rotating file in the
# bind-mounted data volume. The file survives container rebuilds, so the
# admin panel's "Logs" view and the daily digest can see history that
# predates the current container. Rotates daily, keeps 14 days.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = "/app/app/data"
_LOG_FILE = os.path.join(_LOG_DIR, "projection.log")

os.makedirs(_LOG_DIR, exist_ok=True)

_file_handler = TimedRotatingFileHandler(
    _LOG_FILE, when="midnight", backupCount=14, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATEFMT,
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger("main")

app = FastAPI(
    title="My First FastAPI App",
    version="1.0.0"
)


app.include_router(api_router)


async def _assert_utc_session(acquire, release, label):
    """Fail loud at startup if a pool's MySQL session isn't UTC.

    All stored datetimes (fixtures.kickoff_datetime etc.) are UTC, so a
    session whose NOW()/CURRENT_TIMESTAMP drifts from UTC silently skews every
    `kickoff_datetime > NOW()` window — the BST bug that dropped the WC opener
    from the projection an hour early. Both pools set time_zone='+00:00' on
    connect; this is the regression guard that proves it stuck. Raises rather
    than warns: a wrong TZ corrupts data quietly for a whole season, so a
    crash-loop (restart: unless-stopped) is the signal we want, not a buried log.
    """
    conn = await acquire()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT TIMESTAMPDIFF(SECOND, UTC_TIMESTAMP(), NOW())")
            drift = (await cur.fetchone())[0]
    finally:
        release(conn)
    if drift != 0:
        raise RuntimeError(
            f"{label} session is NOT UTC: NOW() drifts {drift}s from UTC. "
            f"Expected time_zone='+00:00'. Refusing to start — projection windows "
            f"would be skewed (see app/source_database.py)."
        )
    logger.info(f"{label} session timezone verified UTC")


@app.on_event("startup")
async def startup():
    # Connect each pool. A DB that's simply unreachable is non-fatal (the
    # service degrades to file-only output) — but the UTC assertion below is
    # deliberately OUTSIDE these handlers so a misconfigured-but-reachable DB
    # raises and crash-loops instead of being swallowed as a connect warning.
    source_up = False
    write_up = False
    try:
        await source_init_db_pool()
        source_up = True
        logger.info("Source database connected")
    except Exception as e:
        logger.warning(f"Could not connect to source database: {e}")
    try:
        await init_db_pool()
        write_up = True
        logger.info("Projections database connected")
    except Exception as e:
        logger.warning(f"Could not connect to database (projections will save to files only): {e}")

    if source_up:
        await _assert_utc_session(
            get_source_connection, release_source_connection, "Source DB"
        )
    if write_up:
        from app.database import pool as _write_pool
        await _assert_utc_session(get_connection, _write_pool.release, "Projections DB")

@app.on_event("shutdown")
async def shutdown():
    try:
        await close_source_db_pool()
    except Exception:
        pass
    try:
        await close_db_pool()
    except Exception:
        pass

