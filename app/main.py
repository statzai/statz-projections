import logging
import os
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.config import Config
from app.database import init_db_pool, close_db_pool
from app.source_database import source_init_db_pool, close_source_db_pool

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


@app.on_event("startup")
async def startup():
    try:
        await source_init_db_pool()
        logger.info("Source database connected")
    except Exception as e:
        logger.warning(f"Could not connect to source database: {e}")
    try:
        await init_db_pool()
        logger.info("Projections database connected")
    except Exception as e:
        logger.warning(f"Could not connect to database (projections will save to files only): {e}")

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

