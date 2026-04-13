import logging
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.config import Config
from app.database import init_db_pool, close_db_pool
from app.source_database import source_init_db_pool, close_source_db_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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

