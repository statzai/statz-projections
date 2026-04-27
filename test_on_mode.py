"""End-to-end test: run one league projection in USE_DB_LOADER=on mode.

Bypasses the running gunicorn workers — instantiates ProjectionService
directly, patches Config.USE_DB_LOADER, runs the full projections() flow.
Verifies the loader path can produce a complete projection without
crashing. Outputs land in DB / projection-outputs as a normal run.

Run inside the container so it shares the source DB pool config:
    docker compose exec statz-projection python test_on_mode.py "La Liga"
"""

import asyncio
import logging
import sys
import time
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_on_mode")


async def main(league: str) -> int:
    from app.config import Config
    from app.source_database import source_init_db_pool, close_source_db_pool
    from app.database import init_db_pool, close_db_pool
    from app.services.projection_service import ProjectionService

    Config.USE_DB_LOADER = "on"
    logger.info(f"Forcing USE_DB_LOADER=on for this test run")

    await source_init_db_pool()
    await init_db_pool()
    try:
        svc = ProjectionService()
        # projection_service.projections takes a request-like object
        # with .league attribute
        request = SimpleNamespace(league=league)
        t0 = time.time()
        result = await svc.projections(request)
        elapsed = time.time() - t0
        logger.info(f"projections() returned in {elapsed:.1f}s")
        logger.info(f"Result: {result}")
        return 0
    except Exception:
        logger.exception("projections() failed")
        return 1
    finally:
        await close_source_db_pool()
        await close_db_pool()


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "La Liga"
    sys.exit(asyncio.run(main(league)))
