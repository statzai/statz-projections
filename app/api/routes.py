import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import requests as http_requests

class AllLeaguesRequest(BaseModel):
    leagues: Optional[List[str]] = None

from app.services.premier_league_projections_service import PremierLeagueProjectionsService
from app.services.projection_service import ProjectionService
from app.services.euro_comp_projection_service import EuroCompProjectionService
from app.models.requests.league_request import LeagueRequest
from app.services.projection_all_teams_service import ProjectionAllTeams
from app.services.fetch_all_data_service import FetchAllDataService

router = APIRouter(prefix="/api/projections", tags=["API"])
logger = logging.getLogger("routes")

projection_service = ProjectionService()
premier_league_service = PremierLeagueProjectionsService()
euro_comp_service = EuroCompProjectionService()
projection_all_teams_service = ProjectionAllTeams()
fetch_all_data_service = FetchAllDataService()

_projection_running = False

CALLBACK_URL = os.getenv("STATZ_CALLBACK_URL", "")
CALLBACK_SECRET = os.getenv("STATZ_CALLBACK_SECRET", "")


def _report_status(competition_id: str, status: str, started_at: str, finished_at: str = None, exit_code: int = None, stdout: str = None, stderr: str = None):
    """Report projection run status back to the Statz admin dashboard."""
    if not CALLBACK_URL or not CALLBACK_SECRET:
        logger.warning("Status callback not configured (STATZ_CALLBACK_URL / STATZ_CALLBACK_SECRET missing)")
        return

    try:
        resp = http_requests.post(
            CALLBACK_URL,
            json={
                "competition_id": competition_id,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
            headers={"X-Projections-Secret": CALLBACK_SECRET},
            timeout=10,
        )
        logger.info(f"Status callback for {competition_id}: {status} -> HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"Status callback failed for {competition_id}: {e}")


def _league_to_competition_id(league: str) -> str:
    """Convert league name to competition ID (slug format)."""
    return league.lower().replace(' ', '-').replace('.', '')


async def _run_all_leagues(leagues=None):
    global _projection_running
    try:
        await projection_all_teams_service.projectionAllTeams(leagues=leagues)
    except Exception as e:
        logger.error(f"All-leagues projection FAILED: {e}", exc_info=True)
    finally:
        _projection_running = False
        logger.info("Projection lock released.")


async def _run_single_league(request):
    global _projection_running
    competition_id = _league_to_competition_id(request.league)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        if EuroCompProjectionService.is_euro_comp(request.league):
            await euro_comp_service.projections(request)
        else:
            await projection_service.projections(request)
        finished_at = datetime.now(timezone.utc).isoformat()
        _report_status(competition_id, "success", started_at, finished_at, exit_code=0)
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"[{request.league}] projection FAILED: {e}", exc_info=True)
        _report_status(competition_id, "failed", started_at, finished_at, exit_code=1, stderr=str(e)[:500])
    finally:
        _projection_running = False
        logger.info("Projection lock released.")


async def _run_fetch_data():
    try:
        await fetch_all_data_service.import_all_tables()
    except Exception as e:
        logger.error(f"fetch-data FAILED: {e}", exc_info=True)


@router.post("")
async def projections(request: LeagueRequest, background_tasks: BackgroundTasks):
    """Start league projection in background - returns immediately, no timeout."""
    global _projection_running
    if _projection_running:
        return {"status": "busy", "message": "A projection is already running. Wait for it to finish."}
    _projection_running = True
    background_tasks.add_task(_run_single_league, request)
    return {"status": "started", "league": request.league}


@router.post("/fixtures")
async def fixtures(request: LeagueRequest):
    return await projection_service.fixtures(request)


@router.post("/predicted-tables")
async def predicted_tables(request: LeagueRequest):
    return await projection_service.predicted_table(request)


@router.post("/teams")
async def teams(request: LeagueRequest):
    return await projection_service.teams(request)


@router.post("/players")
async def players(request: LeagueRequest):
    return await projection_service.players(request)


@router.post("/player-props")
async def player_props(request: LeagueRequest):
    return await projection_service.player_props(request.league)


@router.post("/premier-projections")
async def premier_projections():
    return await premier_league_service.projections()


@router.post("/all-leagues")
async def all_leagues(background_tasks: BackgroundTasks, request: AllLeaguesRequest = None):
    """Start all-leagues projection in background. Optionally pass 'leagues' list to run only specific leagues."""
    global _projection_running
    if _projection_running:
        return {"status": "busy", "message": "A projection is already running. Wait for it to finish."}
    _projection_running = True
    leagues = request.leagues if request else None
    background_tasks.add_task(_run_all_leagues, leagues)
    msg = f"leagues: {leagues}" if leagues else "all leagues"
    return {"status": "started", "message": f"Projection started ({msg})"}


@router.post("/fetch-data")
async def fetch_data():
    """Fetch latest data from source DB and invalidate cache (synchronous)."""
    ProjectionService._cache.invalidate()
    t0 = time.time()
    logger.info("fetch-data: starting import_all_tables...")
    await fetch_all_data_service.import_all_tables()
    elapsed = round(time.time() - t0, 1)
    logger.info(f"fetch-data: done in {elapsed}s")
    return {"status": "done", "message": f"Data fetch complete in {elapsed}s"}
