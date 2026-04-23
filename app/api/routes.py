import logging
import time
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from app.repository.projection_run_repo import upsert_run_complete

class AllLeaguesRequest(BaseModel):
    leagues: Optional[List[str]] = None
    fetch_first: Optional[bool] = False


class PromoteModelRequest(BaseModel):
    reason: Optional[str] = "manual promotion via admin panel"

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


async def _report_status(competition_id: str, status: str, started_at: str, finished_at: str = None, exit_code: int = None, stdout: str = None, stderr: str = None):
    """
    Report projection run status back to the Statz admin dashboard.

    Writes directly to the projections_runs table via the DB pool instead
    of POSTing over HTTP. Fixes the deploy-window callback-loss issue
    where Laravel restarts during a deploy would drop incoming HTTP
    requests and leave runs marked 'running' until mark-stuck timed them
    out 30 min later.
    """
    await upsert_run_complete(
        competition_id=competition_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def _league_to_competition_id(league: str) -> str:
    """Convert league name to competition ID (slug format)."""
    return league.lower().replace(' ', '-').replace('.', '')


async def _run_fetch_if_needed(fetch_first: bool):
    """Run data fetch before projections when triggered from admin panel."""
    if not fetch_first:
        return
    logger.info("fetch_first=True — fetching fresh data before projecting...")
    try:
        await fetch_all_data_service.import_all_tables()
        # Invalidate cache so the next projection loads fresh CSVs
        ProjectionService._cache.invalidate()
        logger.info("fetch_first: data fetch complete, cache invalidated")
    except Exception as e:
        logger.error(f"fetch_first: data fetch FAILED: {e}", exc_info=True)
        # Continue with projection anyway — better stale data than no projection


async def _run_all_leagues(leagues=None, fetch_first=False):
    global _projection_running
    try:
        await _run_fetch_if_needed(fetch_first)
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
        await _run_fetch_if_needed(getattr(request, 'fetch_first', False))
        if EuroCompProjectionService.is_euro_comp(request.league):
            await euro_comp_service.projections(request)
        else:
            await projection_service.projections(request)
        finished_at = datetime.now(timezone.utc).isoformat()
        await _report_status(competition_id, "success", started_at, finished_at, exit_code=0)
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"[{request.league}] projection FAILED: {e}", exc_info=True)
        await _report_status(competition_id, "failed", started_at, finished_at, exit_code=1, stderr=str(e)[:500])
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
    fetch_first = request.fetch_first if request else False
    background_tasks.add_task(_run_all_leagues, leagues, fetch_first)
    msg = f"leagues: {leagues}" if leagues else "all leagues"
    fetch_msg = " (fetching data first)" if fetch_first else ""
    return {"status": "started", "message": f"Projection started ({msg}){fetch_msg}"}


@router.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    """Kick off model retraining in the background. Phase 4 = dry-run only
    (logs would-be promotions, doesn't flip is_active). Returns immediately;
    the full retrain takes ~5-15 min depending on data volume.

    Shares the _projection_running lock with the projection endpoints —
    training 132+ PoissonRegressor models + grid searches is memory-heavy
    and can OOM if a projection is running concurrently. Returns 'busy'
    if a projection is already in progress; caller should retry later.
    """
    from app.services.retrain_service import retrain_all_models

    global _projection_running
    if _projection_running:
        return {"status": "busy", "message": "A projection or retrain is already running. Wait for it to finish."}
    _projection_running = True

    async def _run():
        global _projection_running
        try:
            await retrain_all_models(dry_run=True)
        except Exception as e:
            logger.error(f"retrain FAILED: {e}", exc_info=True)
        finally:
            _projection_running = False
            logger.info("Retrain lock released.")

    background_tasks.add_task(_run)
    return {"status": "started", "mode": "dry-run", "message": "Retraining started in background; check projection.log for per-(league,stat) output"}


@router.post("/models/{model_id}/promote")
async def promote_model_endpoint(model_id: int, request: PromoteModelRequest = None):
    """Manually promote a specific projection_models row to is_active=1.
    The per-(competition_id, stat_name) invariant is maintained atomically
    by promote_model() — demotes the incumbent in the same transaction.

    Phase 4 is dry-run (auto-retrain doesn't flip is_active). Manual
    promotion via this endpoint is the Phase 6 admin override — user
    explicitly chose to replace the incumbent, so no guardrail check.
    Also covers rollback: promoting an older version restores it as
    active, demoting the current one.
    """
    from app.repository.projection_model_repo import promote_model

    reason = (request.reason if request else None) or "manual promotion via admin panel"
    try:
        await promote_model(model_id, reason)
        return {"status": "ok", "model_id": model_id, "reason": reason}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"promote_model({model_id}) failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Internal error: {type(e).__name__}"}


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
