import asyncio
import logging
from datetime import datetime, timezone

import app.database as _db
from app.database import get_connection

logger = logging.getLogger("projection_run_repo")


def _to_mysql_datetime(value):
    """
    Coerce ISO-format strings (what routes._run_single_league passes) to
    MySQL-friendly datetime objects. Returns None for falsy input.
    aiomysql binds datetime.datetime as DATETIME natively.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def touch_run_start(competition_id: str):
    """Update the latest 'running' row's started_at to NOW.

    The Laravel-side pre-create in triggerRunAll stamps all 24 running
    rows with the click-time. Sequential per-league processing in
    projection_all_teams_service then takes ~5 min × N leagues, so rows
    for leagues later in the queue cross the mark-stuck 30-min
    threshold while still legitimately queued — causing false
    "stuck" flips. Calling this at the top of each league's iteration
    resets its row's started_at to the actual processing-start time,
    so mark-stuck only catches genuinely-wedged runs.

    Silent on miss (no running row found) — scheduled / Run Now flows
    use the single-league path where the row IS pre-created, but
    direct curl to /all-leagues may skip pre-create entirely.
    """
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(
                "UPDATE projections_runs SET started_at = NOW() "
                "WHERE id = ("
                "  SELECT id FROM ("
                "    SELECT id FROM projections_runs "
                "    WHERE competition_id = %s AND status = 'running' "
                "    ORDER BY started_at DESC LIMIT 1"
                "  ) AS sub"
                ")",
                (competition_id,),
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"[projections_runs] {competition_id}: touch_run_start failed: {e}")
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def upsert_run_complete(
    competition_id: str,
    status: str,
    started_at: str,
    finished_at: str,
    exit_code: int = None,
    stdout: str = None,
    stderr: str = None,
):
    """
    Replacement for the HTTP status callback. Writes projection run
    completion state directly to the projections_runs table instead of
    POSTing to Laravel's /api/internal/projections/status endpoint.

    Mirrors the logic in ProjectionsAdminController::reportStatus — find
    the latest 'running' row for this competition_id and update it; if
    none exists (e.g. the run was never pre-registered), insert a complete
    row.

    Does NOT raise on DB errors — mark-stuck (runs every 5 min on the
    Laravel side) is the safety net. Better to log + move on than block
    the projection lock release.
    """
    stdout_snippet = (stdout or '')[:500]
    stderr_snippet = (stderr or '')[:500]
    started_at_dt = _to_mysql_datetime(started_at)
    finished_at_dt = _to_mysql_datetime(finished_at)

    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT id FROM projections_runs "
                "WHERE competition_id = %s AND status = 'running' "
                "ORDER BY started_at DESC LIMIT 1",
                (competition_id,),
            )
            row = await cursor.fetchone()
            if row:
                run_id = row[0]
                await cursor.execute(
                    "UPDATE projections_runs SET "
                    "status = %s, finished_at = %s, exit_code = %s, "
                    "stdout_snippet = %s, stderr_snippet = %s "
                    "WHERE id = %s",
                    (status, finished_at_dt, exit_code,
                     stdout_snippet, stderr_snippet, run_id),
                )
                await conn.commit()
                logger.info(
                    f"[projections_runs] {competition_id}: updated run {run_id} -> {status}"
                )
            else:
                await cursor.execute(
                    "INSERT INTO projections_runs "
                    "(competition_id, started_at, finished_at, status, "
                    "exit_code, stdout_snippet, stderr_snippet, "
                    "triggered_by, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'schedule', NOW())",
                    (competition_id, started_at_dt, finished_at_dt, status,
                     exit_code, stdout_snippet, stderr_snippet),
                )
                await conn.commit()
                logger.info(
                    f"[projections_runs] {competition_id}: no running row — "
                    f"inserted complete row as {status}"
                )
    except Exception as e:
        logger.error(
            f"[projections_runs] {competition_id}: DB write failed: {e}",
            exc_info=True,
        )
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
