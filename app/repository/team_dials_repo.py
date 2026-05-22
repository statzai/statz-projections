"""
Repo for the projections_team_dials table — per-team Attack / Defence
overrides set by an admin operator in the Projections Admin Console.

Each row is a (competition_id, team_id) override. Values are signed
percentage adjustments (-50..+50) applied to the team's Attack and/or
Defence rating during projection. A row with both adjustments at 0 is
deleted by the Laravel controller, so any row returned here is by
definition "active".

Applied in projection_service.py after the market-value adjustment but
before the rescale-to-mean=100 step, so dialled teams shift the league
mean and other teams' indexed values drift naturally.
"""
import logging

import app.database as _db
from app.database import get_connection

logger = logging.getLogger("team_dials_repo")


async def load_team_dials(competition_id: int) -> dict:
    """Return {team_id: (attack_pct, defense_pct)} for a competition.

    Empty dict if no dials are set. Both values are ints in -50..+50;
    zero is filtered out at write time so any value here is a real
    override.
    """
    if not competition_id:
        return {}

    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT team_id, attack_adjustment, defense_adjustment
                FROM projections_team_dials
                WHERE competition_id = %s
                  AND (attack_adjustment != 0 OR defense_adjustment != 0)
                """,
                (int(competition_id),),
            )
            rows = await cur.fetchall()
        return {int(tid): (int(atk), int(dfn)) for tid, atk, dfn in rows}
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
