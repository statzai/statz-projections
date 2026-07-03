"""Writer for wc_fantasy_projections — the per-(fixture, player) WC Fantasy
points table that the planner reads.

Single-row schema (one row per fixture × player), idempotent upsert via the
(fixture_id, player_id) unique key. Mirrors fpl_repo.insert_fpl_projections_async.
"""
import logging

from app.repository.db_utils import execute_chunked

logger = logging.getLogger("wc_fantasy_repo")


async def insert_wc_fantasy_projections_async(rows):
    """Bulk upsert rows into wc_fantasy_projections.

    Each row is a tuple in the column order of the INSERT statement below.
    Pass an empty list and we no-op.
    """
    if not rows:
        return 0

    sql = """
    INSERT INTO wc_fantasy_projections (
        fixture_id, player_id, kickoff_datetime, venue,
        fantasy_points, wc_round_id, team_id, opponent_id, position,
        xg, xa, saves, tackles, sot, big_chances_created, cs_pct, xgc,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        kickoff_datetime = new.kickoff_datetime,
        venue            = new.venue,
        fantasy_points   = new.fantasy_points,
        wc_round_id      = new.wc_round_id,
        team_id          = new.team_id,
        opponent_id      = new.opponent_id,
        position         = new.position,
        xg                  = new.xg,
        xa                  = new.xa,
        saves               = new.saves,
        tackles             = new.tackles,
        sot                 = new.sot,
        big_chances_created = new.big_chances_created,
        cs_pct              = new.cs_pct,
        xgc                 = new.xgc,
        updated_at       = NOW()
    """
    return await execute_chunked(sql, rows, label="[wc_fantasy_projections]")
