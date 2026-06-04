"""Writer for fanteam_wc_projections — per-(fixture, player) FanTeam WC
fantasy point projections. Sibling of wc_fantasy_repo.py (FIFA scoring).

Idempotent upsert via the (fixture_id, player_id) unique key.
"""
import logging

from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fanteam_wc_repo")


async def insert_fanteam_wc_projections_async(rows):
    """Bulk upsert into fanteam_wc_projections. Pass empty list → no-op."""
    if not rows:
        return 0

    sql = """
    INSERT INTO fanteam_wc_projections (
        fixture_id, player_id, kickoff_datetime, venue,
        fan_team_points, wc_round_id, team_id, opponent_id, position,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        kickoff_datetime = new.kickoff_datetime,
        venue            = new.venue,
        fan_team_points  = new.fan_team_points,
        wc_round_id      = new.wc_round_id,
        team_id          = new.team_id,
        opponent_id      = new.opponent_id,
        position         = new.position,
        updated_at       = NOW()
    """
    return await execute_chunked(sql, rows, label="[fanteam_wc_projections]")
