"""Writer for wc_tournament_player_projections — per-(competition, player)
WC tournament-total goals + assists.

Single row per player per WC; idempotent upsert via the
(competition_id, player_id) unique key. Mirrors the bulk-insert pattern
used by other WC repos.
"""
import logging

from app.repository.db_utils import execute_chunked

logger = logging.getLogger("wc_tournament_player_repo")


async def insert_wc_tournament_player_projections_async(rows):
    """Bulk upsert rows into wc_tournament_player_projections.

    Each row is a tuple matching the INSERT column order:
      (competition_id, player_id, team_id, position,
       tournament_goals, tournament_assists, num_sims)

    Pass an empty list and we no-op.
    """
    if not rows:
        return 0

    sql = """
    INSERT INTO wc_tournament_player_projections (
        competition_id, player_id, team_id, position,
        tournament_goals, tournament_assists, num_sims,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        team_id            = new.team_id,
        position           = new.position,
        tournament_goals   = new.tournament_goals,
        tournament_assists = new.tournament_assists,
        num_sims           = new.num_sims,
        updated_at         = NOW()
    """
    return await execute_chunked(sql, rows, label="[wc_tournament_player_projections]")
