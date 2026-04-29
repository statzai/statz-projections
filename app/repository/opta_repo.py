import logging
import app.database as _db
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("opta_repo")


async def insert_opta_projections_async(data_list):
    if len(data_list) == 0:
        return

    df = data_list.copy()
    df = df.rename(columns={
        "Venue": "venue",
        "PTS": "pts",
        "Floor PTS": "floor_pts",
        "Gameweek": "gameweek_id",
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    has_gw = "gameweek_id" in df.columns
    has_team = "team_id" in df.columns
    has_opp = "opponent_id" in df.columns

    def _int_or_none(v):
        if v is None:
            return None
        try:
            if v != v:  # NaN
                return None
        except Exception:
            pass
        try:
            return int(v)
        except Exception:
            return None

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            row.get("pts"),
            row.get("floor_pts"),
            _int_or_none(row.get("gameweek_id")) if has_gw else None,
            _int_or_none(row.get("team_id")) if has_team else None,
            _int_or_none(row.get("opponent_id")) if has_opp else None,
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO opta_projections (
        fixture_id, player_id, kickoff_datetime, venue, pts,
        floor_pts, gameweek_id, team_id, opponent_id,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        pts = new.pts,
        floor_pts = new.floor_pts,
        gameweek_id = new.gameweek_id,
        team_id = new.team_id,
        opponent_id = new.opponent_id,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[opta_projections]")
