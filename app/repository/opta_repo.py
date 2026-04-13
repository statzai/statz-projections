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
        "Floor PTS": "floor_pts"
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            row.get("pts"),
            row.get("floor_pts"),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO opta_projections (
        fixture_id, player_id, kickoff_datetime, venue, pts,
        floor_pts, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        pts = new.pts,
        floor_pts = new.floor_pts,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[opta_projections]")
