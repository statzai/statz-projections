import logging
import app.database as _db
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fpl_repo")


async def insert_fpl_projections_async(data_list):
    if len(data_list) == 0:
        return

    df = data_list.copy()
    df = df.rename(columns={
        "FPL Points": "fpl_points",
        "Venue": "venue"
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            row.get("fpl_points"),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO fpl_projections (
        fixture_id, player_id, kickoff_datetime, venue, fpl_points,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        fpl_points = new.fpl_points,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[fpl_projections]")
