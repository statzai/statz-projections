import logging
import app.database as _db
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fanteam_repo")


async def insert_fanteam_projections_async(data_list):
    if len(data_list) == 0:
        return

    df = data_list.copy()
    df = df.rename(columns={
        "Venue": "venue",
        "Price": "price",
        "FanTeam Points": "fan_team_points",
        "Value": "value"
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            row.get("price"),
            row.get("fan_team_points"),
            row.get("value"),
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO fanteam_projections (
        fixture_id, player_id, kickoff_datetime, venue, price,
        fan_team_points, value, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        fan_team_points = new.fan_team_points,
        price = new.price,
        value = new.value,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[fanteam_projections]")
