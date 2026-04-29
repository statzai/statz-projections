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
        "Value": "value",
        "Gameweek": "gameweek_id",
    })

    if hasattr(df['kickoff_datetime'].iloc[0], 'strftime'):
        df['kickoff_datetime'] = df['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Optional non-key columns — tolerate missing (NaN/None) since we no
    # longer source price from Fanteam Data.csv.
    has_price = "price" in df.columns
    has_value = "value" in df.columns
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

    def _num_or_none(v):
        if v is None:
            return None
        try:
            if v != v:  # NaN
                return None
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            return None

    values = [
        (
            row.get("fixture_id"),
            row.get("player_id"),
            row.get("kickoff_datetime"),
            row.get("venue"),
            _num_or_none(row.get("price")) if has_price else None,
            row.get("fan_team_points"),
            _num_or_none(row.get("value")) if has_value else None,
            _int_or_none(row.get("gameweek_id")) if has_gw else None,
            _int_or_none(row.get("team_id")) if has_team else None,
            _int_or_none(row.get("opponent_id")) if has_opp else None,
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO fanteam_projections (
        fixture_id, player_id, kickoff_datetime, venue, price,
        fan_team_points, value, gameweek_id, team_id, opponent_id,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        fan_team_points = new.fan_team_points,
        price = new.price,
        value = new.value,
        gameweek_id = new.gameweek_id,
        team_id = new.team_id,
        opponent_id = new.opponent_id,
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[fanteam_projections]")
