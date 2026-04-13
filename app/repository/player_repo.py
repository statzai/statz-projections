import logging
from datetime import datetime
import pandas as pd
from app.source_database import get_source_connection, release_source_connection
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("player_repo")

def convert_start(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["yes", "y", "1", "true"]:
            return 1
        if v in ["no", "n", "0", "false"]:
            return 0
    return 0


STATUS_TYPES = {
    "Shots Total": 42,
    "Offsides": 51,
    "Goals": 52,
    "Fouls": 56,
    "Saves": 57,
    "Tackles": 78,
    "Assists": 79,
    "Passes": 80,
    "Yellow Cards": 84,
    "Shots On Target": 86,
    "Fouls Drawn": 96,
    "Total Crosses": 98,
    "Interceptions": 100,
    "Accurate Passes": 116,
    "Key Passes": 117,
}


async def insert_player_async(data_list):
    if len(data_list) == 0:
        return

    api_pl_projections = data_list.copy()
    api_pl_projections = api_pl_projections.rename(columns={
        "Player": "player_name",
        "Position": "position",
        "Team": "team",
        "Opponent": "opponent",
        "Venue": "venue",
        "Start?": "start",
        "Kickoff": "kickoff_datetime"
    })

    api_pl_projections['kickoff_datetime'] = api_pl_projections['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    records = []
    for _, row in api_pl_projections.iterrows():
        for stat_name, stat_id in STATUS_TYPES.items():
            if stat_name in row:
                value = row[stat_name]
                if value is None:
                    continue
                records.append((
                    row.get("fixture_id"),
                    row.get("player_id"),
                    stat_id,
                    row.get("player_name"),
                    row.get("position"),
                    row.get("team"),
                    row.get("opponent"),
                    row.get("venue"),
                    convert_start(row.get("start")),
                    float(value),
                    row.get("kickoff_datetime"),
                    now,
                    now,
                ))

    sql = """
    INSERT INTO player_projections (
        fixture_id, player_id, stats_type_id, player_name, position, team, opponent,
        venue, start, stats_value, kickoff_datetime, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        player_name = VALUES(player_name),
        position = VALUES(position),
        team = VALUES(team),
        opponent = VALUES(opponent),
        venue = VALUES(venue),
        start = VALUES(start),
        stats_value = VALUES(stats_value),
        kickoff_datetime = VALUES(kickoff_datetime),
        updated_at = VALUES(updated_at)
    """
    return await execute_chunked(sql, records, label="[player_projections]")


async def get_players_from_league(league):
    conn = None
    try:
        conn = await get_source_connection()
        async with conn.cursor() as cursor:
            sql = """
            SELECT p.*
            FROM players p
            WHERE p.current_team_id IN (
                SELECT t.id
                FROM competition_season_teams cst
                LEFT OUTER JOIN competitions c ON c.id = cst.competition_id
                LEFT OUTER JOIN seasons s ON s.id = cst.season_id
                LEFT OUTER JOIN teams t ON t.id = cst.team_id
                WHERE c.name = %s AND s.is_current = true
            )
            """
            await cursor.execute(sql, (league,))
            rows = await cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        logger.error(f"Error fetching players from league {league}: {e}")
        raise
    finally:
        release_source_connection(conn)
