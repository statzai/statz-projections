import logging
import math
from datetime import datetime
import pandas as pd
from app.source_database import get_source_connection, release_source_connection
from app.repository.db_utils import execute_chunked, resolve_team_id

logger = logging.getLogger("player_repo")

def convert_start(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["yes", "y", "1", "true"]:
            return 1
        if v in ["no", "n", "0", "false"]:
            return 0
    return 0


# Projected per-fixture player stats persisted to `player_projections`.
# Key = the column name on the projection DataFrame; value = stats_type id.
# A stat is only stored when its column is present on the row, so leagues
# that don't compute a given column (e.g. the PL-only defensive stats
# below) simply skip it — no breakage.
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
    "Big Chances Created": 580,
    # PL-only defensive projections — computed for the team-down CBIT
    # calc (see projection_service.py) but previously discarded. Stored
    # so the FPL planner can show projected defensive output; combined
    # with Tackles (78) these give a full projected defensive
    # contribution (DEF: Tackles+CBI, MID/FWD: Tackles+CBI+Recoveries).
    "Ball Recovery": 27271,
    "Clearances Blocks Interceptions (FPL)": 999002,
}


async def insert_player_async(data_list, teams=None, competition_id=None, comp_teams=None):
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

    # Note: row.get("team") / row.get("opponent") / row.get("player_name")
    # remain as input (resolve_team_id needs the names), but the string
    # columns are no longer written to the DB. See nullable migration
    # 2026_04_17_120000.
    # Build WITHOUT iterrows (Series-per-row is the slowest pandas iteration)
    # and WITHOUT re-resolving the same ~40 team names once per player.
    # Column-array iteration + memoised resolver — byte-identical output.
    # The `value is None` skip (NOT a NaN skip) is preserved exactly: these
    # frames always have string columns (position/venue/team…), so an iterrows
    # row-Series was already object-dtype — None stayed None, NaN stayed NaN —
    # which is exactly what iterating each column directly gives. NaN values
    # still pass through as float(nan) and are cleaned to NULL by
    # execute_chunked, same as before. `stat_name in row` (index membership)
    # → `stat_name in columns`, checked once.
    _n = len(api_pl_projections)

    def _colvals(name):
        return api_pl_projections[name].tolist() if name in api_pl_projections.columns else [None] * _n

    _fid = _colvals('fixture_id')
    _pid = _colvals('player_id')
    _pos = _colvals('position')
    _team = _colvals('team')
    _opp = _colvals('opponent')
    _ven = _colvals('venue')
    _start = _colvals('start')
    _ko = _colvals('kickoff_datetime')
    _stat_arrays = [
        (stat_id, api_pl_projections[stat_name].tolist())
        for stat_name, stat_id in STATUS_TYPES.items()
        if stat_name in api_pl_projections.columns
    ]

    _tid_cache = {}
    def _resolve(name):
        if teams is None:
            return None
        key = '\x00NAN' if isinstance(name, float) and math.isnan(name) else name
        if key not in _tid_cache:
            _tid_cache[key] = resolve_team_id(name, teams, competition_id, comp_teams)
        return _tid_cache[key]

    records = []
    for i in range(_n):
        row_team_id = _resolve(_team[i])
        row_opponent_id = _resolve(_opp[i])
        start_v = convert_start(_start[i])
        fid_i, pid_i, pos_i, ven_i, ko_i = _fid[i], _pid[i], _pos[i], _ven[i], _ko[i]
        for stat_id, vals in _stat_arrays:
            value = vals[i]
            if value is None:
                continue
            records.append((
                fid_i, pid_i, stat_id, pos_i,
                row_team_id, row_opponent_id, ven_i,
                start_v, float(value), ko_i, now, now,
            ))

    sql = """
    INSERT INTO player_projections (
        fixture_id, player_id, stats_type_id, position,
        team_id, opponent_id,
        venue, start, stats_value, kickoff_datetime, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        position = VALUES(position),
        team_id = VALUES(team_id),
        opponent_id = VALUES(opponent_id),
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
