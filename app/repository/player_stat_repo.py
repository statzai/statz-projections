import logging
import math
from datetime import datetime
from app.repository.db_utils import execute_chunked, resolve_team_id

logger = logging.getLogger("player_stat_repo")

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
    "Fouls Committed": 56,
}


async def insert_players_stats_async(data_list, teams=None, competition_id=None, comp_teams=None):
    if len(data_list) == 0:
        return

    api_pl_projections = data_list.copy()
    api_pl_projections = api_pl_projections.rename(columns={
        "Player": "player_name",
        "Position": "position",
        "Team": "team",
        "Opponent": "opponent",
        "Venue": "venue",
        "Market": "market_name",
        "Prop": "prop",
        "Projection %": "projection_percent"
    })

    api_pl_projections['kickoff_datetime'] = api_pl_projections['kickoff_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    def _parse_pct(x):
        # Strings like "23.45%" → 23.45; NaN/None/empty → None; numeric pass-through.
        # Previous version didn't reject NaN floats: str(nan) == 'nan',
        # float('nan') is still NaN, and the DB column is NOT NULL → insert
        # fail. Seen on players with no history whose projections became NaN
        # after the 2026-04-24 CSV restore surfaced all the downstream code
        # paths hidden behind earlier parquet crashes.
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        s = str(x).replace('%', '').strip()
        if s == '' or s.lower() == 'nan':
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return None if math.isnan(v) else v

    api_pl_projections['projection_percent'] = api_pl_projections['projection_percent'].apply(_parse_pct)

    # Rows with NULL projection_percent can't land (DB NOT NULL) and aren't
    # useful anyway — drop them with a count log so we can watch for upstream
    # regressions that produce widespread NaNs.
    _before = len(api_pl_projections)
    api_pl_projections = api_pl_projections[api_pl_projections['projection_percent'].notna()]
    _dropped = _before - len(api_pl_projections)
    if _dropped > 0:
        logger.warning(f"[player_prop_projections] dropped {_dropped}/{_before} rows with NULL projection_percent")

    # Note: player_name / team / opponent strings are no longer written to
    # the DB — team_id / opponent_id / player_id FKs replace them. See
    # nullable migration 2026_04_17_120000.
    # Build the insert tuples WITHOUT iterrows (Series-per-row is the slowest
    # pandas iteration) and WITHOUT re-resolving the same ~40 team names ~49k×.
    # Column-array iteration + a memoised resolver — byte-identical output:
    # same per-row logic, same STATUS_TYPES lookup, same resolve_team_id result
    # per name (a deterministic lookup), same None/NaN handling (execute_chunked
    # still cleans NaN→None downstream). row.get(col) → None for an absent
    # column is preserved by pre-filling missing columns with None.
    _cols = ['fixture_id', 'player_id', 'position', 'team', 'opponent',
             'venue', 'market_name', 'prop', 'projection_percent', 'kickoff_datetime']
    for _c in _cols:
        if _c not in api_pl_projections.columns:
            api_pl_projections[_c] = None

    _tid_cache = {}
    def _resolve(name):
        if teams is None:
            return None
        # Key NaN separately so a genuine NaN name is still resolved once,
        # identically to the original per-row resolve_team_id(nan, ...).
        key = '\x00NAN' if isinstance(name, float) and math.isnan(name) else name
        if key not in _tid_cache:
            _tid_cache[key] = resolve_team_id(name, teams, competition_id, comp_teams)
        return _tid_cache[key]

    values = []
    for (fid, pid, pos, team, opp, ven, mkt, prop, pct, ko) in zip(
        api_pl_projections['fixture_id'], api_pl_projections['player_id'],
        api_pl_projections['position'], api_pl_projections['team'],
        api_pl_projections['opponent'], api_pl_projections['venue'],
        api_pl_projections['market_name'], api_pl_projections['prop'],
        api_pl_projections['projection_percent'], api_pl_projections['kickoff_datetime'],
    ):
        values.append((
            fid, pid, pos,
            _resolve(team), _resolve(opp),
            ven, mkt, STATUS_TYPES.get(mkt, 0), prop, pct, ko,
        ))

    sql = """
    INSERT INTO player_prop_projections (
        fixture_id, player_id, position,
        team_id, opponent_id,
        venue, market_name, stats_type_id, prop, projection_percent,
        kickoff_datetime, created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON DUPLICATE KEY UPDATE
        position = VALUES(position),
        team_id = VALUES(team_id),
        opponent_id = VALUES(opponent_id),
        venue = VALUES(venue),
        stats_type_id = VALUES(stats_type_id),
        projection_percent = VALUES(projection_percent),
        kickoff_datetime = VALUES(kickoff_datetime),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[player_prop_projections]")
