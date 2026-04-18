import logging
import pandas as pd
from app.repository.db_utils import execute_chunked, resolve_team_id

logger = logging.getLogger("team_ratings_repo")


async def insert_team_ratings_async(ratings_df, league_name, competition_id, teams):
    """
    Insert/upsert team ratings for one league into the team_ratings DB table.

    Replaces the legacy parquet/xlsx write paths across all 4 projection
    services. Ratings are resolved from (League name, Team name) to
    (competition_id, team_id) at write time so the DB layer is FK-friendly.

    Args:
        ratings_df: DataFrame with columns Team, Attack, Defense, Overall,
            Movement. Date and League are attached per-row below.
        league_name: League name (e.g. "Premier League"), stored on the
            League column for observability and as a fallback if FK
            resolution fails.
        competition_id: Resolved Laravel competition_id for this league.
        teams: DataFrame of teams (id, name, ...) from DataCache, used to
            resolve Team name → team_id.

    Rows with unresolved team_id are dropped with a warning — ratings for
    a team we can't FK-resolve are useless (the UI layer joins via the FK).
    """
    if ratings_df is None or len(ratings_df) == 0:
        logger.info(f"[team_ratings] {league_name}: nothing to insert")
        return

    df = ratings_df.copy()

    # Ensure the columns we need exist — callers may pass a reduced frame.
    for col in ('Attack', 'Defense', 'Overall', 'Attack_xG', 'Defense_xG', 'Overall_xG', 'Movement'):
        if col not in df.columns:
            df[col] = None

    today = pd.Timestamp('today').date()

    values = []
    unresolved = 0
    for _, row in df.iterrows():
        team_name = row.get('Team')
        if team_name is None or (isinstance(team_name, float) and pd.isna(team_name)):
            unresolved += 1
            continue
        team_id = resolve_team_id(team_name, teams) if teams is not None else None
        if team_id is None:
            unresolved += 1
            continue

        def _val(v):
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            return v

        values.append((
            competition_id,
            team_id,
            today,
            _val(row.get('Attack')),
            _val(row.get('Defense')),
            _val(row.get('Overall')),
            _val(row.get('Attack_xG')),
            _val(row.get('Defense_xG')),
            _val(row.get('Overall_xG')),
            _val(row.get('Movement')),
            _val(row.get('Inverse')),
        ))

    if unresolved:
        logger.warning(f"[team_ratings] {league_name}: skipped {unresolved} rows with unresolved team_id")

    if not values:
        logger.info(f"[team_ratings] {league_name}: nothing to insert after resolution")
        return

    sql = """
    INSERT INTO team_ratings (
        competition_id, team_id, date,
        attack, defense, overall,
        attack_xg, defense_xg, overall_xg,
        movement, inverse,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON DUPLICATE KEY UPDATE
        attack = VALUES(attack),
        defense = VALUES(defense),
        overall = VALUES(overall),
        attack_xg = VALUES(attack_xg),
        defense_xg = VALUES(defense_xg),
        overall_xg = VALUES(overall_xg),
        movement = VALUES(movement),
        inverse = VALUES(inverse),
        updated_at = NOW()
    """
    await execute_chunked(sql, values, label=f"[team_ratings {league_name}]")
