import logging
from app.repository.db_utils import execute_chunked
from app.repository.league_outcome_repo import (
    is_split_format_competition,
    resolve_current_season_id,
)

logger = logging.getLogger("predicted_table_repo")


async def insert_predicted_table_async(data_list, teams, comps, league):
    """Write the projected league table — position / points / goals / point
    range — to league_projections.

    The per-market probabilities are NOT written here: they live in
    league_projection_outcomes, one row per (team, market), written by
    league_outcome_repo. The legacy fixed `*_percent` columns were dropped
    2026-05-19 (rollout step 6) once the rule-driven outcomes had soaked at
    full cross-league parity.
    """
    if league == 'Brazil Serie A':
        competition_id = 648
    else:
        competition_id = comps.loc[comps['name'] == league, "id"]
        if not competition_id.empty:
            competition_id = int(competition_id.iloc[0])
        else:
            raise Exception(f"League {league} not found in comps")

    # Tie the projected table to the current season + skip split-format
    # competitions (Scottish Premiership, Austrian/Belgian/Danish/Greek
    # split rounds) — a single continuous table can't represent them.
    # Defensive: a failure resolving this must not break the projection,
    # so on error we proceed with the write (season_id NULL).
    season_id = None
    try:
        season_id = await resolve_current_season_id(competition_id)
        if season_id is not None and await is_split_format_competition(competition_id, season_id):
            logger.info(f"[league_projections:{league}] split-format competition — "
                        f"skipping predicted table write")
            return 0
    except Exception as e:
        logger.warning(f"[league_projections:{league}] season/split-format check failed "
                        f"(proceeding with write): {type(e).__name__}: {e}")

    df = data_list.copy()
    df['competition_id'] = competition_id
    df['season_id'] = season_id

    for idx, row in df.iterrows():
        team_name = row["Team"]
        team_id = teams.loc[teams["name"] == team_name, "id"]
        if not team_id.empty:
            df.at[idx, "team_id"] = int(team_id.iloc[0])
        else:
            df.at[idx, "team_id"] = None

    missing = df[df["team_id"].isnull()]
    if not missing.empty:
        logger.error("Missing team IDs for: %s", missing["Team"].tolist())
        raise Exception("Some teams from data_list do not exist in teams list")

    df = df.rename(columns={
        "Position": "position",
        "Points": "points",
        "Goals For": "goals_for",
        "Goals Against": "goals_against",
        "Goal Difference": "goal_difference",
        "Max Points": "max_points",
        "Min Points": "min_points",
    })

    values = [
        (
            row['position'],
            row['team_id'],
            row['points'],
            row['goals_for'],
            row['goals_against'],
            row['goal_difference'],
            row['max_points'],
            row['min_points'],
            row['competition_id'],
            row['season_id'],
        )
        for _, row in df.iterrows()
    ]

    sql = """
    INSERT INTO league_projections (
        position, team_id, points, goals_for, goals_against, goal_difference,
        max_points, min_points, competition_id, season_id,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON DUPLICATE KEY UPDATE
        position = VALUES(position),
        points = VALUES(points),
        goals_for = VALUES(goals_for),
        goals_against = VALUES(goals_against),
        goal_difference = VALUES(goal_difference),
        max_points = VALUES(max_points),
        min_points = VALUES(min_points),
        season_id = VALUES(season_id),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label=f"[league_projections:{league}]")
