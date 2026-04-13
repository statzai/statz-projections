import logging
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("fixtures_repo")


def clean_percentage(value):
    if isinstance(value, str) and '%' in value:
        return float(value.replace('%', '').strip())
    return float(value) if value is not None else None


async def insert_fixtures_async(data_list):
    if len(data_list) == 0:
        return

    api_score_preds = data_list.copy()
    api_score_preds = api_score_preds.rename(columns={
        "id": "fixture_id",
        "Home Team": "home_team_name",
        "Home Goals": "home_goals",
        "Away Goals": "away_goals",
        "Away Team": "away_team_name",
        "Home Win %": "home_win_percent",
        "Draw %": "draw_percent",
        "Away Win %": "away_win_percent",
        "Home Clean Sheet %": "home_clean_sheet_percent",
        "Away Clean Sheet %": "away_clean_sheet_percent",
        "Over 1.5 Goals %": "over_15_goals_percent",
        "Over 2.5 Goals %": "over_25_goals_percent",
        "Both Teams Score %": "both_teams_shore_percent",
    })

    for col in ['home_win_percent', 'draw_percent', 'away_win_percent',
                'home_clean_sheet_percent', 'away_clean_sheet_percent',
                'over_15_goals_percent', 'over_25_goals_percent', 'both_teams_shore_percent']:
        if col in api_score_preds.columns:
            api_score_preds[col] = api_score_preds[col].apply(clean_percentage)

    values = [
        (
            row['fixture_id'],
            row['home_team_name'],
            row['away_team_name'],
            row['home_goals'],
            row['away_goals'],
            row['home_win_percent'],
            row['away_win_percent'],
            row['draw_percent'],
            row['home_clean_sheet_percent'],
            row['away_clean_sheet_percent'],
            row['over_15_goals_percent'],
            row['over_25_goals_percent'],
            row['both_teams_shore_percent'],
            row['kickoff_datetime'],
        )
        for _, row in api_score_preds.iterrows()
    ]

    sql = """
    INSERT INTO fixture_projections (
        fixture_id, home_team_name, away_team_name, home_goals, away_goals,
        home_win_percent, away_win_percent, draw_percent,
        home_clean_sheet_percent, away_clean_sheet_percent,
        over_15_goals_percent, over_25_goals_percent, both_teams_shore_percent,
        kickoff_datetime,
        created_at, updated_at
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s,
        NOW(), NOW()
    )
    ON DUPLICATE KEY UPDATE
        home_team_name = VALUES(home_team_name),
        away_team_name = VALUES(away_team_name),
        home_goals = VALUES(home_goals),
        away_goals = VALUES(away_goals),
        home_win_percent = VALUES(home_win_percent),
        away_win_percent = VALUES(away_win_percent),
        draw_percent = VALUES(draw_percent),
        home_clean_sheet_percent = VALUES(home_clean_sheet_percent),
        away_clean_sheet_percent = VALUES(away_clean_sheet_percent),
        over_15_goals_percent = VALUES(over_15_goals_percent),
        over_25_goals_percent = VALUES(over_25_goals_percent),
        both_teams_shore_percent = VALUES(both_teams_shore_percent),
        kickoff_datetime = VALUES(kickoff_datetime),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[fixture_projections]")
