import logging
from datetime import datetime
from app.repository.db_utils import execute_chunked, resolve_team_id

logger = logging.getLogger("team_repo")


async def insert_teams_async(data_list, teams=None, competition_id=None, comp_teams=None):
    api_team_projections_save = data_list.copy()
    api_team_projections_save = api_team_projections_save.rename(columns={
        "Team": "team",
        "Opponent": "opponent",
        "Venue": "venue",
        "Goals": "goals",
        "Shots Total": "shots_total",
        "Shots On Target": "shots_on_target",
        "Corners": "corners",
        "Fouls": "fouls",
        "Yellowcards": "yellowcards",
        "Tackles": "tackles",
        "Passes": "passes",
        "Successful Passes": "successful_passes",
        "Total Crosses": "total_crosses",
        "Interceptions": "interceptions",
        "Offsides": "offsides",
    })

    api_team_projections_save['kickoff_datetime'] = api_team_projections_save['kickoff_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Note: row['team'] / row['opponent'] remain as input to resolve_team_id()
    # but are no longer written to the DB — team_id / opponent_id replace
    # them. See nullable migration 2026_04_17_120000.
    values = [
        (
            row['fixture_id'],
            resolve_team_id(row['team'], teams, competition_id, comp_teams) if teams is not None else None,
            resolve_team_id(row['opponent'], teams, competition_id, comp_teams) if teams is not None else None,
            row['venue'],
            row['goals'],
            row['shots_total'],
            row['shots_on_target'],
            row['corners'],
            row['fouls'],
            row['yellowcards'],
            row['tackles'],
            row['passes'],
            row.get('successful_passes'),
            row['total_crosses'],
            row.get('interceptions'),
            row['offsides'],
            row['kickoff_datetime'],
        )
        for _, row in api_team_projections_save.iterrows()
    ]

    sql = """
    INSERT INTO team_projections (
        fixture_id, team_id, opponent_id,
        venue, goals,
        shots_total, shots_on_target, corners,
        fouls, yellowcards, tackles, passes, successful_passes,
        total_crosses, interceptions, offsides, kickoff_datetime,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON DUPLICATE KEY UPDATE
        team_id = VALUES(team_id),
        opponent_id = VALUES(opponent_id),
        venue = VALUES(venue),
        goals = VALUES(goals),
        shots_total = VALUES(shots_total),
        shots_on_target = VALUES(shots_on_target),
        corners = VALUES(corners),
        fouls = VALUES(fouls),
        yellowcards = VALUES(yellowcards),
        tackles = VALUES(tackles),
        passes = VALUES(passes),
        successful_passes = VALUES(successful_passes),
        total_crosses = VALUES(total_crosses),
        interceptions = VALUES(interceptions),
        offsides = VALUES(offsides),
        kickoff_datetime = VALUES(kickoff_datetime),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label="[team_projections]")
