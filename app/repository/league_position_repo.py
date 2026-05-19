"""
League position probabilities — the full per-position finishing distribution.

Phase 2 of the league-projections redesign (docs/league-projections-redesign.md).
`sim_multiple_seasons()` produces a finishing-position histogram per team;
`league_outcome_repo` collapses it into rule sections and discards the rest —
this persists it RAW: one row per (competition, season, team, position) with
P(finish exactly there).

The raw distribution is the sufficient statistic — every positional betting
market (Win, Top 4/5/6, Not Top 4, Top/Bottom Half, Finish Bottom, relegation)
is a read-time range-sum over it, so the generator never changes again when a
bookmaker adds a market or a qualification allocation shifts.

Dual-writes alongside `league_outcome_repo` for one cycle so the read side can
parity-check the derived win/relegation numbers; `league_projection_outcomes`
retires once parity holds.
"""
import logging

from app.repository.db_utils import execute, execute_chunked, fetch_all
# Shared resolvers — split-format / season / competition-id detection. These
# move out of league_outcome_repo when league_projection_outcomes retires;
# for now they are imported, mirroring predicted_table_repo.
from app.repository.league_outcome_repo import (
    is_split_format_competition,
    resolve_competition_id,
    resolve_current_season_id,
)

logger = logging.getLogger("league_position_repo")

_INSERT_SQL = """
INSERT INTO league_position_probabilities
    (competition_id, season_id, team_id, position, probability,
     created_at, updated_at)
VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
ON DUPLICATE KEY UPDATE
    probability = VALUES(probability),
    updated_at  = NOW()
"""


def build_position_rows(all_tables, name_to_id, competition_id, season_id):
    """Pure: the full finishing distribution as league_position_probabilities
    value tuples.

    For each (team, position): probability = P(finish EXACTLY at position) =
    (sim count at that position / total sims) * 100. Every position 1..N is
    emitted — including the ~0 ones — so the read side's range-sums never hit
    a gap.

    name_to_id : {team name: team_id} scoped to the competition roster — the
        same ambiguous-name guard league_outcome_repo uses ("Liverpool"
        England vs Uruguay etc.).

    Returns (value_tuples, n_teams, n_positions).
    """
    if all_tables is None or len(all_tables) == 0:
        return [], 0, 0

    num_sims = int(all_tables['Simulation'].nunique())
    if num_sims == 0:
        return [], 0, 0

    # Histogram: index=Team, columns=Position, value=sim count.
    pos_pivot = (all_tables.groupby(['Team', 'Position']).size()
                 .unstack(fill_value=0))

    # Each sim ranks all N teams 1..N, so the distribution spans positions
    # 1..N (N = team count). reindex guarantees every position column is
    # present — probability 0.000 where a team never finished — so the
    # stored distribution has no gaps.
    n_teams = len(pos_pivot.index)
    positions = list(range(1, n_teams + 1))
    pos_pivot = pos_pivot.reindex(columns=positions, fill_value=0)

    values = []
    n_teams_written = 0
    for team in pos_pivot.index:
        team_id = name_to_id.get(team)
        if team_id is None:
            logger.warning(f"[league_positions] sim team '{team}' not in competition "
                           f"roster — distribution skipped")
            continue
        n_teams_written += 1
        dist = pos_pivot.loc[team]
        for pos in positions:
            prob = float(dist[pos]) / num_sims * 100.0
            values.append((
                competition_id, season_id, int(team_id), int(pos), round(prob, 3),
            ))
    return values, n_teams_written, len(positions)


async def write_position_probabilities_async(all_tables, teams, comps, league):
    """Compute + write league_position_probabilities for one competition.

    Skips split-format competitions (a single continuous finishing table
    can't represent them). Idempotent per (competition, season): DELETE then
    re-insert, so a team leaving the league doesn't leave a stale row behind.
    """
    competition_id = resolve_competition_id(comps, league)
    season_id = await resolve_current_season_id(competition_id)
    if season_id is None:
        logger.warning(f"[league_positions:{league}] no current season for competition "
                        f"{competition_id} — skipping")
        return 0

    if await is_split_format_competition(competition_id, season_id):
        logger.info(f"[league_positions:{league}] split-format competition — skipping")
        return 0

    # Competition-scoped team name -> id map. The sim keys teams by name;
    # resolving against the full teams table is ambiguous (two clubs both
    # "Liverpool"). The current-season standings roster is the authoritative,
    # unambiguous team set — same guard as league_outcome_repo.
    roster_rows = await fetch_all(
        "SELECT DISTINCT team_id FROM standings "
        "WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    id_to_name = dict(zip(teams['id'], teams['name']))
    name_to_id = {}
    for (tid,) in roster_rows:
        nm = id_to_name.get(tid)
        if nm is not None:
            name_to_id[nm] = int(tid)

    values, n_teams, n_positions = build_position_rows(
        all_tables, name_to_id, competition_id, season_id
    )
    if not values:
        logger.warning(f"[league_positions:{league}] no distribution rows built — skipping write")
        return 0

    # Idempotent per (competition, season).
    await execute(
        "DELETE FROM league_position_probabilities WHERE competition_id = %s AND season_id = %s",
        (competition_id, season_id),
    )
    written = await execute_chunked(
        _INSERT_SQL, values, label=f"[league_position_probabilities:{league}]"
    )
    logger.info(
        f"[league_positions:{league}] wrote {written} rows for season {season_id} — "
        f"{n_teams} teams x {n_positions} positions"
    )
    return written
