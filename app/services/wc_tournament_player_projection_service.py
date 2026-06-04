"""WC tournament-total player projections — Step 7 of InternationalProjectionService.

Aggregates per-fixture player projections (from `player_projections`) and
scales them by the team's full-tournament expected goals (from
`tournament_projections`). Writes per-(competition, player) totals to
`wc_tournament_player_projections`.

The math collapses to a single scaling factor per team:

    multiplier = team.expected_goals_for / SUM(team_projections.goals across group)

    tournament_goals   = SUM(player.goals across group)   × multiplier
    tournament_assists = SUM(player.assists across group) × multiplier

Same multiplier for both because the 0.82 derived-assist ratio cancels: at
the team level, group-stage assists are derived as team_goals × 0.82 and the
tournament-level derived assists use the same 0.82 — both share the 0.82,
so the ratio of (tournament / group) is just `expected_goals_for /
team_group_goals`.

Gated on `scope.bracket_config is not None` — same condition as the
tournament_simulator, since both need the bracket Monte Carlo output.
Friendlies / qualifiers / Nations League skip this step.

Idempotent: DELETE all comp-scoped rows from wc_tournament_player_projections,
then bulk-insert the new batch.
"""
import logging
from typing import List, Tuple

from app.repository.wc_tournament_player_repo import (
    insert_wc_tournament_player_projections_async,
)
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_tournament_player")

GOALS_STAT_ID = 52
ASSISTS_STAT_ID = 79


class WcTournamentPlayerProjectionService:
    """Compute + write tournament-total goals + assists per player.

    Stateless — instance method only for parity with other intl projection
    services.
    """

    def __init__(self, scope):
        self.scope = scope

    async def project(self, commit: bool = True) -> dict:
        """Compute per-player tournament_goals + tournament_assists from
        existing per-fixture player_projections + team_projections and the
        team-level expected_goals_for in tournament_projections.

        Returns stats dict: {n_rows, n_teams, num_sims, committed}.
        """
        comp_id = self.scope.competition_id

        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                # 1. Team-level: tournament expected_goals_for + num_sims
                #    Source of truth for the multiplier denominator's other
                #    half (group goals) is summed below from team_projections
                #    to stay self-consistent with the per-fixture projections.
                await cur.execute(
                    """
                    SELECT team_id, expected_goals_for, num_sims
                    FROM tournament_projections
                    WHERE competition_id = %s
                    """,
                    (comp_id,),
                )
                tp_rows = await cur.fetchall()
                if not tp_rows:
                    logger.warning(
                        f"{self.scope.competition_name}: no tournament_projections rows for "
                        f"comp {comp_id} — skipping tournament player projections"
                    )
                    return {'n_rows': 0, 'n_teams': 0, 'num_sims': None, 'committed': False}
                # team_id -> (expected_goals_for, num_sims)
                team_tour = {int(t[0]): (float(t[1]), int(t[2])) for t in tp_rows}

                # 2. Team-projected goals summed across upcoming group fixtures
                #    per (team_id). Denominator for the multiplier.
                await cur.execute(
                    """
                    SELECT tp.team_id, SUM(tp.goals) AS group_goals
                    FROM team_projections tp
                    JOIN fixtures f ON f.id = tp.fixture_id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > NOW()
                    GROUP BY tp.team_id
                    """,
                    (comp_id,),
                )
                team_group_goals = {int(t[0]): float(t[1]) for t in await cur.fetchall()}

                # 3. Player-projected goals + assists summed across upcoming
                #    group fixtures per (player_id, team_id). One row per
                #    player per WC (multi-team within a tournament not
                #    possible — players are nation-locked).
                await cur.execute(
                    """
                    SELECT pp.player_id, pp.team_id,
                           SUM(CASE WHEN pp.stats_type_id = %s THEN pp.stats_value ELSE 0 END) AS p_goals,
                           SUM(CASE WHEN pp.stats_type_id = %s THEN pp.stats_value ELSE 0 END) AS p_assists
                    FROM player_projections pp
                    JOIN fixtures f ON f.id = pp.fixture_id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > NOW()
                    GROUP BY pp.player_id, pp.team_id
                    """,
                    (GOALS_STAT_ID, ASSISTS_STAT_ID, comp_id),
                )
                player_grp = await cur.fetchall()

                # 4. Position lookup from wc_players (carried into our
                #    output so leaderboards can filter by position without
                #    re-joining).
                await cur.execute(
                    """
                    SELECT player_id, position FROM wc_players
                    WHERE status = 'playing' AND player_id IS NOT NULL
                    """
                )
                positions = {int(r[0]): r[1] for r in await cur.fetchall()}
        finally:
            release_source_connection(conn)

        # 5. Compute + assemble rows.
        rows: List[Tuple] = []
        n_teams_used = set()
        n_skipped_no_tournament = 0
        n_skipped_no_group_goals = 0
        num_sims_seen = None
        for player_id, team_id, p_goals, p_assists in player_grp:
            pid, tid = int(player_id), int(team_id)
            p_goals = float(p_goals or 0.0)
            p_assists = float(p_assists or 0.0)

            if tid not in team_tour:
                n_skipped_no_tournament += 1
                continue
            team_group = team_group_goals.get(tid)
            if not team_group or team_group <= 0:
                n_skipped_no_group_goals += 1
                continue

            expected_goals, num_sims = team_tour[tid]
            num_sims_seen = num_sims_seen or num_sims
            multiplier = expected_goals / team_group

            tour_goals = round(p_goals * multiplier, 3)
            tour_assists = round(p_assists * multiplier, 3)

            rows.append((
                comp_id,
                pid,
                tid,
                positions.get(pid),  # nullable
                tour_goals,
                tour_assists,
                num_sims,
            ))
            n_teams_used.add(tid)

        logger.info(
            f"{self.scope.competition_name} tournament player projections ready: "
            f"{len(rows)} rows across {len(n_teams_used)} teams "
            f"(skipped: no_tournament={n_skipped_no_tournament} "
            f"no_group_goals={n_skipped_no_group_goals})"
        )

        # 6. Write — DELETE comp rows + bulk INSERT.
        if commit and rows:
            conn = await get_source_connection()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM wc_tournament_player_projections WHERE competition_id = %s",
                        (comp_id,),
                    )
                    deleted = cur.rowcount
                await conn.commit()
            finally:
                release_source_connection(conn)
            inserted = await insert_wc_tournament_player_projections_async(rows)
            logger.info(
                f"{self.scope.competition_name} tournament player projections written: "
                f"deleted={deleted} inserted={inserted}"
            )

        return {
            'n_rows': len(rows),
            'n_teams': len(n_teams_used),
            'num_sims': num_sims_seen,
            'n_skipped_no_tournament': n_skipped_no_tournament,
            'n_skipped_no_group_goals': n_skipped_no_group_goals,
            'committed': commit and bool(rows),
        }
