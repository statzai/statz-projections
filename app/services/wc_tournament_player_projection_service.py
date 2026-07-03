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

                # 4b. ACTUAL goals/assists already scored in PLAYED games of
                #     THIS tournament. Season-scoped via the kickoff window so
                #     comp 732's previous editions (e.g. 2022) can't leak in.
                #     fixture_player_stats only stores non-zero rows, so SUM =
                #     the player's real tally to date.
                await cur.execute(
                    """
                    SELECT fps.player_id, fps.team_id,
                           SUM(CASE WHEN fps.stats_type_id = %s THEN fps.value ELSE 0 END) AS a_goals,
                           SUM(CASE WHEN fps.stats_type_id = %s THEN fps.value ELSE 0 END) AS a_assists
                    FROM fixture_player_stats fps
                    JOIN fixtures f ON f.id = fps.fixture_id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > '2026-05-01'
                      AND f.kickoff_datetime <= NOW()
                      AND f.state_id IN (5, 7, 8)
                      AND fps.stats_type_id IN (%s, %s)
                    GROUP BY fps.player_id, fps.team_id
                    """,
                    (GOALS_STAT_ID, ASSISTS_STAT_ID, comp_id, GOALS_STAT_ID, ASSISTS_STAT_ID),
                )
                actual_player = {
                    (int(r[0]), int(r[1])): (float(r[2] or 0.0), float(r[3] or 0.0))
                    for r in await cur.fetchall()
                }

                # 4c. ACTUAL team goals scored in played games (from the
                #     fixture scoreline). Used to subtract off the sim's
                #     expected_goals_for, leaving the team's expected REMAINING
                #     goals to distribute over the rest of the tournament.
                await cur.execute(
                    """
                    -- state_id IN (5,7,8): FT, AET, pens. Knockout games that
                    -- finish after extra time land at state 7/8, NOT 5 — a
                    -- state=5-only filter stopped banking those games' goals
                    -- (Tielemans showed 0.45 tournament goals after scoring
                    -- twice in the AET R32 win, 2026-07-03).
                    SELECT team_id, SUM(gf) AS goals FROM (
                        SELECT home_team_id AS team_id, home_team_goals AS gf
                        FROM fixtures
                        WHERE competition_id = %s AND kickoff_datetime > '2026-05-01'
                          AND kickoff_datetime <= NOW() AND state_id IN (5, 7, 8)
                        UNION ALL
                        SELECT away_team_id AS team_id, away_team_goals AS gf
                        FROM fixtures
                        WHERE competition_id = %s AND kickoff_datetime > '2026-05-01'
                          AND kickoff_datetime <= NOW() AND state_id IN (5, 7, 8)
                    ) z GROUP BY team_id
                    """,
                    (comp_id, comp_id),
                )
                team_actual_goals = {int(r[0]): float(r[1] or 0.0) for r in await cur.fetchall()}
        finally:
            release_source_connection(conn)

        # 5. Compute + assemble rows: ACTUAL-to-date + projected REMAINING.
        #
        #   tournament_goals = actual_goals
        #                    + goal_share × max(0, team_EG − team_actual_goals)
        #
        # team_EG (sim) already includes played actuals + sampled remaining, so
        # (team_EG − team_actual) is the team's expected goals STILL TO COME.
        # goal_share is the player's share over the remaining (upcoming) games.
        # Pre-tournament (no actuals) this reduces to the old
        # p_goals × (team_EG / team_group). Assists mirror it with team assists
        # ≈ goals × 0.82 (team_projections has no assists column — same ratio
        # the per-fixture derivation uses).
        ASSIST_PER_GOAL = 0.82

        # Per-team actual assists = Σ players' actual assists on that team.
        team_actual_assists = {}
        for (pid, tid), (a_g, a_a) in actual_player.items():
            team_actual_assists[tid] = team_actual_assists.get(tid, 0.0) + a_a

        # Combine the projected (upcoming) players with the already-scored
        # players — so a finished/eliminated team's scorers still get a row
        # carrying their actual tally (the old code skipped them entirely).
        upcoming = {
            (int(pid), int(tid)): (float(pg or 0.0), float(pa or 0.0))
            for pid, tid, pg, pa in player_grp
        }
        all_keys = set(upcoming) | set(actual_player)

        rows: List[Tuple] = []
        n_teams_used = set()
        n_skipped_no_tournament = 0
        num_sims_seen = None
        for (pid, tid) in all_keys:
            if tid not in team_tour:
                n_skipped_no_tournament += 1
                continue

            p_goals, p_assists = upcoming.get((pid, tid), (0.0, 0.0))   # remaining-fixture projections
            act_g, act_a = actual_player.get((pid, tid), (0.0, 0.0))    # scored to date
            expected_goals, num_sims = team_tour[tid]
            num_sims_seen = num_sims_seen or num_sims

            team_group = team_group_goals.get(tid, 0.0) or 0.0          # team remaining projected goals
            team_act_g = team_actual_goals.get(tid, 0.0)
            team_act_a = team_actual_assists.get(tid, 0.0)

            # Team expected goals/assists STILL TO COME.
            rem_team_goals = max(0.0, expected_goals - team_act_g)
            rem_team_assists = max(0.0, expected_goals * ASSIST_PER_GOAL - team_act_a)

            # Player's share of the remaining output (0 if no upcoming games,
            # e.g. team eliminated — then only the actual tally stands).
            g_share = (p_goals / team_group) if team_group > 0 else 0.0
            team_group_assists = team_group * ASSIST_PER_GOAL
            a_share = (p_assists / team_group_assists) if team_group_assists > 0 else 0.0

            tour_goals = round(act_g + g_share * rem_team_goals, 3)
            tour_assists = round(act_a + a_share * rem_team_assists, 3)

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
            f"(actual+remaining; skipped no_tournament={n_skipped_no_tournament})"
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
            'committed': commit and bool(rows),
        }
