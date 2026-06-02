"""
World Cup 2026 fixture projections.

Self-contained — ratings AND projections are both refreshed every run,
mirroring the domestic projection pattern.

Score prediction is the **same code path as domestic** (calls the shared
get_result_probs + find_inputs_for_probs helpers in statz_functions).
The only differences are the ratings source (international_ratings rather
than the domestic get_ratings pipeline) and the per-fixture lambda
construction (single AVG_GOALS + host bonus, no league-specific weightings).

Pipeline:
1. Compute fresh Statz international ratings (TEB v5 FIFA opp + MV v8
   nudge + symmetric caps) for today's date, committing to team_ratings.
2. Cross-Poisson per fixture from those ratings: λ_h = (h.atk/100) ×
   (a.def/100) × AVG_GOALS, λ_a similarly. 1.10× host bonus for
   USA/Canada/Mexico at home.
3. Model 1X2 from get_result_probs(home_goals, away_goals, boost).
4. If bet365 1X2 odds exist: de-vig (divide by overround) + linear
   blend at ODDS_BETA → adjusted 1X2 → find_inputs_for_probs to re-solve
   λs that produce the adjusted 1X2.
5. Downstream markets (CS, O1.5, O2.5, BTTS) from the final Poisson grid.
6. Write to fixture_projections (idempotent — DELETE comp=732 + INSERT).

Skipped: fixtures with unknown team rating (knockout bracket placeholders
like "1st Group A vs 3rd Group C/E/F/H/I").

After the fixture markets: a Monte Carlo tournament simulation, then
per-team stat projections (InternationalTeamStatService → team_projections), then
per-player stat projections (WcPlayerStatService → player_projections,
scoped to nations with a confirmed tournament_squads entry).
"""
import logging
from datetime import date, datetime

import numpy as np
from scipy.stats import poisson

from app.services.international_ratings import compute_international_ratings
from app.services.statz_functions import get_result_probs, find_inputs_for_probs
from app.services.tournament_configs import WC_2026
from app.services.tournament_simulation_service import TournamentSimulator
from app.services.international_team_stat_service import InternationalTeamStatService
from app.services.wc_player_stat_service import WcPlayerStatService
from app.services.wc_fantasy_points_service import WcFantasyPointsService
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("international_projection")

WC_COMP_ID = 732
AVG_GOALS = 1.3
HOST_BONUS = 1.10
HOST_PENALTY = 0.90   # opp playing in a host country: -10% expected goals
HOSTS = {'United States', 'Mexico', 'Canada'}

# odds_beta = bet365 goal-line blend weight (0 = pure model, 1 = pure bookie).
# Bumped to 0.5 alongside the goals odds-blend cascade rewrite 2026-05-29 —
# matches euro_comp_projection_service. boost=1.0 = no draw inflation
# (international games don't need the draw nudge that league football
# gets per-comp via projection_config).
ODDS_BETA = 0.5
BOOST = 1.0


class InternationalProjectionService:
    """Routes any international fixture (national-team comp) through the
    same pipeline. WC is one comp among several — friendlies, qualifiers,
    Euros etc. will be added by registering scopes in INTL_SCOPES (see
    below, introduced in the IntlProjectionScope refactor).

    Stateless — instance method only for parity with EuroCompProjectionService.
    """

    INTERNATIONAL_COMPS = ['World Cup']

    @staticmethod
    def is_international_comp(league: str) -> bool:
        return league in InternationalProjectionService.INTERNATIONAL_COMPS

    async def projections(self, league_request=None, commit: bool = True) -> dict:
        """Compute + (optionally) write WC fixture projections.

        Self-contained: refreshes Statz international ratings inline
        (committing to team_ratings) before computing fixture projections.
        Score prediction uses the same shared helpers as domestic
        (get_result_probs + find_inputs_for_probs).

        Returns a stats dict:
          {n_total, n_projected, n_blended, n_skipped_unknown_team,
           n_ratings, ratings_snapshot_date, committed}
        """
        # Per-fixture mode (set via LeagueRequest.fixture_ids) skips
        # the slow fixture-independent / bracket-wide steps:
        #   - Step 1 (rating refresh): uses last nightly snapshot from
        #     team_ratings instead of recomputing
        #   - Step 3 (tournament simulator): bracket-wide, not per-fixture
        # Steps 2/4/5/6 each filter SQL by the supplied fixture_ids.
        fixture_ids_filter = None
        if league_request is not None and getattr(league_request, 'fixture_ids', None):
            fixture_ids_filter = [int(x) for x in league_request.fixture_ids]

        logger.info(
            f"International projection start — commit={commit}, odds_beta={ODDS_BETA}, "
            f"boost={BOOST}, fixture_ids={fixture_ids_filter}"
        )

        # Step 1: refresh Statz ratings inline (skipped in per-fixture mode).
        ratings: dict
        if fixture_ids_filter:
            # Read latest cached team_ratings rows from team_ratings table
            # without recomputing. Fast — single SQL query.
            _r_conn = await get_source_connection()
            try:
                async with _r_conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT t.name, tr.attack, tr.defense
                        FROM team_ratings tr
                        JOIN teams t ON t.id = tr.team_id
                        WHERE tr.competition_id = %s
                          AND tr.date = (
                              SELECT MAX(date) FROM team_ratings
                              WHERE competition_id = %s AND team_id = tr.team_id
                          )
                        """,
                        (WC_COMP_ID, WC_COMP_ID),
                    )
                    rows = await cur.fetchall()
                ratings = {r[0]: (float(r[1]), float(r[2])) for r in rows}
            finally:
                release_source_connection(_r_conn)
            logger.info(f"Per-fixture mode: loaded {len(ratings)} ratings from cache (skipped recompute)")
        else:
            ratings_date = date.today()
            statz_df, _ = await compute_international_ratings(ratings_date, commit=commit)
            ratings = {
                row['team_name']: (float(row['attack']), float(row['defense']))
                for _, row in statz_df.iterrows()
            } if not statz_df.empty else {}
            logger.info(f"Refreshed {len(ratings)} Statz ratings for {ratings_date}")

        conn = await get_source_connection()
        try:
            async with conn.cursor() as cur:
                # Upcoming WC fixtures + bet365 1X2 odds (LEFT JOIN, null-safe).
                # When fixture_ids_filter is set, scope the SELECT to that
                # list — typically exactly one fixture for the per-fixture
                # re-projection trigger.
                fid_filter_sql = ""
                fid_filter_params: tuple = ()
                if fixture_ids_filter:
                    placeholders = ",".join(["%s"] * len(fixture_ids_filter))
                    fid_filter_sql = f" AND f.id IN ({placeholders})"
                    fid_filter_params = tuple(fixture_ids_filter)
                await cur.execute(
                    f"""
                    SELECT f.id, f.home_team_id, f.away_team_id, f.kickoff_datetime,
                           th.name AS h, ta.name AS a,
                           bo.home_win_odd, bo.draw_odd, bo.away_win_odd
                    FROM fixtures f
                    JOIN teams th ON th.id = f.home_team_id
                    JOIN teams ta ON ta.id = f.away_team_id
                    LEFT JOIN bet365_fixture_odds bo ON bo.fixture_id = f.id
                    WHERE f.competition_id = %s
                      AND f.kickoff_datetime > NOW()
                      AND f.state_id = 1
                      {fid_filter_sql}
                    ORDER BY f.kickoff_datetime
                    """,
                    (WC_COMP_ID,) + fid_filter_params,
                )
                fixtures = await cur.fetchall()
                logger.info(f"Loaded {len(fixtures)} upcoming WC fixtures")

            # Pre-load bet365 goals over/under for these fixtures. The
            # blend cascade (paths 1-3) uses per-team and match-total
            # ladders directly; path 4 (legacy 1X2) is the fallback.
            from app.services.odds_blend import (
                load_goals_odds_for_fixtures,
                compute_final_goals_and_probs,
            )
            wc_fixture_ids = [row[0] for row in fixtures]
            goals_odds_map = await load_goals_odds_for_fixtures(conn, wc_fixture_ids)

            n_blended = 0
            n_skipped_unknown_team = 0
            inserts = []
            for fid, h_tid, a_tid, ko, home, away, oh, od, oa in fixtures:
                if home not in ratings or away not in ratings:
                    n_skipped_unknown_team += 1
                    continue

                # λ from cross-Poisson rating product + host bonus
                h_atk, h_def = ratings[home]
                a_atk, a_def = ratings[away]
                home_goals = (h_atk / 100) * (a_def / 100) * AVG_GOALS
                away_goals = (a_atk / 100) * (h_def / 100) * AVG_GOALS
                if home in HOSTS:
                    home_goals *= HOST_BONUS
                    away_goals *= HOST_PENALTY
                elif away in HOSTS:
                    away_goals *= HOST_BONUS
                    home_goals *= HOST_PENALTY

                # Score prediction via the shared odds-blend cascade.
                # Paths 1-3 consume bet365 goals over/under directly;
                # path 4 (legacy 1X2 blend + reverse-solve) is the
                # fallback; path 5 (no odds at all) leaves model output
                # unchanged.
                bookie_1x2_pct = None
                if oh and od and oa:
                    home_odds_pct = (1.0 / float(oh)) * 100
                    draw_odds_pct = (1.0 / float(od)) * 100
                    away_odds_pct = (1.0 / float(oa)) * 100
                    bookie_margin = 1 + (home_odds_pct + draw_odds_pct + away_odds_pct - 100) / 100
                    bookie_1x2_pct = (
                        home_odds_pct / bookie_margin / 100.0,
                        draw_odds_pct / bookie_margin / 100.0,
                        away_odds_pct / bookie_margin / 100.0,
                    )
                    n_blended += 1

                new_home_goals, new_away_goals, adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = (
                    compute_final_goals_and_probs(
                        fid,
                        float(home_goals), float(away_goals),
                        bookie_1x2_pct,
                        goals_odds_map.get(fid, {}),
                        ODDS_BETA,
                        BOOST,
                    )
                )

                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)
                x = np.arange(0, 9)
                y = np.arange(0, 9)
                X, Y = np.meshgrid(x, y)
                Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)
                over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
                over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
                both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
                # === End shared score-prediction block ===

                inserts.append((
                    fid, h_tid, a_tid,
                    str(round(new_home_goals, 2)), str(round(new_away_goals, 2)),
                    round(adjusted_home_win_prob, 2),
                    round(adjusted_away_win_prob, 2),
                    round(adjusted_draw_prob, 2),
                    round(home_clean_sheet * 100, 2),
                    round(away_clean_sheet * 100, 2),
                    round(over_1_goals, 2),
                    round(over_2_goals, 2),
                    round(both_teams_score_prob, 2),
                    ko,
                ))

            logger.info(
                f"International projection ready: total={len(fixtures)} projected={len(inserts)} "
                f"blended={n_blended} skipped_unknown_team={n_skipped_unknown_team}"
            )

            if commit and inserts:
                async with conn.cursor() as cur:
                    # Per-fixture mode scopes the DELETE to the requested
                    # fixture_ids so we don't wipe the rest of the WC
                    # projection rows. Full-comp mode keeps the original
                    # "delete-all-WC-then-reinsert" idempotent pattern.
                    if fixture_ids_filter:
                        del_ph = ",".join(["%s"] * len(fixture_ids_filter))
                        await cur.execute(
                            f"DELETE FROM fixture_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids_filter),
                        )
                    else:
                        await cur.execute(
                            """
                            DELETE fp FROM fixture_projections fp
                            JOIN fixtures f ON f.id = fp.fixture_id
                            WHERE f.competition_id = %s
                            """,
                            (WC_COMP_ID,),
                        )
                    deleted = cur.rowcount
                    now = datetime.now()
                    rows = [(
                        fid, h_tid, a_tid, hg, ag,
                        hw, aw, dw, hcs, acs, o15, o25, btts,
                        ko, now, now
                    ) for fid, h_tid, a_tid, hg, ag, hw, aw, dw, hcs, acs, o15, o25, btts, ko in inserts]
                    for i in range(0, len(rows), 100):
                        await cur.executemany(
                            """INSERT INTO fixture_projections
                               (fixture_id, home_team_id, away_team_id,
                                home_goals, away_goals,
                                home_win_percent, away_win_percent, draw_percent,
                                home_clean_sheet_percent, away_clean_sheet_percent,
                                over_15_goals_percent, over_25_goals_percent,
                                both_teams_shore_percent,
                                kickoff_datetime, created_at, updated_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            rows[i:i+100],
                        )
                await conn.commit()
                logger.info(f"WROTE: deleted={deleted}, inserted={len(rows)}")
        finally:
            release_source_connection(conn)

        # Step 3: Monte Carlo tournament simulator (bracket-wide).
        # Skipped in per-fixture mode — the sim reads the entire WC
        # fixture projection set and walks the bracket; a single-fixture
        # change doesn't shift bracket outcomes enough to be worth a full
        # 10k-sim rerun. The next nightly full pass will refresh it.
        sim_result = None
        if commit and not fixture_ids_filter:
            sim_result = await TournamentSimulator().run(WC_2026, num_sims=10_000)
            logger.info(f"WC tournament simulation: {sim_result}")

        # Step 4: Per-team stat projections.
        team_stat_result = None
        if commit:
            try:
                team_stat_result = await InternationalTeamStatService().project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"WC team-stat projection: {team_stat_result}")
            except Exception as e:
                logger.exception(f"WC team-stat projection failed: {e}")
                team_stat_result = {'error': str(e)}

        # Step 5: Per-player stat projections.
        player_stat_result = None
        if commit:
            try:
                player_stat_result = await WcPlayerStatService().project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"WC player-stat projection: {player_stat_result}")
            except Exception as e:
                logger.exception(f"WC player-stat projection failed: {e}")
                player_stat_result = {'error': str(e)}

        # Step 6: Per-(fixture, player) WC Fantasy point projections.
        fantasy_points_result = None
        if commit:
            try:
                fantasy_points_result = await WcFantasyPointsService().project(
                    commit=commit, fixture_ids=fixture_ids_filter,
                )
                logger.info(f"WC fantasy points projection: {fantasy_points_result}")
            except Exception as e:
                logger.exception(f"WC fantasy points projection failed: {e}")
                fantasy_points_result = {'error': str(e)}

        return {
            'n_total': len(fixtures),
            'n_projected': len(inserts),
            'n_blended': n_blended,
            'n_skipped_unknown_team': n_skipped_unknown_team,
            'n_ratings': len(ratings),
            'ratings_snapshot_date': 'cached' if fixture_ids_filter else str(ratings_date),
            'fixture_ids': fixture_ids_filter,
            'committed': commit,
            'tournament_simulation': sim_result,
            'team_stat_projection': team_stat_result,
            'player_stat_projection': player_stat_result,
            'fantasy_points_projection': fantasy_points_result,
        }
