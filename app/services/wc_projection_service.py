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

No team-level / player-level projections yet — those are deferred until
squad announcements (~late May 2026). Service writes only fixture-level
markets.
"""
import logging
from datetime import date, datetime

import numpy as np
from scipy.stats import poisson

from app.services.international_ratings import compute_international_ratings
from app.services.statz_functions import get_result_probs, find_inputs_for_probs
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_projection")

WC_COMP_ID = 732
AVG_GOALS = 1.3
HOST_BONUS = 1.10
HOSTS = {'United States', 'Mexico', 'Canada'}

# Match domestic projection_service.py defaults. odds_beta=0.3 = bet365 1X2
# blend weight; boost=1.0 = no draw inflation (international games don't
# need the draw nudge that league football gets per-comp via projection_config).
ODDS_BETA = 0.3
BOOST = 1.0


class WcProjectionService:
    """Stateless — instance method only for parity with EuroCompProjectionService."""

    WC_COMPS = ['World Cup']

    @staticmethod
    def is_wc_comp(league: str) -> bool:
        return league in WcProjectionService.WC_COMPS

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
        logger.info(f"WC projection start — commit={commit}, odds_beta={ODDS_BETA}, boost={BOOST}")

        # Step 1: refresh Statz ratings inline.
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
                # Upcoming WC fixtures + bet365 1X2 odds (LEFT JOIN, null-safe)
                await cur.execute(
                    """
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
                    ORDER BY f.kickoff_datetime
                    """,
                    (WC_COMP_ID,),
                )
                fixtures = await cur.fetchall()
                logger.info(f"Loaded {len(fixtures)} upcoming WC fixtures")

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
                elif away in HOSTS:
                    away_goals *= HOST_BONUS

                # === Score prediction — same code path as projection_service.py:947-980 ===
                if oh and od and oa:
                    # Implied % (with vig), de-vig by dividing by overround
                    home_odds_pct = (1.0 / float(oh)) * 100
                    draw_odds_pct = (1.0 / float(od)) * 100
                    away_odds_pct = (1.0 / float(oa)) * 100
                    bookie_margin = 1 + (home_odds_pct + draw_odds_pct + away_odds_pct - 100) / 100
                    home_odds_pct /= bookie_margin
                    draw_odds_pct /= bookie_margin
                    away_odds_pct /= bookie_margin

                    home_win_prob, draw_prob, away_win_prob = get_result_probs(home_goals, away_goals, BOOST)
                    adjusted_home_win_prob = home_win_prob + ((home_odds_pct - home_win_prob) * ODDS_BETA)
                    adjusted_draw_prob = draw_prob + ((draw_odds_pct - draw_prob) * ODDS_BETA)
                    adjusted_away_win_prob = away_win_prob + ((away_odds_pct - away_win_prob) * ODDS_BETA)
                    new_home_goals, new_away_goals = find_inputs_for_probs(
                        home_goals, away_goals,
                        adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob,
                        BOOST,
                    )
                    n_blended += 1
                else:
                    new_home_goals = home_goals
                    new_away_goals = away_goals
                    adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = get_result_probs(
                        home_goals, away_goals, BOOST
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
                f"WC projection ready: total={len(fixtures)} projected={len(inserts)} "
                f"blended={n_blended} skipped_unknown_team={n_skipped_unknown_team}"
            )

            if commit and inserts:
                async with conn.cursor() as cur:
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

            return {
                'n_total': len(fixtures),
                'n_projected': len(inserts),
                'n_blended': n_blended,
                'n_skipped_unknown_team': n_skipped_unknown_team,
                'n_ratings': len(ratings),
                'ratings_snapshot_date': str(ratings_date),
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
