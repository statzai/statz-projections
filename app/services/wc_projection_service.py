"""
World Cup 2026 fixture projections.

Self-contained — ratings AND projections are both refreshed every run,
mirroring the domestic projection pattern.

Pipeline:
1. Compute fresh Statz ratings (TEB v5 + MV v8 + symmetric caps) for
   today's date via international_ratings.compute_quarterly_snapshot,
   committing to team_ratings (comp 732, inverse='No', date=today).
2. Cross-Poisson per fixture from those ratings: λ_h = (h.atk/100) ×
   (a.def/100) × AVG_GOALS, λ_a similarly. 1.10× host bonus for
   USA/Canada/Mexico at home.
3. Compute model 1X2 + downstream markets (BTTS, O1.5, O2.5, CS) from
   the Poisson grid (max 8 goals each side).
4. If bet365 1X2 odds exist on the fixture: de-vig + linear blend at
   β=0.3 → blended 1X2 → scipy.fsolve for new (λ_h, λ_a) that match
   blended 1X2 → recompute downstream markets from the new λs.
   Mirrors domestic projection_service.py:947-965.
5. Write to fixture_projections (idempotent — DELETE comp=732 + INSERT).

Skipped: fixtures with unknown team rating (knockout bracket placeholders
like "1st Group A vs 3rd Group C/E/F/H/I").

No team-level / player-level projections yet — those are deferred until
squad announcements (~late May 2026). Service writes only fixture-level
markets.
"""
import logging
import math
from datetime import date, datetime
from typing import Tuple

import numpy as np
from scipy.optimize import fsolve

from app.services.international_ratings import compute_international_ratings
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("wc_projection")

WC_COMP_ID = 732
AVG_GOALS = 1.3
HOST_BONUS = 1.10
HOSTS = {'United States', 'Mexico', 'Canada'}
ODDS_BETA = 0.3       # bet365 1X2 blend weight (domestic default)
MAX_GOALS = 8         # Poisson grid cap


def _poisson_pmf(lam: float) -> list:
    return [math.exp(-lam) * lam**k / math.factorial(k) for k in range(MAX_GOALS + 1)]


def _poisson_markets(lam_h: float, lam_a: float) -> dict:
    """Compute all 1X2 + BTTS + OU + CS probabilities from Poisson grid."""
    p_h = _poisson_pmf(lam_h)
    p_a = _poisson_pmf(lam_a)
    home_win = draw = away_win = 0.0
    over_15 = over_25 = btts = 0.0
    home_cs = away_cs = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = p_h[h] * p_a[a]
            if h > a: home_win += p
            elif h == a: draw += p
            else: away_win += p
            if h + a >= 2: over_15 += p
            if h + a >= 3: over_25 += p
            if h >= 1 and a >= 1: btts += p
            if a == 0: home_cs += p
            if h == 0: away_cs += p
    return {
        'home_win': home_win, 'draw': draw, 'away_win': away_win,
        'over_15': over_15, 'over_25': over_25, 'btts': btts,
        'home_cs': home_cs, 'away_cs': away_cs,
    }


def _solve_lambdas_for_probs(target_h: float, target_a: float,
                              lam_h0: float, lam_a0: float) -> Tuple[float, float]:
    """Find (λ_h, λ_a) producing target P(home), P(away). Returns starting
    guess if numerical solver fails to converge."""
    def f(params):
        lh = max(0.01, params[0])
        la = max(0.01, params[1])
        m = _poisson_markets(lh, la)
        return [m['home_win'] - target_h, m['away_win'] - target_a]
    try:
        sol, info, ier, msg = fsolve(f, [lam_h0, lam_a0], full_output=True)
        if ier != 1:
            return lam_h0, lam_a0
        return max(0.01, float(sol[0])), max(0.01, float(sol[1]))
    except Exception:
        return lam_h0, lam_a0


class WcProjectionService:
    """Stateless — instance method only for parity with EuroCompProjectionService."""

    async def projections(self, commit: bool = True) -> dict:
        """Compute + (optionally) write WC fixture projections.

        Self-contained: refreshes Statz ratings inline (committing to
        team_ratings) before computing fixture projections.

        Returns a stats dict:
          {n_total, n_projected, n_blended, n_skipped_unknown_team,
           n_ratings, ratings_snapshot_date}
        """
        logger.info(f"WC projection start — commit={commit}, β={ODDS_BETA}")

        # Step 1: refresh Statz ratings inline (always commit to team_ratings
        # so the snapshot is persisted for later inspection / accuracy
        # comparisons / future-fixture opp lookups). Mirrors the domestic
        # projection pattern where ratings are computed every run.
        ratings_date = date.today()
        statz_df, _ = await compute_international_ratings(ratings_date, commit=True)
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
                    # Knockout bracket placeholders ("1st Group A" etc) or
                    # genuine unknowns — skip rather than write garbage.
                    n_skipped_unknown_team += 1
                    continue

                h_atk, h_def = ratings[home]
                a_atk, a_def = ratings[away]
                lam_h = (h_atk / 100) * (a_def / 100) * AVG_GOALS
                lam_a = (a_atk / 100) * (h_def / 100) * AVG_GOALS
                if home in HOSTS:
                    lam_h *= HOST_BONUS
                elif away in HOSTS:
                    lam_a *= HOST_BONUS

                m = _poisson_markets(lam_h, lam_a)

                # bet365 1X2 blend (null-safe)
                if oh and od and oa:
                    ih = 1.0 / float(oh); id_ = 1.0 / float(od); ia = 1.0 / float(oa)
                    margin = ih + id_ + ia
                    mh_imp = ih / margin
                    ma_imp = ia / margin
                    blend_h = m['home_win'] + (mh_imp - m['home_win']) * ODDS_BETA
                    blend_a = m['away_win'] + (ma_imp - m['away_win']) * ODDS_BETA
                    new_lh, new_la = _solve_lambdas_for_probs(blend_h, blend_a, lam_h, lam_a)
                    m_new = _poisson_markets(new_lh, new_la)
                    if abs(m_new['home_win'] - blend_h) < 0.005 and abs(m_new['away_win'] - blend_a) < 0.005:
                        lam_h, lam_a = new_lh, new_la
                        m = m_new
                        n_blended += 1

                inserts.append((
                    fid, h_tid, a_tid,
                    str(round(lam_h, 2)), str(round(lam_a, 2)),
                    round(m['home_win'] * 100, 2),
                    round(m['away_win'] * 100, 2),
                    round(m['draw'] * 100, 2),
                    round(m['home_cs'] * 100, 2),
                    round(m['away_cs'] * 100, 2),
                    round(m['over_15'] * 100, 2),
                    round(m['over_25'] * 100, 2),
                    round(m['btts'] * 100, 2),
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
