"""
World Cup team-stat projections — per-fixture, per-team rolling-avg
regression with half-strength opp adjustment for production volume stats.

Pipeline per upcoming WC fixture (called by InternationalProjectionService.projections()
after the existing fixture-projection + tournament-simulation steps):

  For each team in the fixture:
    For each stat in STAT_LIST (12 stats):
      1. team_history = weighted avg of team's intl fixtures (no game cap).
         Each past fixture's stat value is opp-adjusted at HALF strength
         using the PAST opp's rating at the time of that fixture
         (FIFA-vs-Statz convention handled by _within_fixture_factor).
         Weighting = importance (v4 weights) × decay (0.995^(weeks-3)).
         Mirrors the per-fixture neutralisation in international_ratings.py.
      2. opp_history = weighted avg of opponent's concession of this stat
         in their intl history, same weighting + same half-strength
         per-fixture adjustment but neutralised by the PAST ATTACKER's
         strength (symmetric to side 1).
      3. Regression: model.predict([[team_history, opp_history]]) using
         the "All Leagues" fallback model loaded from disk.

  Post-process:
    - Shots Total / Shots On Target → adjust_shots_projection() w=0.5 to
      anchor with the projected goals value (consistency correction).
    - Goals: overwritten with fixture_projections.home_goals (already
      cross-Poisson + bet365 blended upstream — no regression for Goals).

  Storage: upsert into team_projections (all 12 stats). Successful Passes
  and Interceptions are stored too — the WC player-stat projection
  distributes the team-level value of both per player.

Pooling is across ALL international comps (mirrors international_ratings.py
INTERNATIONAL_COMP_IDS) — not just comp 732. Same importance + decay system
as the international ratings pipeline.
"""
import logging
import math
import os
import pickle
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.repository.team_repo import insert_teams_async
from app.services.statz_functions import adjust_shots_projection
from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("international_team_stats")

# Comp-agnostic constant — team_ratings storage bucket for all
# international football (211 national teams' ratings live under this id).
# Distinct from the comp being projected, which comes from scope.competition_id.
INTL_RATINGS_BUCKET_COMP_ID = 732

# Mirror the rating pipeline's pool of international comps.
INTERNATIONAL_COMP_IDS = [
    732, 720, 711, 714, 717, 723, 726, 729,
    1325, 1326, 1114, 1117, 1118, 1105, 1106, 1538, 1082,
]

# v4 importance weights (same as international_ratings.COMP_IMPORTANCE).
COMP_IMPORTANCE = {
    732: 2.25, 1326: 2.0, 1114: 2.0, 1117: 2.0, 1105: 2.0,
    720: 1.75, 711: 1.75, 714: 1.75, 717: 1.75,
    723: 1.75, 726: 1.75, 729: 1.75,
    1325: 1.5, 1118: 1.5, 1106: 1.5,
    1538: 1.25,
    1082: 0.5,
}
DECAY_WEIGHT = 0.995          # weekly decay base, same as int'l ratings
DECAY_GRACE_WEEKS = 3         # first 3 weeks at full weight (matches ratings)
# No game-count cap — matches international_ratings.py. Decay + importance
# weighting handle the recency, so e.g. a 5-year-old friendly contributes
# ~0.07× weight relative to a recent WCQ.

# Half-strength opp adjustment — blends the raw factor toward 1.0
# (no adjustment). Set 0 to disable, 1.0 to use full strength.
OPP_ADJ_STRENGTH = 0.5

# Caps on opp rating values used in the divisor (mirror ratings caps).
STATZ_OPP_LO = 40.0
STATZ_OPP_HI = 250.0

# Goal-anchor post-process weight (same as domestic).
SHOTS_GOAL_ANCHOR_WEIGHT = 0.5

# Host nations get domestic-style venue effect on team stats; their opp gets
# the inverse effect. Non-host fixtures get no venue effect (neutral).
# Resolved per-scope from scope.hosts at call time — empty frozenset for
# non-host scopes (friendlies, qualifiers etc.) yields no venue effect.

# Main tournaments are at neutral venues (except for the host nation, whose
# group games are at home). Exclude these from a team's H/A venue split.
# Tournament-qualifying comps remain real H/A.
NEUTRAL_TOURNAMENT_COMPS = {732, 1326, 1114, 1117, 1105}

# All Leagues trained model path on the container.
MODEL_DIR = '/app/app/model-builds'

# All 12 stats we project. Tier 1 gets opp-adjusted in the rolling avg;
# Tier 3 stays raw. Goals isn't regression-based (copied from fixture_projections).
STAT_LIST = [
    'Goals',
    'Shots Total',
    'Shots On Target',
    'Corners',
    'Fouls',
    'Yellowcards',
    'Tackles',
    'Passes',
    'Successful Passes',
    'Total Crosses',
    'Interceptions',
    'Offsides',
]
TIER_1_OPP_ADJ = {
    'Shots Total', 'Shots On Target', 'Corners',
    'Total Crosses', 'Passes', 'Successful Passes',
}
GOAL_ANCHORED_STATS = {'Shots Total', 'Shots On Target'}

# Sportmonks stats_type_id mapping (queried 2026-05-15).
STAT_TYPE_IDS = {
    'Goals': 52,
    'Shots Total': 42,
    'Shots On Target': 86,
    'Corners': 34,
    'Fouls': 56,
    'Yellowcards': 84,
    'Tackles': 78,
    'Passes': 80,
    'Successful Passes': 81,
    'Total Crosses': 98,
    'Interceptions': 100,
    'Offsides': 51,
}

# Quality bounds — drop fixture-stat rows outside these before averaging
# (caught during the coverage audit; e.g. "Passes = 5" is corrupt data).
STAT_QUALITY_BOUNDS = {
    'Goals': (0, 12),
    'Shots Total': (1, 40),
    'Shots On Target': (0, 20),
    'Corners': (0, 25),
    'Fouls': (2, 40),
    'Yellowcards': (0, 12),
    'Tackles': (3, 60),
    'Passes': (50, 1000),
    'Successful Passes': (30, 1000),
    'Total Crosses': (0, 50),
    'Interceptions': (1, 40),
    'Offsides': (0, 15),
}

# Stats persisted to team_projections — all 12 of STAT_LIST. Successful
# Passes + Interceptions added 2026-05-18 (migration add_passes_
# interceptions_to_team_projections) so the WC player-stat projection can
# distribute the team-level value of both per player.
STORED_STATS = [
    'Goals', 'Shots Total', 'Shots On Target', 'Corners',
    'Fouls', 'Yellowcards', 'Tackles', 'Passes', 'Successful Passes',
    'Total Crosses', 'Interceptions', 'Offsides',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _within_fixture_factor(opp_rating: Optional[dict], side: str) -> float:
    """Half-strength per-fixture opp adjustment, applied INSIDE the rolling
    weighted average (one factor per historical fixture).

    Sign depends on which side we're computing AND the rating convention:

      side='team' (team's own past output, neutralise by PAST OPP DEFENCE)
        - vs strong defence → boost raw (more credit)
        - vs weak defence   → reduce raw
        - Statz: factor = 100 / opp.defense   (high def = weak → factor<1)
        - FIFA:  factor = opp.overall / 100    (high overall = strong → factor>1)

      side='opp_concession' (opp's past concession, neutralise by PAST ATTACKER)
        - vs strong attacker → reduce raw (it was hard to keep them quiet)
        - vs weak attacker   → boost raw  (concession looks worse)
        - Statz: factor = 100 / past_attacker.attack
        - FIFA:  factor = 100 / past_attacker.overall

    Final factor is blended toward 1.0 by OPP_ADJ_STRENGTH (0.5)."""
    if opp_rating is None:
        return 1.0
    is_fifa = opp_rating.get('is_fifa')
    if side == 'team':
        if is_fifa:
            full = opp_rating['overall'] / 100.0
        else:
            full = 100.0 / max(opp_rating['defense'], 1.0)
    else:  # opp_concession
        if is_fifa:
            full = 100.0 / max(opp_rating['overall'], 1.0)
        else:
            full = 100.0 / max(opp_rating['attack'], 1.0)
    return 1.0 + (full - 1.0) * OPP_ADJ_STRENGTH


def _clip_opp(val: float) -> float:
    return max(STATZ_OPP_LO, min(STATZ_OPP_HI, val))


def _venue_fallback(side: str, venue: str) -> float:
    """1.10/0.90 fallback when a team has <5 fixtures to compute a real ratio.
    Mirrors statz_functions.calculate_team_venue_effect / calculate_opp_venue_effect.
    Opp-concession side is inverted: at HOME, opps tend to concede LESS."""
    if side == 'team':
        return 1.10 if venue == 'H' else 0.90
    else:  # opp_concession
        return 0.90 if venue == 'H' else 1.10


def _calculate_venue_effect(
    team_id: int, stat: str, side: str, venue: str,
    fixtures_df: pd.DataFrame, stats_df: pd.DataFrame,
) -> float:
    """Team's multiplicative venue effect for this stat, computed from intl
    fixtures that had a real home/away venue (i.e. excluding the main
    tournaments where games are at neutral venues per NEUTRAL_TOURNAMENT_COMPS).

    side='team'           → team's own production of stat at H vs A
    side='opp_concession' → team's concession (opp production) at H vs A
    venue='H' or 'A'      → which slot in the upcoming fixture

    Returns (mean_at_venue / mean_overall). Fallback if <5 fixtures: see
    _venue_fallback (1.10 / 0.90 with the right sign per side).
    """
    stat_type_id = STAT_TYPE_IDS[stat]
    lo, hi = STAT_QUALITY_BOUNDS.get(stat, (None, None))

    fx = fixtures_df[
        ((fixtures_df['home_team_id'] == team_id) | (fixtures_df['away_team_id'] == team_id))
        & (~fixtures_df['competition_id'].astype(int).isin(NEUTRAL_TOURNAMENT_COMPS))
    ].copy()
    if fx.empty:
        return _venue_fallback(side, venue)
    fx['team_venue'] = np.where(fx['home_team_id'] == team_id, 'H', 'A')

    s = stats_df[
        (stats_df['fixture_id'].isin(fx['fixture_id'])) & (stats_df['stats_type_id'] == stat_type_id)
    ]
    if side == 'team':
        s = s[s['team_id'] == team_id]
    else:
        s = s[s['team_id'] != team_id]
    s = s.merge(fx[['fixture_id', 'team_venue']], on='fixture_id', how='left')
    if lo is not None:
        s = s[(s['value'] >= lo) & (s['value'] <= hi)]
    if len(s) < 5:
        return _venue_fallback(side, venue)

    s = s.copy()
    s['value'] = s['value'].astype(float)
    overall_mean = float(s['value'].mean())
    if overall_mean == 0:
        return 1.0
    h_subset = s[s['team_venue'] == 'H']['value']
    a_subset = s[s['team_venue'] == 'A']['value']
    if len(h_subset) == 0 or len(a_subset) == 0:
        return _venue_fallback(side, venue)
    h_mean = float(h_subset.mean())
    a_mean = float(a_subset.mean())
    return h_mean / overall_mean if venue == 'H' else a_mean / overall_mean


def _load_all_leagues_models() -> Dict[str, object]:
    """Pre-load the 11 trained regression models (Goals has no model — copied
    from fixture_projections instead). Returns dict stat→sklearn model."""
    models = {}
    for stat in STAT_LIST:
        if stat == 'Goals':
            continue
        path = os.path.join(MODEL_DIR, 'All Leagues', f'All_Leagues_{stat}_model.sav')
        if not os.path.exists(path):
            logger.warning(f"Missing All Leagues model for {stat}: {path}")
            continue
        with open(path, 'rb') as f:
            models[stat] = pickle.load(f)
    logger.info(f"Loaded {len(models)} All Leagues trained models")
    return models


async def _load_data(conn, competition_id: int, fixture_ids_filter=None) -> dict:
    """Pull everything in one round-trip set: international fixtures, team
    stats, team_ratings, upcoming fixtures for the requested competition_id
    (+ their fixture_projections goals lambdas).

    team_ratings always reads from INTL_RATINGS_BUCKET_COMP_ID (732) — the
    rating-storage bucket is comp-agnostic across international football.
    """
    placeholders_comp = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
    placeholders_stat = ",".join(["%s"] * len(STAT_TYPE_IDS))

    # Discard any stale snapshot on this pooled connection — the prior
    # WC step (fixture projections) commits just before this runs, and
    # InnoDB REPEATABLE READ would otherwise pin reads to a pre-commit
    # snapshot.
    await conn.rollback()
    async with conn.cursor() as cur:
        # 1. All finished international fixtures
        await cur.execute(
            f"""
            SELECT id, competition_id, kickoff_datetime, home_team_id, away_team_id
            FROM fixtures
            WHERE competition_id IN ({placeholders_comp})
              AND state_id IN (5, 7, 8)
            """,
            tuple(INTERNATIONAL_COMP_IDS),
        )
        fixtures_rows = await cur.fetchall()

        # 2. Team stats for those fixtures (only the 12 stats we project)
        fixture_ids = [r[0] for r in fixtures_rows]
        if fixture_ids:
            ph_fid = ",".join(["%s"] * len(fixture_ids))
            await cur.execute(
                f"""
                SELECT fixture_id, team_id, stats_type_id, value
                FROM fixture_team_stats
                WHERE fixture_id IN ({ph_fid})
                  AND stats_type_id IN ({placeholders_stat})
                """,
                tuple(fixture_ids) + tuple(STAT_TYPE_IDS.values()),
            )
            stats_rows = await cur.fetchall()
        else:
            stats_rows = []

        # 3. team_ratings (FIFA + Statz)
        await cur.execute(
            """
            SELECT team_id, date, attack, defense, overall, inverse
            FROM team_ratings WHERE competition_id = %s
            ORDER BY team_id, date
            """,
            (INTL_RATINGS_BUCKET_COMP_ID,),
        )
        ratings_rows = await cur.fetchall()

        # 4. Upcoming WC fixtures (with both teams known — skip placeholders).
        # Per-fixture mode scopes the query to fixture_ids_filter so only
        # the requested fixture's row gets projected.
        wc_fid_filter_sql = ""
        wc_fid_filter_params: tuple = ()
        if fixture_ids_filter:
            ph_wcf = ",".join(["%s"] * len(fixture_ids_filter))
            wc_fid_filter_sql = f" AND f.id IN ({ph_wcf})"
            wc_fid_filter_params = tuple(fixture_ids_filter)
        # INNER JOIN on fixture_projections — inherits every gate the
        # orchestrator's fixture loop applied (odds presence, Statz
        # rating, venue classification, FIFA-carry-forward exclusion).
        # If we didn't project the fixture itself, we shouldn't emit
        # team-stat projections for it either.
        await cur.execute(
            f"""
            SELECT f.id, f.kickoff_datetime, f.home_team_id, f.away_team_id,
                   th.name AS home_name, ta.name AS away_name,
                   fp.home_goals, fp.away_goals
            FROM fixtures f
            JOIN teams th ON th.id = f.home_team_id
            JOIN teams ta ON ta.id = f.away_team_id
            INNER JOIN fixture_projections fp ON fp.fixture_id = f.id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              AND f.state_id = 1
              {wc_fid_filter_sql}
            ORDER BY f.kickoff_datetime
            """,
            (competition_id,) + wc_fid_filter_params,
        )
        wc_fixtures_rows = await cur.fetchall()

    fixtures_df = pd.DataFrame(fixtures_rows, columns=[
        'fixture_id', 'competition_id', 'kickoff_datetime',
        'home_team_id', 'away_team_id',
    ])
    fixtures_df['kickoff_datetime'] = pd.to_datetime(fixtures_df['kickoff_datetime'])

    stats_df = pd.DataFrame(stats_rows, columns=['fixture_id', 'team_id', 'stats_type_id', 'value'])
    if not stats_df.empty:
        stats_df['value'] = pd.to_numeric(stats_df['value'], errors='coerce')

    ratings_df = pd.DataFrame(ratings_rows, columns=[
        'team_id', 'date', 'attack', 'defense', 'overall', 'inverse',
    ])
    if not ratings_df.empty:
        ratings_df['date'] = pd.to_datetime(ratings_df['date'])
        for c in ('attack', 'defense', 'overall'):
            ratings_df[c] = pd.to_numeric(ratings_df[c], errors='coerce')

    wc_fixtures = []
    for fid, ko, h_id, a_id, h_name, a_name, h_goals, a_goals in wc_fixtures_rows:
        wc_fixtures.append({
            'fixture_id': fid, 'kickoff_datetime': ko,
            'home_team_id': h_id, 'away_team_id': a_id,
            'home_team_name': h_name, 'away_team_name': a_name,
            'home_goals': float(h_goals) if h_goals is not None else None,
            'away_goals': float(a_goals) if a_goals is not None else None,
        })

    return {
        'fixtures_df': fixtures_df,
        'stats_df': stats_df,
        'ratings_df': ratings_df,
        'wc_fixtures': wc_fixtures,
    }


def _lookup_opp_rating(team_id: int, before_dt, ratings_df: pd.DataFrame) -> Optional[dict]:
    """Most recent team_ratings row for team_id with date < before_dt.

    Returns a dict with `is_fifa` flag — callers MUST check it, since the
    rating-direction convention is opposite between FIFA and Statz rows:
      - FIFA (inverse='Yes'): higher overall = STRONGER team
      - Statz (inverse='No'): higher defense = WEAKER defense (more xGA)"""
    sub = ratings_df[(ratings_df['team_id'] == team_id) & (ratings_df['date'] < before_dt)]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    if row['inverse'] == 'Yes':
        return {'overall': float(row['overall']), 'is_fifa': True}
    return {
        'attack': _clip_opp(float(row['attack'])),
        'defense': _clip_opp(float(row['defense'])),
        'is_fifa': False,
    }


def _compute_history(
    team_id: int,
    stat: str,
    target_dt,
    side: str,  # 'team' or 'opp_concession'
    fixtures_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> Tuple[Optional[float], int]:
    """Compute weighted rolling-avg of (team_id)'s production OR concession of stat.

    side='team'             → team's own past stat output
    side='opp_concession'   → team's past CONCESSION (what opponents produced
                              against them) — used for opp_history input

    Returns (weighted_avg or None, sample_size).
    """
    stat_type_id = STAT_TYPE_IDS[stat]
    lo, hi = STAT_QUALITY_BOUNDS.get(stat, (None, None))

    # Filter fixtures where this team played
    fx = fixtures_df[
        (fixtures_df['home_team_id'] == team_id) | (fixtures_df['away_team_id'] == team_id)
    ].copy()
    fx = fx[fx['kickoff_datetime'] < target_dt]
    if fx.empty:
        return None, 0

    # Get this stat's rows for those fixtures
    fx_ids = fx['fixture_id'].tolist()
    s = stats_df[
        (stats_df['fixture_id'].isin(fx_ids)) & (stats_df['stats_type_id'] == stat_type_id)
    ]

    # For side='team' we want this team's value; for 'opp_concession' we want the OPP's value
    if side == 'team':
        s = s[s['team_id'] == team_id]
    else:
        s = s[s['team_id'] != team_id]
    if s.empty:
        return None, 0

    # Join with fixture metadata
    fx_indexed = fx.set_index('fixture_id')
    s = s.merge(
        fx_indexed[['competition_id', 'kickoff_datetime', 'home_team_id', 'away_team_id']],
        left_on='fixture_id', right_index=True, how='left',
    )

    # Quality filter
    if lo is not None:
        s = s[(s['value'] >= lo) & (s['value'] <= hi)]
    if s.empty:
        return None, 0

    # Identify opponent for each row (for opp adjustment)
    if side == 'team':
        s['opp_id'] = np.where(s['home_team_id'] == team_id, s['away_team_id'], s['home_team_id'])
    else:
        # For concession side, the row's team_id IS the attacker; team_id of THIS side is opp
        s['opp_id'] = s['team_id']  # the past attacker

    # Use ALL qualifying fixtures (no game cap — matches international_ratings.py).
    # Decay × importance handles recency.
    s = s.sort_values('kickoff_datetime').reset_index(drop=True)
    if s.empty:
        return None, 0

    # Per-fixture opp adjustment (half strength) for Tier 1 production-volume
    # stats only. Mirrors how international_ratings.py neutralises per-fixture
    # xG by opp strength. Tier 3 stats (fouls, cards, tackles, ints, offsides)
    # stay raw — opp strength doesn't cleanly predict these.
    target_dt_norm = pd.to_datetime(target_dt)
    if stat in TIER_1_OPP_ADJ:
        adj_values = []
        for _, row in s.iterrows():
            raw_val = float(row['value'])
            opp = _lookup_opp_rating(int(row['opp_id']), pd.to_datetime(row['kickoff_datetime']), ratings_df)
            adj_values.append(raw_val * _within_fixture_factor(opp, side))
        s['adj_value'] = adj_values
    else:
        s['adj_value'] = s['value'].astype(float)

    # Importance × decay weighting
    s['importance'] = s['competition_id'].astype(int).map(COMP_IMPORTANCE).fillna(1.0)
    s['weeks_since'] = ((target_dt_norm - pd.to_datetime(s['kickoff_datetime'])).dt.days // 7).clip(lower=0)
    s['decay'] = DECAY_WEIGHT ** (s['weeks_since'] - DECAY_GRACE_WEEKS).clip(lower=0)
    s['weight'] = s['importance'] * s['decay']
    weight_sum = float(s['weight'].sum())
    if weight_sum <= 0:
        return None, 0
    weighted_avg = float((s['adj_value'] * s['weight']).sum() / weight_sum)
    return weighted_avg, len(s)


def _compute_avg_shots_per_goal(fixtures_df: pd.DataFrame, stats_df: pd.DataFrame) -> Tuple[float, float]:
    """League average shots per goal + SoT per goal across recent intl fixtures.
    Used by adjust_shots_projection() to anchor projections to goals."""
    goals_stat_id = STAT_TYPE_IDS['Goals']
    shots_stat_id = STAT_TYPE_IDS['Shots Total']
    sot_stat_id = STAT_TYPE_IDS['Shots On Target']

    goals = stats_df[stats_df['stats_type_id'] == goals_stat_id]
    shots = stats_df[stats_df['stats_type_id'] == shots_stat_id]
    sot = stats_df[stats_df['stats_type_id'] == sot_stat_id]

    # Inner join goals + shots + sot per (fixture, team)
    merged = goals[['fixture_id', 'team_id', 'value']].rename(columns={'value': 'goals'}).merge(
        shots[['fixture_id', 'team_id', 'value']].rename(columns={'value': 'shots'}),
        on=['fixture_id', 'team_id'],
    ).merge(
        sot[['fixture_id', 'team_id', 'value']].rename(columns={'value': 'sot'}),
        on=['fixture_id', 'team_id'],
    )
    merged = merged[(merged['goals'] > 0) & (merged['shots'] > 0) & (merged['sot'] > 0)]
    if merged.empty:
        return 5.5, 1.9  # safe fallback

    avg_shots_per_goal = (merged['shots'].sum() / merged['goals'].sum())
    avg_sot_per_goal = (merged['sot'].sum() / merged['goals'].sum())
    return float(avg_shots_per_goal), float(avg_sot_per_goal)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class InternationalTeamStatService:
    """Compute + write per-team stat projections for upcoming international
    fixtures. Scope (which comp, which hosts, which bracket-config) is
    driven by the IntlProjectionScope passed at construction.
    """

    def __init__(self, scope=None):
        # Lazy import to avoid the natural circular: international_projection_service
        # imports this class at module-load time.
        if scope is None:
            from app.services.international_projection_service import INTL_SCOPES
            scope = INTL_SCOPES['World Cup']
        self.scope = scope

    async def project(self, commit: bool = True, fixture_ids: list = None) -> dict:
        """fixture_ids: optional list — when set, scope the projection
        to just those fixtures (used by per-fixture re-projection
        triggered on confirmed-lineup arrival)."""
        logger.info(
            f"{self.scope.competition_name} team-stat projection start — "
            f"commit={commit}, opp_adj_strength={OPP_ADJ_STRENGTH} "
            f"(no game cap), fixture_ids={fixture_ids}"
        )

        models = _load_all_leagues_models()
        if len(models) < len(STAT_LIST) - 1:  # -1 because Goals doesn't have one
            logger.warning(
                f"Only {len(models)}/{len(STAT_LIST)-1} regression models loaded — "
                "team-stat projections may be incomplete."
            )

        conn = await get_source_connection()
        try:
            data = await _load_data(
                conn,
                competition_id=self.scope.competition_id,
                fixture_ids_filter=fixture_ids,
            )
            n_fixtures = len(data['fixtures_df'])
            n_stats_rows = len(data['stats_df'])
            n_ratings = len(data['ratings_df'])
            n_wc = len(data['wc_fixtures'])
            logger.info(
                f"Loaded {n_fixtures} intl fixtures, {n_stats_rows} stat rows, "
                f"{n_ratings} ratings rows, {n_wc} upcoming WC fixtures"
            )

            avg_shots_per_goal, avg_sot_per_goal = _compute_avg_shots_per_goal(
                data['fixtures_df'], data['stats_df']
            )
            logger.info(
                f"Intl avg_shots_per_goal={avg_shots_per_goal:.2f}, "
                f"avg_sot_per_goal={avg_sot_per_goal:.2f}"
            )

            # Build set of team_ids that have any team_ratings row — used to
            # filter out knockout-placeholder fixtures (e.g. "Winner Match 73"
            # team rows exist but have no ratings + no history).
            rated_team_ids = (
                set(data['ratings_df']['team_id'].unique().tolist())
                if not data['ratings_df'].empty else set()
            )

            output_rows = []
            n_skipped_placeholder = 0
            n_host_fixtures = 0
            for wc in data['wc_fixtures']:
                # Skip bracket-placeholder fixtures (no rated team).
                if wc['home_team_id'] not in rated_team_ids or wc['away_team_id'] not in rated_team_ids:
                    n_skipped_placeholder += 1
                    continue

                target_dt = pd.to_datetime(wc['kickoff_datetime'])
                home_id = wc['home_team_id']
                away_id = wc['away_team_id']

                # Host-involved fixture? Determines whether venue effect applies.
                # scope.hosts is the per-comp host set — empty for friendlies /
                # qualifiers / Nations League etc., so host_fixture is always
                # False and venue effect is skipped (matches the orchestrator's
                # λ logic which also no-ops on empty scope.hosts).
                home_is_host = wc['home_team_name'] in self.scope.hosts
                away_is_host = wc['away_team_name'] in self.scope.hosts
                host_fixture = home_is_host or away_is_host
                if host_fixture:
                    n_host_fixtures += 1

                for team_id, opp_id, venue, goals_val in (
                    (home_id, away_id, 'H', wc['home_goals']),
                    (away_id, home_id, 'A', wc['away_goals']),
                ):
                    team_name = (wc['home_team_name'] if team_id == home_id else wc['away_team_name'])
                    opp_name = (wc['away_team_name'] if team_id == home_id else wc['home_team_name'])

                    # Determine effective H/A for host-involved fixtures:
                    #   - the host plays at H
                    #   - the host's opponent plays at A
                    # Non-host fixtures get no venue effect (treated as neutral).
                    if host_fixture:
                        team_is_host = team_name in self.scope.hosts
                        team_v = 'H' if team_is_host else 'A'
                        opp_v = 'A' if team_is_host else 'H'
                    else:
                        team_v = None
                        opp_v = None

                    row = {
                        'fixture_id': wc['fixture_id'],
                        'kickoff_datetime': target_dt,
                        'Team': team_name,
                        'Opponent': opp_name,
                        'Venue': venue,
                    }

                    # Goals: copy from fixture_projections (already opp-adjusted upstream).
                    row['Goals'] = round(goals_val, 2) if goals_val is not None else 0.0

                    # Regression-based stats
                    for stat in STAT_LIST:
                        if stat == 'Goals':
                            continue
                        team_history, n_team = _compute_history(
                            team_id, stat, target_dt, 'team',
                            data['fixtures_df'], data['stats_df'], data['ratings_df'],
                        )
                        opp_history, n_opp = _compute_history(
                            opp_id, stat, target_dt, 'opp_concession',
                            data['fixtures_df'], data['stats_df'], data['ratings_df'],
                        )
                        # Fallback: if either side has no data, projection = NaN
                        # and we skip this team-stat (model can't predict).
                        if team_history is None or opp_history is None:
                            row[stat] = 0.0
                            continue

                        # Host-fixture venue effect (domestic-style team-specific
                        # H/A ratios, from non-neutral comps only).
                        if host_fixture:
                            team_eff = _calculate_venue_effect(
                                team_id, stat, 'team', team_v,
                                data['fixtures_df'], data['stats_df'],
                            )
                            opp_eff = _calculate_venue_effect(
                                opp_id, stat, 'opp_concession', opp_v,
                                data['fixtures_df'], data['stats_df'],
                            )
                            team_history *= team_eff
                            opp_history *= opp_eff

                        model = models.get(stat)
                        if model is None:
                            row[stat] = (team_history + opp_history) / 2.0
                        else:
                            row[stat] = float(model.predict([[team_history, opp_history]])[0])

                    output_rows.append(row)

            # Post-process Shots / SoT: goal-anchor consistency correction.
            for row in output_rows:
                team_goals = row['Goals']
                adj_shots, adj_sot = adjust_shots_projection(
                    team_goals,
                    row.get('Shots Total', 0.0),
                    row.get('Shots On Target', 0.0),
                    avg_shots_per_goal,
                    avg_sot_per_goal,
                    weight=SHOTS_GOAL_ANCHOR_WEIGHT,
                )
                row['Shots Total'] = adj_shots
                row['Shots On Target'] = adj_sot

            # Round everything to 2dp
            for row in output_rows:
                for stat in STAT_LIST:
                    if stat in row and row[stat] is not None:
                        row[stat] = round(float(row[stat]), 2)

            logger.info(
                f"{self.scope.competition_name} team-stat projection ready: "
                f"{len(output_rows)} team-fixture rows, "
                f"skipped {n_skipped_placeholder} placeholder fixtures, "
                f"{n_host_fixtures} host-involved fixtures (venue-effect applied)"
            )

            if commit and output_rows:
                # Build the dataframe insert_teams_async expects: columns named
                # 'Team', 'Opponent', 'Venue', 'Goals', plus the 10 stored stat names.
                df = pd.DataFrame(output_rows)
                # Keep only the columns insert_teams_async maps.
                cols_for_insert = [
                    'fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue',
                ] + STORED_STATS
                df = df[[c for c in cols_for_insert if c in df.columns]]
                df['kickoff_datetime'] = pd.to_datetime(df['kickoff_datetime'])

                # ── Team-stat odds-blend ──
                # Reels each team's projected stats (corners/cards/shots/
                # SoT/fouls/tackles) toward bookmaker expected via the
                # cascade. Same WC blend weight as 1X2 goals (ODDS_BETA=0.5).
                # All WC fixtures are bracket-home/away; bookies still
                # price by that role even at neutral venues.
                from app.services.odds_blend import (
                    load_team_stat_odds, blend_team_stat,
                    TEAM_STAT_BOOKIE_PRIORITY, STAT_COLUMN_TO_MARKET,
                )
                from app.services.international_projection_service import ODDS_BETA as _WC_ODDS_BETA
                _fix_ids = df['fixture_id'].astype(int).unique().tolist()
                _odds_per_market = {}
                for _market, _books in TEAM_STAT_BOOKIE_PRIORITY.items():
                    _odds_per_market[_market] = await load_team_stat_odds(
                        conn, _fix_ids, _market, _books,
                    )

                # Build fixture → home_team_name map from data['wc_fixtures'].
                # (data['wc_fixtures'] is the list-of-dicts return from
                # _load_data, with each item carrying 'fixture_id' +
                # 'home_name'. wc_fixtures_rows was the raw SQL list
                # local to _load_data — not available here.)
                _fid_to_home_name = {wc['fixture_id']: wc['home_team_name'] for wc in data['wc_fixtures']}

                _seen = set()
                for _i in range(len(df)):
                    fid = int(df['fixture_id'].iloc[_i])
                    if fid in _seen:
                        continue
                    _seen.add(fid)
                    pair = df[df['fixture_id'] == fid]
                    if len(pair) != 2:
                        continue
                    home_name = _fid_to_home_name.get(fid)
                    if not home_name:
                        continue
                    home_mask = (df['fixture_id'] == fid) & (df['Team'] == home_name)
                    away_mask = (df['fixture_id'] == fid) & (df['Team'] != home_name)

                    for stat_col, market in STAT_COLUMN_TO_MARKET.items():
                        if stat_col not in df.columns:
                            continue
                        try:
                            mh = float(df.loc[home_mask, stat_col].iloc[0])
                            ma = float(df.loc[away_mask, stat_col].iloc[0])
                        except (IndexError, KeyError, ValueError):
                            continue
                        fh, fa = blend_team_stat(
                            mh, ma,
                            _odds_per_market.get(market, {}).get(fid, {}),
                            market, _WC_ODDS_BETA,
                        )
                        df.loc[home_mask, stat_col] = round(fh, 2)
                        df.loc[away_mask, stat_col] = round(fa, 2)

                # Delete existing rows before insert (idempotent — mirrors
                # the international_projection_service pattern). Per-fixture
                # mode scopes the delete to the requested fixtures only so
                # we don't wipe other team_projections rows for this comp.
                async with conn.cursor() as cur:
                    if fixture_ids:
                        del_ph = ",".join(["%s"] * len(fixture_ids))
                        await cur.execute(
                            f"DELETE FROM team_projections WHERE fixture_id IN ({del_ph})",
                            tuple(fixture_ids),
                        )
                    else:
                        await cur.execute(
                            """DELETE tp FROM team_projections tp
                               JOIN fixtures f ON f.id = tp.fixture_id
                               WHERE f.competition_id = %s""",
                            (self.scope.competition_id,),
                        )
                await conn.commit()

                # Use the existing insert helper — no teams/comp_teams since
                # we already have team_id values.
                # Adapt: insert_teams_async expects 'Team' and 'Opponent' names
                # for resolve_team_id. Pass names; the loader resolves them.
                # We need a teams DataFrame for resolve_team_id.
                async with conn.cursor() as cur:
                    await cur.execute("SELECT id, name FROM teams")
                    teams_rows = await cur.fetchall()
                teams_df = pd.DataFrame(teams_rows, columns=['id', 'name'])
                await insert_teams_async(df, teams=teams_df, competition_id=self.scope.competition_id)
                logger.info(
                    f"{self.scope.competition_name} team-stat projections written: {len(df)} rows"
                )

            return {
                'n_team_fixture_rows': len(output_rows),
                'n_wc_fixtures': n_wc,
                'n_skipped_placeholder': n_skipped_placeholder,
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
