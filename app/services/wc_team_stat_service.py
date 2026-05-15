"""
World Cup team-stat projections — per-fixture, per-team rolling-avg
regression with half-strength opp adjustment for production volume stats.

Pipeline per upcoming WC fixture (called by WcProjectionService.projections()
after the existing fixture-projection + tournament-simulation steps):

  For each team in the fixture:
    For each stat in STAT_LIST (12 stats):
      1. team_history = weighted avg of team's intl fixtures (no game cap),
         weighting = importance (v4 weights) × decay (0.995^(weeks-3)).
         RAW — no per-fixture opp adjustment. Matches what the All Leagues
         model was trained on (see projection_service.get_team_weighted_average).
      2. opp_history = weighted avg of opponent's concession of this stat
         in their intl history, same weighting scheme. Also raw. The model
         learns the team×opp interaction from these two inputs.
      3. Regression: model.predict([[team_history, opp_history]]) using
         the "All Leagues" fallback model loaded from disk.

  Post-process:
    - Tier 1 production-volume stats (Shots / SoT / Corners / Crosses /
      Passes / Successful Passes): apply half-strength opp-strength
      multiplier using the current opp Statz/FIFA rating. Mirrors the
      euro_comp_projection_service pattern (which adjusts the OUTPUT of
      the model, not the inputs).
    - Shots Total / Shots On Target → adjust_shots_projection() w=0.5 to
      anchor with the projected goals value (final consistency correction).
    - Goals: overwritten with fixture_projections.home_goals (already
      cross-Poisson + bet365 blended upstream — no regression for Goals).

  Storage: upsert into team_projections (10 of the 12 stats). The 2 that
  aren't stored (Successful Passes, Interceptions) are computed because
  downstream player-projection logic will need them.

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

logger = logging.getLogger("wc_team_stats")

WC_COMP_ID = 732

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

# Stats persisted to team_projections (the rest are computed for downstream
# player-projection use but not stored in this table).
STORED_STATS = [
    'Goals', 'Shots Total', 'Shots On Target', 'Corners',
    'Fouls', 'Yellowcards', 'Tackles', 'Passes',
    'Total Crosses', 'Offsides',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opp_adj_factor_def(opp_def: float) -> float:
    """Half-strength opp-defense factor for Tier 1 production-volume stats
    (Shots/SoT/Corners/Crosses/Passes/Successful Passes). Applied to the
    MODEL OUTPUT, not the history inputs.

    Statz convention: higher defense rating = MORE xGA conceded (weaker
    defense). So vs weak defense we want MORE volume, vs strong defense
    LESS. Factor = opp_def/100 (full strength), blended toward 1.0 by
    OPP_ADJ_STRENGTH.

    (Note: the ratings pipeline uses the *inverse* form 100/opp_def because
    it's crediting attack output, not predicting volume — different sign.)"""
    full = max(opp_def, 1.0) / 100.0
    return 1.0 + (full - 1.0) * OPP_ADJ_STRENGTH


def _clip_opp(val: float) -> float:
    return max(STATZ_OPP_LO, min(STATZ_OPP_HI, val))


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


async def _load_data(conn) -> dict:
    """Pull everything in one round-trip set: international fixtures, team
    stats, team_ratings, upcoming WC fixtures (+ their fixture_projections
    goals lambdas)."""
    placeholders_comp = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
    placeholders_stat = ",".join(["%s"] * len(STAT_TYPE_IDS))

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
            (WC_COMP_ID,),
        )
        ratings_rows = await cur.fetchall()

        # 4. Upcoming WC fixtures (with both teams known — skip placeholders)
        await cur.execute(
            """
            SELECT f.id, f.kickoff_datetime, f.home_team_id, f.away_team_id,
                   th.name AS home_name, ta.name AS away_name,
                   fp.home_goals, fp.away_goals
            FROM fixtures f
            JOIN teams th ON th.id = f.home_team_id
            JOIN teams ta ON ta.id = f.away_team_id
            LEFT JOIN fixture_projections fp ON fp.fixture_id = f.id
            WHERE f.competition_id = %s
              AND f.kickoff_datetime > NOW()
              AND f.state_id = 1
            ORDER BY f.kickoff_datetime
            """,
            (WC_COMP_ID,),
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
    """Most recent team_ratings row for team_id with date < before_dt."""
    sub = ratings_df[(ratings_df['team_id'] == team_id) & (ratings_df['date'] < before_dt)]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    if row['inverse'] == 'Yes':
        # FIFA row: attack = defense = overall (single opp value)
        v = float(row['overall'])
        return {'attack': v, 'defense': v}
    return {
        'attack': _clip_opp(float(row['attack'])),
        'defense': _clip_opp(float(row['defense'])),
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

    # Build per-row weight (importance × decay). No opp adjustment here —
    # matches what the All Leagues regression model was trained on (domestic
    # get_team_weighted_average is also a raw weighted avg, no opp adj).
    target_dt_norm = pd.to_datetime(target_dt)
    s['importance'] = s['competition_id'].astype(int).map(COMP_IMPORTANCE).fillna(1.0)
    s['weeks_since'] = ((target_dt_norm - pd.to_datetime(s['kickoff_datetime'])).dt.days // 7).clip(lower=0)
    s['decay'] = DECAY_WEIGHT ** (s['weeks_since'] - DECAY_GRACE_WEEKS).clip(lower=0)
    s['weight'] = s['importance'] * s['decay']
    weight_sum = float(s['weight'].sum())
    if weight_sum <= 0:
        return None, 0
    weighted_avg = float((s['value'].astype(float) * s['weight']).sum() / weight_sum)
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

class WcTeamStatService:
    """Compute + write per-team stat projections for upcoming WC fixtures."""

    async def project(self, commit: bool = True) -> dict:
        logger.info(
            f"WC team-stat projection start — commit={commit}, "
            f"opp_adj_strength={OPP_ADJ_STRENGTH} (no game cap)"
        )

        models = _load_all_leagues_models()
        if len(models) < len(STAT_LIST) - 1:  # -1 because Goals doesn't have one
            logger.warning(
                f"Only {len(models)}/{len(STAT_LIST)-1} regression models loaded — "
                "team-stat projections may be incomplete."
            )

        conn = await get_source_connection()
        try:
            data = await _load_data(conn)
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
            for wc in data['wc_fixtures']:
                # Skip bracket-placeholder fixtures (no rated team).
                if wc['home_team_id'] not in rated_team_ids or wc['away_team_id'] not in rated_team_ids:
                    n_skipped_placeholder += 1
                    continue

                target_dt = pd.to_datetime(wc['kickoff_datetime'])
                home_id = wc['home_team_id']
                away_id = wc['away_team_id']

                for team_id, opp_id, venue, goals_val in (
                    (home_id, away_id, 'H', wc['home_goals']),
                    (away_id, home_id, 'A', wc['away_goals']),
                ):
                    team_name = (wc['home_team_name'] if team_id == home_id else wc['away_team_name'])
                    opp_name = (wc['away_team_name'] if team_id == home_id else wc['home_team_name'])

                    # Current opp rating — used for Tier 1 post-process multiplier
                    # (applied to the MODEL OUTPUT, not the history inputs).
                    opp_rating_now = _lookup_opp_rating(opp_id, target_dt, data['ratings_df'])

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
                        model = models.get(stat)
                        if model is None:
                            pred = (team_history + opp_history) / 2.0
                        else:
                            pred = float(model.predict([[team_history, opp_history]])[0])

                        # Tier 1 post-process: half-strength opp-strength multiplier
                        # using current opp rating. Stat output × (1 + (100/opp_def − 1) × 0.5).
                        if stat in TIER_1_OPP_ADJ and opp_rating_now is not None:
                            pred *= _opp_adj_factor_def(opp_rating_now['defense'])

                        row[stat] = pred

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
                f"WC team-stat projection ready: {len(output_rows)} team-fixture rows, "
                f"skipped {n_skipped_placeholder} placeholder fixtures"
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

                # Delete existing WC rows before insert (idempotent — mirrors
                # the wc_projection_service pattern).
                async with conn.cursor() as cur:
                    await cur.execute(
                        """DELETE tp FROM team_projections tp
                           JOIN fixtures f ON f.id = tp.fixture_id
                           WHERE f.competition_id = %s""",
                        (WC_COMP_ID,),
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
                await insert_teams_async(df, teams=teams_df, competition_id=WC_COMP_ID)
                logger.info(f"WC team-stat projections written: {len(df)} rows")

            return {
                'n_team_fixture_rows': len(output_rows),
                'n_wc_fixtures': n_wc,
                'n_skipped_placeholder': n_skipped_placeholder,
                'committed': commit,
            }
        finally:
            release_source_connection(conn)
