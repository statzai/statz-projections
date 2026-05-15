"""
International team ratings (nations).

Computes per-nation Attack + Defense from international fixtures and writes
them on a mean=100 rescaled scale (Attack/Defense each have mean 100 across
all qualifying teams), with raw weighted-xG pre-rescale also stored.

Pipeline (settled 2026-05-14):
  1. xG blend per fixture: 0.3 * goals + 0.7 * xG (both sides)
  2. Opp adjustment, branched on team_ratings.inverse flag:
       FIFA (inverse='Yes'): adj_g  = blend_g  * (opp.overall / 100)
                             adj_ga = blend_ga / (opp.overall / 100)
       Statz (inverse='No'): adj_g  = blend_g  / (clip(opp.defense, 40, 250) / 100)
                             adj_ga = blend_ga / (clip(opp.attack,  40, 250) / 100)
     Symmetric caps [40, 250] mirror the Top End Buff v5 FIFA curve range
     so both paths give the same [0.40, 2.50] multiplier ranges.
  3. Soft cap on adj_g / adj_ga: T=3, M=5, scale=2 exponential dampen.
  4. Weighted mean per team across fixtures, with importance × decay weights:
       Attack_xg  = Σ(adj_g  * weight) / Σ(weight)
       Defense_xg = Σ(adj_ga * weight) / Σ(weight)
  5. MV v8 nudge — Transfermarkt market value pulls atk_xg / def_xg toward
     a per-team target proportional to log10(MV). β=0.20. Cap=3.25.
  6. Rescale to cross-team mean=100 separately for Attack and Defense.

Eligibility:
  - ≥ MIN_GAMES (10) total fixtures with xG data
  - ≥ MIN_COMPETITIVE_GAMES (3) non-friendly fixtures
  - Below either → FIFA carry-forward row (inverse='Yes', attack/defense
    copied from the most recent FIFA snapshot ≤ target_date).

Opp lookup is time-aware: each fixture uses the most recent ratings row
(FIFA or Statz) strictly before its kickoff date.
"""
import json
import logging
import math
import os
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("international_ratings")

# All international competition IDs we pool fixtures from.
INTERNATIONAL_COMP_IDS = [
    732,   # World Cup
    720,   # WC Qualification Europe
    711,   # CAF World Cup Qualifiers
    714,   # WC Qualification Asia
    717,   # WC Qualification Concacaf
    723,   # WC Qualification Oceania
    726,   # WC Qualification South America
    729,   # WC Qualification Intercontinental Playoffs
    1325,  # Euro Qualification
    1326,  # European Championship
    1114,  # Copa America
    1117,  # Africa Cup of Nations
    1118,  # Africa Cup of Nations Qualifications
    1105,  # AFC Asian Cup
    1106,  # Asian Cup Qualification
    1538,  # UEFA Nations League
    1082,  # Friendly International
]

# Per-competition importance multiplier (v4 settled 2026-05-14).
COMP_IMPORTANCE = {
    732:  2.25,  # World Cup
    1326: 2.0,   # European Championship
    1114: 2.0,   # Copa America
    1117: 2.0,   # Africa Cup of Nations
    1105: 2.0,   # AFC Asian Cup
    720:  1.75,  # WC Qual Europe
    711:  1.75,  # WC Qual CAF
    714:  1.75,  # WC Qual Asia
    717:  1.75,  # WC Qual Concacaf
    723:  1.75,  # WC Qual Oceania
    726:  1.75,  # WC Qual South America
    729:  1.75,  # WC Qual Intercontinental Playoffs
    1325: 1.5,   # Euro Qualification
    1118: 1.5,   # Africa Cup of Nations Qualifications
    1106: 1.5,   # Asian Cup Qualification
    1538: 1.25,  # UEFA Nations League
    1082: 0.5,   # Friendly International
}

FRIENDLY_COMP_ID = 1082

# Per-week recency decay. Half-life ~138 weeks (~2.7 years).
DECAY_WEIGHT = 0.995

# Soft cap on adj_g / adj_ga: leave ≤ T alone, asymptote to M.
# T=3 means goal-rates up to 3 pass through; above 3 dampens toward 5.
SOFT_CAP_T = 3.0
SOFT_CAP_M = 5.0
SOFT_CAP_SCALE = 2.0

# Symmetric caps on Statz opp.attack / opp.defense lookups (mirrors TEB v5 FIFA range).
# Multiplier range: 40/100 → 250/100 = 0.40 to 2.50 (same on both paths).
STATZ_OPP_LO = 40.0
STATZ_OPP_HI = 250.0

# MV v8 anchors — log10(avg market value in millions €) → mv_idx multiplier
# (1.0 = neutral, France caps at 3.25, minnows floor at 0.25).
MV_LOG_PTS = [-2.50, -2.00, -1.50, -1.00, -0.50, -0.22,  0.00,  0.50,  1.00,  1.30,  1.50,  1.65,  1.74]
MV_IDX     = [ 0.25,  0.30,  0.35,  0.55,  0.80,  1.00,  1.10,  1.25,  1.75,  2.00,  2.50,  3.00,  3.25]
INTL_MV_BETA = 0.20
_MV_PCHIP = PchipInterpolator(MV_LOG_PTS, MV_IDX, extrapolate=False)

# Cached Transfermarkt scrape — kept in the volume-mounted data folder so it
# survives container rebuilds. Previously at /tmp/ which is ephemeral; the
# 2026-05-14 docker rebuild wiped it and the next WC run silently skipped
# the MV nudge step. Seed by running mv_iter.py or by running the TM scrape
# block in curve_to_wc_projection.py.
TM_MV_CACHE = '/app/app/data/tm_mv_cache.json'

# Transfermarkt team name → Statz teams.name aliases (kept in sync with mv_iter.py).
TM_TO_STATZ = {
    'Turkiye':'Turkey','Türkiye':'Turkey','Czechia':'Czech Republic',
    'South Korea':'Korea Republic','North Korea':'Korea DPR',
    'Cape Verde':'Cape Verde Islands','Cabo Verde':'Cape Verde Islands',
    'IR Iran':'Iran','Curaçao':'Curacao','Macau':'Macao',
    'Timor-Leste':'East Timor','Ivory Coast':"Côte d'Ivoire",
    'Hong Kong, China':'Hong Kong',
    'Saint Kitts and Nevis':'St. Kitts and Nevis','Saint Lucia':'St. Lucia',
    'Saint Vincent and the Grenadines':'St. Vincent and the Grenadines',
    'Bosnia-Herzegovina':'Bosnia and Herzegovina',
    'Saudi-Arabia':'Saudi Arabia','New-Zealand':'New Zealand',
    'Democratic Republic of the Congo':'Congo DR','DR Congo':'Congo DR',
    'The Gambia':'Gambia','China':'China PR',
    'Kyrgyzstan':'Kyrgyz Republic','Republic of the Congo':'Congo',
}

# All international snapshots stored under World Cup comp_id.
RATINGS_COMP_ID = 732

# Sportmonks stats_type_id for Expected Goals (xG) on fixture_team_stats.
XG_STAT_TYPE_ID = 5304

# Finished-match states: 5 = Full Time, 7 = After Extra Time,
# 8 = After Penalty Shootout. Knockout games that go to ET/pens are
# recorded as 7/8 (e.g. Euro 2024, AFCON, Copa knockouts) — must be
# included or all those high-importance games get silently dropped.
#
# DELIBERATELY EXCLUDED — do not add these to STATE_FINISHED:
#   state_id 10 = postponed / cancelled
#   state_id 17 = awarded / walkover (administrative result, on-pitch
#     performance is competitively void — worse than no data point)
STATE_FINISHED = (5, 7, 8)

# Eligibility for Statz Rating (below either threshold → FIFA carry-forward).
MIN_GAMES_FOR_STATZ = 10
MIN_COMPETITIVE_GAMES = 3


def soft_cap(x: float) -> float:
    """Smooth dampen above SOFT_CAP_T, asymptoting toward SOFT_CAP_M."""
    if pd.isna(x):
        return x
    if x <= SOFT_CAP_T:
        return x
    return SOFT_CAP_T + (SOFT_CAP_M - SOFT_CAP_T) * (1 - math.exp(-(x - SOFT_CAP_T) / SOFT_CAP_SCALE))


def mv_curve(log_mv_m: float) -> float:
    """Map log10(market_value_in_millions_€) → mv_idx multiplier via MV v8 PCHIP curve."""
    if log_mv_m <= MV_LOG_PTS[0]:
        return float(MV_IDX[0])
    if log_mv_m >= MV_LOG_PTS[-1]:
        return float(MV_IDX[-1])
    return float(_MV_PCHIP(log_mv_m))


def load_mv_values() -> Optional[pd.DataFrame]:
    """Load Transfermarkt MV values from local cache, mapped to Statz team names.

    Returns DataFrame with columns: statz_name, avg_mv_m, mv_idx. None if
    cache missing — caller skips the MV nudge step and logs a warning.
    """
    if not os.path.exists(TM_MV_CACHE):
        logger.warning(
            f"TM MV cache missing at {TM_MV_CACHE} — skipping MV nudge step. "
            f"Seed the cache by running mv_iter.py or curve_to_wc_projection.py."
        )
        return None
    with open(TM_MV_CACHE) as f:
        rows = json.load(f)
    mv_df = pd.DataFrame([r for r in rows if r.get('avg_mv', 0) > 0])
    if mv_df.empty:
        logger.warning(f"TM MV cache at {TM_MV_CACHE} is empty — skipping MV nudge.")
        return None
    mv_df['avg_mv_m'] = mv_df['avg_mv'] / 1e6
    mv_df['log_mv'] = np.log10(mv_df['avg_mv_m'].clip(lower=1e-4))
    mv_df['mv_idx'] = mv_df['log_mv'].apply(mv_curve)
    mv_df['statz_name'] = mv_df['team'].map(TM_TO_STATZ).fillna(mv_df['team'])
    return mv_df[['statz_name', 'avg_mv_m', 'mv_idx']]


async def _load_opp_rows(conn) -> pd.DataFrame:
    """Load every team_ratings row for comp 732 (both FIFA inverse='Yes' and
    Statz inverse='No') for time-aware opp lookups.

    Returns columns: team_id, attack, defense, overall, date, inverse.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT team_id, attack, defense, overall, date, inverse
            FROM team_ratings
            WHERE competition_id = %s
            ORDER BY team_id, date
            """,
            (RATINGS_COMP_ID,),
        )
        rows = await cur.fetchall()
    df = pd.DataFrame(rows, columns=['team_id', 'attack', 'defense', 'overall', 'date', 'inverse'])
    for c in ('attack', 'defense', 'overall'):
        df[c] = df[c].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    return df


async def _load_int_fixtures(conn, date_to: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load every finished int'l fixture in DB up to date_to + their xG rows."""
    placeholders = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
    state_ph = ",".join(["%s"] * len(STATE_FINISHED))
    async with conn.cursor() as cur:
        # A fixture counts as played if it's in a finished state, OR it has a
        # final score and isn't cancelled (~150 AFCON-Qual-2024 rows have
        # state_id = NULL despite having results — Sportmonks import gap).
        await cur.execute(
            f"""
            SELECT f.id, f.competition_id, f.kickoff_datetime,
                   f.home_team_id, f.away_team_id,
                   f.home_team_goals AS home_score, f.away_team_goals AS away_score
            FROM fixtures f
            WHERE f.competition_id IN ({placeholders})
              AND f.kickoff_datetime <= %s
              AND ( f.state_id IN ({state_ph})
                    OR (f.state_id IS NULL AND f.home_team_goals IS NOT NULL) )
            """,
            tuple(INTERNATIONAL_COMP_IDS) + (date_to,) + STATE_FINISHED,
        )
        rows = await cur.fetchall()
    fixtures = pd.DataFrame(rows, columns=[
        'fixture_id', 'competition_id', 'kickoff_datetime',
        'home_team_id', 'away_team_id', 'home_score', 'away_score',
    ])
    if fixtures.empty:
        return fixtures, pd.DataFrame(columns=['fixture_id', 'team_id', 'xg'])

    fids = fixtures['fixture_id'].tolist()
    placeholders_f = ",".join(["%s"] * len(fids))
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT fixture_id, team_id, value
            FROM fixture_team_stats
            WHERE fixture_id IN ({placeholders_f})
              AND stats_type_id = %s
            """,
            tuple(fids) + (XG_STAT_TYPE_ID,),
        )
        xg_rows = await cur.fetchall()
    xg = pd.DataFrame(xg_rows, columns=['fixture_id', 'team_id', 'xg'])
    if not xg.empty:
        xg['xg'] = xg['xg'].astype(float)
    return fixtures, xg


class OppLookup:
    """Time-aware opponent strength lookup against team_ratings.

    Handles BOTH FIFA (inverse='Yes') and Statz (inverse='No') rows. Returns
    the most recent row strictly before game_date. Caller branches the
    adjustment formula on the inverse flag.
    """

    def __init__(self, opp_df: pd.DataFrame):
        self._by_team = {
            tid: g.sort_values('date').reset_index(drop=True)
            for tid, g in opp_df.groupby('team_id')
        }

    def get(self, team_id: int, game_date) -> Optional[dict]:
        team_df = self._by_team.get(int(team_id))
        if team_df is None:
            return None
        gd = pd.to_datetime(game_date)
        valid = team_df[team_df['date'] < gd]
        if valid.empty:
            return None
        row = valid.iloc[-1]
        return {
            'attack': float(row['attack']),
            'defense': float(row['defense']),
            'overall': float(row['overall']),
            'inverse': row['inverse'],
        }


def _compute_one_team(
    team_id: int,
    all_fixtures: pd.DataFrame,
    xg: pd.DataFrame,
    opp_lookup: OppLookup,
    target_date: date,
) -> Optional[dict]:
    """Compute weighted-avg atk_xg + def_xg for one team. None if no fixtures.

    Returns raw pre-MV-nudge, pre-rescale values. Caller applies MV nudge
    and mean=100 rescale across the team set.
    """
    mask = (all_fixtures['home_team_id'] == team_id) | (all_fixtures['away_team_id'] == team_id)
    fx = all_fixtures[mask].copy()
    if fx.empty:
        return None

    fx['is_home'] = fx['home_team_id'] == team_id
    fx['g']  = fx['home_score'].where(fx['is_home'], fx['away_score'])
    fx['ga'] = fx['away_score'].where(fx['is_home'], fx['home_score'])
    fx['opponent_id'] = fx['away_team_id'].where(fx['is_home'], fx['home_team_id'])
    fx['g']  = pd.to_numeric(fx['g'],  errors='coerce')
    fx['ga'] = pd.to_numeric(fx['ga'], errors='coerce')
    fx = fx.dropna(subset=['g', 'ga'])
    if fx.empty:
        return None

    # xG join: (fixture_id, this team) → xg_for; (fixture_id, opp) → xg_against
    xg_for = xg[xg['team_id'] == team_id][['fixture_id', 'xg']].rename(columns={'xg': 'xg_for'})
    fx = fx.merge(xg_for, on='fixture_id', how='left')
    xg_against = xg.rename(columns={'team_id': 'opponent_id', 'xg': 'xg_against'})
    fx = fx.merge(xg_against, on=['fixture_id', 'opponent_id'], how='left')
    fx['xg_for'] = pd.to_numeric(fx['xg_for'], errors='coerce')
    fx['xg_against'] = pd.to_numeric(fx['xg_against'], errors='coerce')
    # Drop fixtures without xG — rate calculation requires real xG.
    fx = fx.dropna(subset=['xg_for', 'xg_against'])
    if fx.empty:
        return None

    # Blend: 0.3 * raw goals + 0.7 * xG
    fx['blend_g']  = 0.3 * fx['g']  + 0.7 * fx['xg_for']
    fx['blend_ga'] = 0.3 * fx['ga'] + 0.7 * fx['xg_against']

    # Time-aware opp adjustment, branched on opp row's inverse flag.
    def _apply_opp_adj(row):
        opp = opp_lookup.get(int(row['opponent_id']), row['kickoff_datetime'])
        if opp is None:
            return row['blend_g'], row['blend_ga']
        if opp['inverse'] == 'Yes':
            # FIFA path: multiply by opp.overall (attack=defense=overall in FIFA rows).
            v = opp['overall']
            if v == 0:
                return row['blend_g'], row['blend_ga']
            adj_g  = row['blend_g']  * (v / 100.0)
            adj_ga = row['blend_ga'] / (v / 100.0)
        else:
            # Statz path: divide by opp.defense (for adj_g) and opp.attack (for adj_ga),
            # both capped to [40, 250] to mirror FIFA-path multiplier range.
            opp_def = min(max(opp['defense'], STATZ_OPP_LO), STATZ_OPP_HI)
            opp_atk = min(max(opp['attack'],  STATZ_OPP_LO), STATZ_OPP_HI)
            adj_g  = row['blend_g']  / (opp_def / 100.0)
            adj_ga = row['blend_ga'] / (opp_atk / 100.0)
        return adj_g, adj_ga

    adj = fx.apply(_apply_opp_adj, axis=1, result_type='expand')
    fx['adj_g_raw']  = adj[0]
    fx['adj_ga_raw'] = adj[1]
    fx['adj_g']  = fx['adj_g_raw'].apply(soft_cap)
    fx['adj_ga'] = fx['adj_ga_raw'].apply(soft_cap)

    # Importance × weekly decay → game weight (decay starts after 3-week grace)
    fx['importance'] = fx['competition_id'].map(COMP_IMPORTANCE).fillna(1.0)
    target_dt = pd.to_datetime(target_date)
    fx['kickoff_datetime'] = pd.to_datetime(fx['kickoff_datetime'])
    fx['weeks_since'] = ((target_dt - fx['kickoff_datetime']).dt.days // 7).clip(lower=0)
    fx['decay'] = DECAY_WEIGHT ** (fx['weeks_since'] - 3).clip(lower=0)
    fx['game_weight'] = fx['importance'] * fx['decay']

    total_weight = fx['game_weight'].sum()
    if total_weight == 0:
        return None

    atk_xg = (fx['adj_g']  * fx['game_weight']).sum() / total_weight
    def_xg = (fx['adj_ga'] * fx['game_weight']).sum() / total_weight
    n_competitive = int((fx['competition_id'] != FRIENDLY_COMP_ID).sum())

    return {
        'team_id': int(team_id),
        'atk_xg': float(atk_xg),
        'def_xg': float(def_xg),
        'n_games': int(len(fx)),
        'n_competitive': n_competitive,
    }


async def compute_international_ratings(
    target_date: date,
    team_ids: Optional[List[int]] = None,
    commit: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute international team ratings as of target_date.

    Returns (statz_df, fifa_carry_df):
      statz_df columns:
        team_id, team_name, attack, defense, overall, attack_xg, defense_xg,
        overall_xg, n_games, inverse='No', date
        (attack/defense rescaled to mean=100 across qualifying teams;
         *_xg = raw pre-rescale weighted xG values after MV nudge)
      carry_df columns:
        team_id, team_name, attack, defense, overall, inverse='Yes', date
        (low-sample teams; attack/defense copied from most recent FIFA
         snapshot ≤ target_date)

    Set commit=True to write to team_ratings (DELETE existing date-rows
    for comp 732 then INSERT). Default commit=False returns DataFrames
    for inspection without writing.
    """
    conn = await get_source_connection()
    try:
        opp_df = await _load_opp_rows(conn)
        if opp_df.empty:
            raise ValueError("No team_ratings rows found for comp 732 — backfill FIFA snapshots first.")
        n_fifa = int((opp_df['inverse'] == 'Yes').sum())
        n_statz = int((opp_df['inverse'] == 'No').sum())
        logger.info(
            f"Loaded {len(opp_df)} opp rows ({n_fifa} FIFA + {n_statz} Statz) across "
            f"{opp_df['date'].nunique()} dates "
            f"({opp_df['date'].min().date()} → {opp_df['date'].max().date()})"
        )

        fixtures, xg = await _load_int_fixtures(conn, target_date)
        logger.info(
            f"Loaded {len(fixtures)} fixtures + {len(xg)} xG rows up to {target_date}"
        )

        # Default team set = every nation that has any FIFA baseline
        fifa_team_ids = sorted(opp_df[opp_df['inverse'] == 'Yes']['team_id'].unique().tolist())
        if team_ids is None:
            team_ids = fifa_team_ids

        # Name map for display
        async with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(team_ids))
            await cur.execute(f"SELECT id, name FROM teams WHERE id IN ({ph})", tuple(team_ids))
            name_map = dict(await cur.fetchall())

        # Most recent FIFA snapshot ≤ target_date per team (for carry-forward)
        target_dt = pd.to_datetime(target_date)
        fifa_only = opp_df[(opp_df['inverse'] == 'Yes') & (opp_df['date'] <= target_dt)]
        latest_fifa_per_team = (
            fifa_only
            .sort_values('date')
            .groupby('team_id')
            .tail(1)
            .set_index('team_id')
        )
    finally:
        release_source_connection(conn)

    opp_lookup = OppLookup(opp_df)
    mv_df = load_mv_values()

    raw_rows = []     # eligible teams, pre-MV, pre-rescale
    carry_rows = []   # FIFA carry-forwards
    for tid in team_ids:
        r = _compute_one_team(tid, fixtures, xg, opp_lookup, target_date)
        n = (r or {}).get('n_games', 0)
        n_comp = (r or {}).get('n_competitive', 0)
        name = name_map.get(tid, f'Team {tid}')

        is_eligible = (
            r is not None
            and n >= MIN_GAMES_FOR_STATZ
            and n_comp >= MIN_COMPETITIVE_GAMES
        )
        if is_eligible:
            raw_rows.append({
                'team_id': tid,
                'team_name': name,
                'atk_xg': r['atk_xg'],
                'def_xg': r['def_xg'],
                'n_games': n,
            })
        else:
            if tid not in latest_fifa_per_team.index:
                continue  # no FIFA prior — skip
            prior = latest_fifa_per_team.loc[tid]
            carry_rows.append({
                'team_id': tid,
                'team_name': name,
                'attack': float(prior['attack']),
                'defense': float(prior['defense']),
                'overall': float(prior['overall']),
                'inverse': 'Yes',
                'date': target_date,
                'fifa_source_date': prior['date'].date(),
            })

    statz_df = pd.DataFrame(raw_rows)

    # MV v8 nudge (if cache available) — pulls atk_xg/def_xg toward MV target.
    if not statz_df.empty and mv_df is not None:
        statz_df = statz_df.merge(mv_df, left_on='team_name', right_on='statz_name', how='left')
        n_unmapped = int(statz_df['mv_idx'].isna().sum())
        statz_df['mv_idx_filled'] = statz_df['mv_idx'].fillna(1.0)
        mean_mv = statz_df['mv_idx_filled'].mean()
        statz_df['mv_rev'] = mean_mv / statz_df['mv_idx_filled']
        statz_df['mv_rev'] = statz_df['mv_rev'] / statz_df['mv_rev'].mean()

        atk_mean = statz_df['atk_xg'].mean()
        def_mean = statz_df['def_xg'].mean()
        statz_df['atk_xg'] = statz_df['atk_xg'] * (
            1 + ((statz_df['mv_idx_filled'] - statz_df['atk_xg']/atk_mean) * INTL_MV_BETA) / statz_df['atk_xg']
        )
        statz_df['def_xg'] = statz_df['def_xg'] * (
            1 + ((statz_df['mv_rev'] - statz_df['def_xg']/def_mean) * INTL_MV_BETA) / statz_df['def_xg']
        )
        logger.info(f"MV v8 nudge applied: {len(statz_df) - n_unmapped}/{len(statz_df)} teams mapped, β={INTL_MV_BETA}")

    # Rescale to mean=100 separately for Attack and Defense.
    if not statz_df.empty:
        raw_ma = statz_df['atk_xg'].mean()
        raw_md = statz_df['def_xg'].mean()
        statz_df['attack']     = (statz_df['atk_xg'] / raw_ma * 100).round(2)
        statz_df['defense']    = (statz_df['def_xg'] / raw_md * 100).round(2)
        statz_df['overall']    = (statz_df['attack'] - statz_df['defense']).round(2)
        statz_df['attack_xg']  = statz_df['atk_xg'].round(4)
        statz_df['defense_xg'] = statz_df['def_xg'].round(4)
        statz_df['overall_xg'] = (statz_df['atk_xg'] - statz_df['def_xg']).round(4)
        statz_df['inverse']    = 'No'
        statz_df['date']       = target_date
        logger.info(f"Rescaled to mean=100: raw_ma={raw_ma:.3f}, raw_md={raw_md:.3f}, n={len(statz_df)}")

    carry_df = pd.DataFrame(carry_rows)

    logger.info(
        f"Snapshot {target_date}: {len(statz_df)} Statz Ratings, "
        f"{len(carry_df)} FIFA carry-forwards"
    )

    if commit:
        await _write_snapshot(statz_df, carry_df, target_date)
        logger.info(f"Committed {len(statz_df) + len(carry_df)} rows to team_ratings")
    else:
        logger.info("NOT committed — pass commit=True to write to team_ratings")

    return statz_df, carry_df


async def _write_snapshot(statz_df: pd.DataFrame, carry_df: pd.DataFrame, target_date: date) -> None:
    """Replace all comp-732 rows for target_date. statz inverse='No' (rescaled +
    raw xg), carry inverse='Yes' (FIFA values copied)."""
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM team_ratings WHERE competition_id=%s AND date=%s",
                (RATINGS_COMP_ID, target_date),
            )
            statz_rows = [(
                RATINGS_COMP_ID, int(r['team_id']), target_date,
                float(r['attack']), float(r['defense']), float(r['overall']),
                float(r['attack_xg']), float(r['defense_xg']), float(r['overall_xg']),
                'No',
            ) for _, r in statz_df.iterrows()]
            carry_rows = [(
                RATINGS_COMP_ID, int(r['team_id']), target_date,
                float(r['attack']), float(r['defense']), float(r['overall']),
                None, None, None,
                'Yes',
            ) for _, r in carry_df.iterrows()]
            all_rows = statz_rows + carry_rows
            for i in range(0, len(all_rows), 100):
                await cur.executemany(
                    """INSERT INTO team_ratings
                       (competition_id, team_id, date,
                        attack, defense, overall,
                        attack_xg, defense_xg, overall_xg,
                        inverse, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())""",
                    all_rows[i:i+100],
                )
        await conn.commit()
    finally:
        release_source_connection(conn)
