"""
International team ratings (nations).

Computes per-nation Attack + Defense from international fixtures, on a
weighted-goals-per-game scale (same shape as domestic get_ratings()):
  Attack  = Σ(adj_g  × game_weight) / Σ(game_weight)   # e.g. 2.4
  Defense = Σ(adj_ga × game_weight) / Σ(game_weight)   # e.g. 0.7

Key differences from the domestic get_ratings():
  - Pool of 17 international competitions (WC, Euros, Copa, AFCON, Asian Cup,
    Nations League, friendlies, all six WC qualifying confederations, three
    continental tournament qualifiers).
  - Per-fixture competition importance multiplier (WC=1.6 ... Friendly=0.5).
  - Per-week recency decay 0.995 (slower than domestic 0.97 since nations
    play far fewer games per calendar week).
  - Opponent strength is ALWAYS sourced from FIFA baselines stored in
    team_ratings with inverse='Yes'. The lookup is time-aware: each
    fixture uses the FIFA snapshot closest before its kickoff date.
  - Goal dampening: per-team per-fixture goal counts (both raw G and xG)
    are soft-capped via piecewise-linear above 4 (decay 0.3) and hard-cap
    at 8. Blunts the rating distortion from 7-0 / 10-0 blowouts that are
    far more common in internationals than domestic football.
  - Below MIN_GAMES_FOR_STATZ (10) total int'l games in our DB, a team
    gets a "FIFA carry-forward" row (inverse='Yes', FIFA values copied
    from the most recent FIFA snapshot ≤ target_date) instead of a
    computed Statz rating. Keeps low-sample teams stable.

This function does NOT write to the DB. Caller passes commit=False (default)
and inspects the returned DataFrame.
"""
import logging
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

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

# Per-competition importance multiplier on each match's contribution.
COMP_IMPORTANCE = {
    732:  1.6,   # World Cup
    720:  1.5,   # WC Qual Europe
    711:  1.5,   # WC Qual CAF
    714:  1.5,   # WC Qual Asia
    717:  1.5,   # WC Qual Concacaf
    723:  1.5,   # WC Qual Oceania
    726:  1.5,   # WC Qual South America
    729:  1.5,   # WC Qual Intercontinental Playoffs
    1326: 1.4,   # European Championship
    1114: 1.4,   # Copa America
    1117: 1.4,   # Africa Cup of Nations
    1105: 1.4,   # AFC Asian Cup
    1325: 1.3,   # Euro Qualification
    1118: 1.3,   # Africa Cup of Nations Qualifications
    1106: 1.3,   # Asian Cup Qualification
    1538: 0.75,  # UEFA Nations League
    1082: 0.5,   # Friendly International
}

# Per-week recency decay. Half-life ~138 weeks (~2.7 years).
DECAY_WEIGHT = 0.995

# All international snapshots stored under World Cup comp_id.
RATINGS_COMP_ID = 732

# Sportmonks stats_type_id for Expected Goals (xG) on fixture_team_stats.
XG_STAT_TYPE_ID = 5304

# state_id=5 = Full Time (the only state we count for ratings).
STATE_FT = 5

# Goal dampening for blowouts.
# Below DAMPEN_THRESHOLD → unchanged.
# Between THRESHOLD and CAP → threshold + (g - threshold) * DECAY.
# Above the implied cap → hard-cap at DAMPEN_CAP.
# Applied to both raw G and xG per fixture per side.
DAMPEN_THRESHOLD = 4
DAMPEN_DECAY = 0.3
DAMPEN_CAP = 8

# Minimum total international fixtures (across our DB) required for a
# nation to receive a Statz Rating. Below this, the row is a FIFA
# carry-forward (inverse='Yes', most recent FIFA snapshot copied).
MIN_GAMES_FOR_STATZ = 10


def dampen_goals(g: float) -> float:
    """Soft-cap goal counts to mute blowout distortion."""
    if pd.isna(g):
        return g
    if g <= DAMPEN_THRESHOLD:
        return g
    damped = DAMPEN_THRESHOLD + (g - DAMPEN_THRESHOLD) * DAMPEN_DECAY
    return min(damped, DAMPEN_CAP)


async def _load_all_fifa_baselines(conn) -> pd.DataFrame:
    """Load every FIFA baseline snapshot ever stored (inverse='Yes').

    Returns columns: team_id, attack, defense, date.
    """
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT team_id, attack, defense, date
            FROM team_ratings
            WHERE competition_id = %s AND inverse = 'Yes'
            ORDER BY team_id, date
            """,
            (RATINGS_COMP_ID,),
        )
        rows = await cur.fetchall()
    df = pd.DataFrame(rows, columns=['team_id', 'attack', 'defense', 'date'])
    df['attack'] = df['attack'].astype(float)
    df['defense'] = df['defense'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    return df


async def _load_int_fixtures(conn, date_to: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load every int'l fixture in DB (FT only) up to date_to + their xG rows."""
    placeholders = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT f.id, f.competition_id, f.kickoff_datetime,
                   f.home_team_id, f.away_team_id,
                   f.home_team_goals AS home_score, f.away_team_goals AS away_score
            FROM fixtures f
            WHERE f.competition_id IN ({placeholders})
              AND f.kickoff_datetime <= %s
              AND f.state_id = %s
            """,
            tuple(INTERNATIONAL_COMP_IDS) + (date_to, STATE_FT),
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


class FifaLookup:
    """Time-aware opponent strength lookup against FIFA snapshots.

    Returns the FIFA Attack + Defense for a given (team_id, game_date),
    using the most recent FIFA snapshot strictly before game_date.

    FIFA Defense is on the 'inverse' scale (high = strong defense). Caller
    uses the inverse-path formula (multiply by Defense/100) for opponent
    goal adjustment.
    """

    def __init__(self, fifa_df: pd.DataFrame):
        # Pre-group by team_id, sorted ascending by date, for binary-search lookups.
        self._by_team = {
            tid: g.sort_values('date').reset_index(drop=True)
            for tid, g in fifa_df.groupby('team_id')
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
        return {'attack': float(row['attack']), 'defense': float(row['defense'])}


def _compute_one_team(
    team_id: int,
    all_fixtures: pd.DataFrame,
    xg: pd.DataFrame,
    fifa_lookup: FifaLookup,
    target_date: date,
) -> Optional[dict]:
    """Compute weighted-avg Attack + Defense for one team. None if no fixtures."""
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
    fx['xg_for']     = pd.to_numeric(fx['xg_for'],     errors='coerce').fillna(fx['g'])
    fx['xg_against'] = pd.to_numeric(fx['xg_against'], errors='coerce').fillna(fx['ga'])

    # Dampen both G and xG before the 0.3*G + 0.7*xG blend
    fx['g_d']          = fx['g'].apply(dampen_goals)
    fx['ga_d']         = fx['ga'].apply(dampen_goals)
    fx['xg_for_d']     = fx['xg_for'].apply(dampen_goals)
    fx['xg_against_d'] = fx['xg_against'].apply(dampen_goals)

    fx['adj_g']  = 0.3 * fx['g_d']  + 0.7 * fx['xg_for_d']
    fx['adj_ga'] = 0.3 * fx['ga_d'] + 0.7 * fx['xg_against_d']

    # Time-aware opponent adjustment using FIFA snapshots.
    # FIFA is inverse='Yes': adj_g *= opp.Defense/100, adj_ga /= opp.Attack/100.
    def _apply_opp_adj(row):
        opp = fifa_lookup.get(int(row['opponent_id']), row['kickoff_datetime'])
        if opp is None or opp['attack'] == 0 or opp['defense'] == 0:
            return row['adj_g'], row['adj_ga']
        adj_g  = row['adj_g']  * (opp['defense'] / 100)
        adj_ga = row['adj_ga'] / (opp['attack']  / 100)
        return adj_g, adj_ga

    adj = fx.apply(_apply_opp_adj, axis=1, result_type='expand')
    fx['adj_g']  = adj[0]
    fx['adj_ga'] = adj[1]

    # Importance + decay → game weight
    fx['importance'] = fx['competition_id'].map(COMP_IMPORTANCE).fillna(1.0)
    target_dt = pd.to_datetime(target_date)
    fx['kickoff_datetime'] = pd.to_datetime(fx['kickoff_datetime'])
    fx['weeks_since'] = ((target_dt - fx['kickoff_datetime']).dt.days // 7).clip(lower=0)
    fx['decay'] = DECAY_WEIGHT ** (fx['weeks_since'] - 3).clip(lower=0)
    fx['game_weight'] = fx['importance'] * fx['decay']

    total_weight = fx['game_weight'].sum()
    if total_weight == 0:
        return None

    attack  = (fx['adj_g']  * fx['game_weight']).sum() / total_weight
    defense = (fx['adj_ga'] * fx['game_weight']).sum() / total_weight

    return {
        'team_id': int(team_id),
        'attack': float(attack),
        'defense': float(defense),
        'n_games': int(len(fx)),
    }


async def compute_quarterly_snapshot(
    target_date: date,
    team_ids: Optional[List[int]] = None,
    commit: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one international ratings snapshot at target_date.

    Pulls every int'l fixture in our DB up to target_date. Recency decay
    naturally diminishes the influence of older games.

    Returns (statz_df, fifa_carry_df):
      statz_df has columns:
        team_id, team_name, attack, defense, n_games, inverse='No', date
      fifa_carry_df has the same columns + inverse='Yes' for low-sample
        teams (n_games < MIN_GAMES_FOR_STATZ or no fixtures at all). attack/
        defense are copied from the most recent FIFA snapshot ≤ target_date.

    Set commit=True to actually write rows to team_ratings (otherwise the
    function returns the data for review without DB writes).
    """
    conn = await get_source_connection()
    try:
        fifa_df = await _load_all_fifa_baselines(conn)
        if fifa_df.empty:
            raise ValueError("No FIFA baselines found in team_ratings")
        logger.info(
            f"Loaded {len(fifa_df)} FIFA baseline rows across "
            f"{fifa_df['date'].nunique()} dates "
            f"({fifa_df['date'].min().date()} → {fifa_df['date'].max().date()})"
        )

        fixtures, xg = await _load_int_fixtures(conn, target_date)
        logger.info(
            f"Loaded {len(fixtures)} fixtures + {len(xg)} xG rows up to {target_date}"
        )

        # Default team set = every nation that has any FIFA baseline
        if team_ids is None:
            team_ids = sorted(fifa_df['team_id'].unique().tolist())

        # Name map for display
        async with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(team_ids))
            await cur.execute(f"SELECT id, name FROM teams WHERE id IN ({ph})", tuple(team_ids))
            name_map = dict(await cur.fetchall())

        # Most recent FIFA snapshot ≤ target_date per team (for carry-forward)
        target_dt = pd.to_datetime(target_date)
        latest_fifa_per_team = (
            fifa_df[fifa_df['date'] <= target_dt]
            .sort_values('date')
            .groupby('team_id')
            .tail(1)
            .set_index('team_id')
        )
    finally:
        release_source_connection(conn)

    fifa_lookup = FifaLookup(fifa_df)

    statz_rows = []
    carry_rows = []
    for tid in team_ids:
        r = _compute_one_team(tid, fixtures, xg, fifa_lookup, target_date)
        n = (r or {}).get('n_games', 0)
        name = name_map.get(tid, f'Team {tid}')

        if r is not None and n >= MIN_GAMES_FOR_STATZ:
            statz_rows.append({
                'team_id': tid,
                'team_name': name,
                'attack': round(r['attack'], 3),
                'defense': round(r['defense'], 3),
                'n_games': n,
                'inverse': 'No',
                'date': target_date,
            })
        else:
            # FIFA carry-forward
            if tid not in latest_fifa_per_team.index:
                continue  # no FIFA prior anywhere — skip
            prior = latest_fifa_per_team.loc[tid]
            carry_rows.append({
                'team_id': tid,
                'team_name': name,
                'attack': float(prior['attack']),
                'defense': float(prior['defense']),
                'n_games': n,
                'inverse': 'Yes',
                'date': target_date,
                'fifa_source_date': prior['date'].date(),
            })

    statz_df = pd.DataFrame(statz_rows)
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
    """Upsert all rows into team_ratings. statz rows inverse='No', carry inverse='Yes'."""
    from app.repository.db_utils import execute_chunked
    rows = []
    for _, r in statz_df.iterrows():
        rows.append((
            RATINGS_COMP_ID, int(r['team_id']), target_date,
            float(r['attack']), float(r['defense']),
            float(r['attack']) - float(r['defense']),
            'No',
        ))
    for _, r in carry_df.iterrows():
        rows.append((
            RATINGS_COMP_ID, int(r['team_id']), target_date,
            float(r['attack']), float(r['defense']),
            float(r['attack']) - float(r['defense']),
            'Yes',
        ))
    sql = """
        INSERT INTO team_ratings
          (competition_id, team_id, date, attack, defense, overall, inverse, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON DUPLICATE KEY UPDATE
          attack = VALUES(attack),
          defense = VALUES(defense),
          overall = VALUES(overall),
          inverse = VALUES(inverse),
          updated_at = NOW()
    """
    await execute_chunked(sql, rows, label='[team_ratings int_snapshot]')
