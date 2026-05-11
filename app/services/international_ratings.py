"""
International team ratings (nations).

Adapted from get_ratings() in statz_functions.py for the international setting:
  - Pools fixtures from ALL international competitions for each nation
  - Per-fixture competition importance multiplier (WC most, Friendlies least)
  - Per-week recency decay 0.995 (slower than domestic 0.97)
  - Opponent-strength lookup by team_id (avoids name ambiguity:
    "Korea" / "South Korea" / "Republic of Korea" etc)
  - Honours team_ratings.inverse: 'Yes' rows (FIFA-points baseline) use
    multiply semantics; 'No' rows (our computed snapshots, goals/game)
    use divide. Identical to the existing domestic path.

Snapshots written to team_ratings with competition_id=732 (World Cup as
the canonical home for international ratings) and Date=quarter-end.

Test-first design: compute_quarterly_snapshot() returns a DataFrame for
review. Caller passes commit=True to actually write rows.
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
    1325,  # Euro Qualification
    1326,  # European Championship
    1114,  # Copa America
    1117,  # Africa Cup of Nations
    1105,  # AFC Asian Cup
    1106,  # Asian Cup Qualification
    1538,  # UEFA Nations League
    1082,  # Friendly International
]

# Per-competition importance multiplier on each match's contribution to ratings.
COMP_IMPORTANCE = {
    732:  1.6,   # World Cup
    720:  1.5,   # WC Qual Europe
    711:  1.5,   # WC Qual CAF
    1326: 1.4,   # European Championship
    1114: 1.4,   # Copa America
    1117: 1.4,   # Africa Cup of Nations
    1105: 1.4,   # AFC Asian Cup
    1325: 1.3,   # Euro Qualification
    1106: 1.3,   # Asian Cup Qualification
    1538: 0.75,  # UEFA Nations League
    1082: 0.5,   # Friendly International
}

# Per-week recency decay. Half-life ~138 weeks (~2.7 years) — slower than
# domestic 0.97 because nations play far fewer games per calendar week.
DECAY_WEIGHT = 0.995

# All international snapshots stored under World Cup comp_id.
RATINGS_COMP_ID = 732

# Sportmonks stats_type_id for Expected Goals (xG) on fixture_team_stats.
XG_STAT_TYPE_ID = 5304

# state_id=5 = Full Time (the only state we count for ratings).
STATE_FT = 5


async def _load_baseline(conn, prior_date: date) -> Tuple[pd.DataFrame, date]:
    """Load the most recent rating snapshot ≤ prior_date for opponent lookup.

    Returns (baseline_df, baseline_date). baseline_df has columns:
      team_id, attack, defense, inverse, date
    """
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT MAX(date) FROM team_ratings WHERE competition_id = %s AND date <= %s",
            (RATINGS_COMP_ID, prior_date),
        )
        row = await cur.fetchone()
        max_date = row[0] if row else None
        if max_date is None:
            raise ValueError(f"No team_ratings rows found for comp={RATINGS_COMP_ID} ≤ {prior_date}")
        await cur.execute(
            """
            SELECT team_id, attack, defense, inverse, date
            FROM team_ratings
            WHERE competition_id = %s AND date = %s
            """,
            (RATINGS_COMP_ID, max_date),
        )
        rows = await cur.fetchall()
    df = pd.DataFrame(rows, columns=['team_id', 'attack', 'defense', 'inverse', 'date'])
    df['attack'] = df['attack'].astype(float)
    df['defense'] = df['defense'].astype(float)
    return df, max_date


async def _load_int_fixtures(conn, date_from: date, date_to: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all int'l fixtures and their xG rows in [date_from, date_to]."""
    placeholders = ",".join(["%s"] * len(INTERNATIONAL_COMP_IDS))
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT f.id, f.competition_id, f.kickoff_datetime,
                   f.home_team_id, f.away_team_id, f.home_score, f.away_score
            FROM fixtures f
            WHERE f.competition_id IN ({placeholders})
              AND f.kickoff_datetime >= %s
              AND f.kickoff_datetime <= %s
              AND f.state_id = %s
            """,
            tuple(INTERNATIONAL_COMP_IDS) + (date_from, date_to, STATE_FT),
        )
        rows = await cur.fetchall()
    fixtures = pd.DataFrame(rows, columns=[
        'fixture_id', 'competition_id', 'kickoff_datetime',
        'home_team_id', 'away_team_id', 'home_score', 'away_score',
    ])
    if fixtures.empty:
        return fixtures, pd.DataFrame(columns=['fixture_id', 'team_id', 'xg'])

    fids = fixtures['fixture_id'].tolist()
    # MySQL placeholder list — chunk if many fixtures (max ~30k packet)
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


def _adjust_for_opponent(row, baseline_lookup) -> Tuple[float, float]:
    """Apply opponent-strength adjustment.

    Matches the existing get_ratings() logic exactly:
      - Adj Goals Against /= opp.attack/100  (always; Attack semantics consistent)
      - If opp.inverse == 'Yes': Adj Goals *= opp.defense/100
      - else:                    Adj Goals /= opp.defense/100
    """
    opp = baseline_lookup.get(row['opponent_id'])
    if opp is None:
        return row['adj_g'], row['adj_ga']
    att = opp['attack']
    deff = opp['defense']
    inv = opp['inverse']
    if att == 0 or deff == 0:
        return row['adj_g'], row['adj_ga']
    adj_g = row['adj_g'] * (deff / 100) if inv == 'Yes' else row['adj_g'] / (deff / 100)
    adj_ga = row['adj_ga'] / (att / 100)
    return adj_g, adj_ga


def _compute_one_team(team_id: int, all_fixtures: pd.DataFrame, xg: pd.DataFrame,
                      baseline_lookup: dict, target_date: date) -> Optional[dict]:
    """Compute raw weighted Attack + Defense for one team."""
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

    # xG: join on (fixture_id, team_id=this team) and (fixture_id, team_id=opponent)
    xg_for = xg[xg['team_id'] == team_id][['fixture_id', 'xg']].rename(columns={'xg': 'xg_for'})
    fx = fx.merge(xg_for, on='fixture_id', how='left')
    xg_against = xg.rename(columns={'team_id': 'opponent_id', 'xg': 'xg_against'})
    fx = fx.merge(xg_against, on=['fixture_id', 'opponent_id'], how='left')
    fx['xg_for']     = fx['xg_for'].fillna(fx['g'])
    fx['xg_against'] = fx['xg_against'].fillna(fx['ga'])

    # Adjusted goals: 0.3*G + 0.7*xG (matches domestic get_ratings)
    fx['adj_g']  = 0.3 * fx['g']  + 0.7 * fx['xg_for']
    fx['adj_ga'] = 0.3 * fx['ga'] + 0.7 * fx['xg_against']

    # Opponent adjustment
    adj = fx.apply(lambda r: _adjust_for_opponent(r, baseline_lookup), axis=1)
    fx['adj_g']  = adj.apply(lambda t: t[0])
    fx['adj_ga'] = adj.apply(lambda t: t[1])

    # Competition importance multiplier
    fx['importance'] = fx['competition_id'].map(COMP_IMPORTANCE).fillna(1.0)

    # Recency decay: 0.995 ^ (weeks_since_kickoff - 3), cap at 1 for ≤4w
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
        'team_id': team_id,
        'attack_raw': attack,
        'defense_raw': defense,
        'n_games': len(fx),
    }


async def compute_quarterly_snapshot(
    target_date: date,
    prior_date: Optional[date] = None,
    team_ids: Optional[List[int]] = None,
    commit: bool = False,
) -> pd.DataFrame:
    """Compute one international ratings snapshot at target_date.

    Args:
        target_date: snapshot date (typically quarter-end)
        prior_date: pull most recent rating snapshot ≤ this date for opponent
                    lookup. Defaults to target_date - 7d.
        team_ids: which teams to compute. None = all teams with fixtures
                  in the window.
        commit: if True, write rows to team_ratings table. Default False
                so the caller can review.

    Returns DataFrame with columns:
        team_id, team_name, attack, defense, attack_raw, defense_raw, n_games
    """
    if prior_date is None:
        prior_date = pd.Timestamp(target_date) - pd.Timedelta(days=7)
        prior_date = prior_date.date()

    conn = await get_source_connection()
    try:
        baseline, baseline_date = await _load_baseline(conn, prior_date)
        logger.info(f"Loaded baseline {baseline_date} ({len(baseline)} teams)")

        # Load fixtures in [baseline_date, target_date]
        fixtures, xg = await _load_int_fixtures(conn, baseline_date, target_date)
        logger.info(
            f"Loaded {len(fixtures)} fixtures + {len(xg)} xG rows in "
            f"[{baseline_date} → {target_date}]"
        )

        if team_ids is None:
            team_ids = sorted(set(
                fixtures['home_team_id'].tolist() + fixtures['away_team_id'].tolist()
            ))

        async with conn.cursor() as cur:
            ph = ",".join(["%s"] * len(team_ids))
            await cur.execute(f"SELECT id, name FROM teams WHERE id IN ({ph})", tuple(team_ids))
            name_map = dict(await cur.fetchall())
    finally:
        release_source_connection(conn)

    baseline_lookup = baseline.set_index('team_id').to_dict('index')

    results = []
    for tid in team_ids:
        r = _compute_one_team(tid, fixtures, xg, baseline_lookup, target_date)
        if r is None:
            continue
        r['team_name'] = name_map.get(tid, f'Team {tid}')
        results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Normalize raw values to mean=100 (matches domestic projection_service.py:754)
    df['attack']  = df['attack_raw']  / df['attack_raw'].mean()  * 100
    df['defense'] = df['defense_raw'] / df['defense_raw'].mean() * 100
    df['attack']  = df['attack'].round(2)
    df['defense'] = df['defense'].round(2)
    df['attack_raw']  = df['attack_raw'].round(3)
    df['defense_raw'] = df['defense_raw'].round(3)
    df['date'] = target_date

    if commit:
        await _write_snapshot(df, target_date)
        logger.info(f"Committed {len(df)} rows to team_ratings ({target_date})")
    else:
        logger.info(f"Computed {len(df)} ratings (NOT committed — pass commit=True to write)")

    return df


async def _write_snapshot(df: pd.DataFrame, target_date: date) -> None:
    """Upsert snapshot rows into team_ratings."""
    from app.repository.db_utils import execute_chunked
    rows = [
        (RATINGS_COMP_ID, int(r['team_id']), target_date,
         float(r['attack']), float(r['defense']),
         float(r['attack']) - float(r['defense']),
         'No')
        for _, r in df.iterrows()
    ]
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
