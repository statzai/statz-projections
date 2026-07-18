import logging
import asyncio
import aiomysql
import pandas as pd
from app.database import get_connection
from app.repository.db_utils import execute_chunked

logger = logging.getLogger("transfermarkt_mv_repo")


async def insert_market_value_snapshots_async(df: pd.DataFrame, league_dashed: str) -> int:
    """Upsert today's Transfermarkt scrape into transfermarkt_market_value_snapshots.

    Called after every successful scrape so the cache stays current. Unique
    key is (league_dashed, team_name) — repeated scrapes for the same
    league overwrite the last snapshot for each team.
    """
    if df is None or len(df) == 0:
        return 0

    values = []
    for _, row in df.iterrows():
        team = row.get('Team')
        mv = row.get('Market Value')
        if team is None or mv is None:
            continue
        values.append((league_dashed, str(team).strip(), str(mv).strip()))

    if not values:
        return 0

    sql = """
    INSERT INTO transfermarkt_market_value_snapshots (
        league_dashed, team_name, market_value, scraped_at,
        created_at, updated_at
    ) VALUES (%s, %s, %s, NOW(), NOW(), NOW())
    AS new
    ON DUPLICATE KEY UPDATE
        market_value = new.market_value,
        scraped_at = NOW(),
        updated_at = NOW()
    """
    return await execute_chunked(sql, values, label=f"[transfermarkt_mv_snapshots:{league_dashed}]")


async def read_latest_market_values_async(league_dashed: str) -> pd.DataFrame:
    """Read the most recent snapshot for league_dashed.

    Returned df has the same shape as a successful get_market_value scrape:
    columns ['Team', 'Market Value']. Empty df if there's no cached data
    yet (first-time scrape failure on a league we've never seen).
    """
    conn = await asyncio.wait_for(get_connection(), timeout=10)
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT team_name AS Team, market_value AS `Market Value`, scraped_at "
                "FROM transfermarkt_market_value_snapshots "
                "WHERE league_dashed = %s "
                "ORDER BY scraped_at DESC",
                (league_dashed,),
            )
            rows = await cur.fetchall()
    finally:
        import app.database as _db
        if _db.pool:
            _db.pool.release(conn)

    if not rows:
        return pd.DataFrame(columns=['Team', 'Market Value'])

    df = pd.DataFrame(rows)
    most_recent_at = df['scraped_at'].iloc[0]
    # Only the newest batch: rows upsert per (league, team), so teams that
    # left the league (relegation/season turnover) linger with an old
    # scraped_at. Without this filter the fallback served current teams
    # PLUS last season's relegated ones, skewing the MV index and keeping
    # phantom "unmapped team" warnings alive.
    df = df[df['scraped_at'] == most_recent_at]
    logger.info(
        f"[transfermarkt_mv_snapshots:{league_dashed}] "
        f"Returning {len(df)} cached MVs from {most_recent_at} "
        f"(scrape failed, falling back to last-good)"
    )
    return df[['Team', 'Market Value']]
