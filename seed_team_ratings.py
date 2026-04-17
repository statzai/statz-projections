"""
One-off backfill script: read the existing Team Ratings.parquet file and
insert all rows into the team_ratings DB table.

Run once after the Laravel migration has created the empty team_ratings
table. Safe to re-run (INSERT IGNORE preserves existing rows via the
(competition_id, team_id, date) unique key).

Usage (from inside the projection container):
    docker compose exec statz-projection python3 seed_team_ratings.py

Or via SSH on the projection server:
    ssh ... 'docker compose -f .../docker-compose.yml exec -T statz-projection python3 seed_team_ratings.py'
"""
import asyncio
import logging
import pandas as pd
from app.database import init_db_pool, get_connection
from app.source_database import source_init_db_pool, get_source_connection, release_source_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed")


async def fetch_lookups():
    """Pull competitions + teams from source DB to resolve name → id."""
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM competitions")
            comps = await cur.fetchall()
            await cur.execute("SELECT id, name FROM teams")
            teams = await cur.fetchall()
        comps_df = pd.DataFrame(comps, columns=["id", "name"])
        teams_df = pd.DataFrame(teams, columns=["id", "name"])
        return comps_df, teams_df
    finally:
        release_source_connection(conn)


async def main():
    await init_db_pool()
    await source_init_db_pool()

    logger.info("Reading Team Ratings.parquet...")
    df = pd.read_parquet("/app/app/data/Team Ratings.parquet")
    logger.info(f"Loaded {len(df)} rows from parquet")

    logger.info("Fetching competitions + teams lookups...")
    comps_df, teams_df = await fetch_lookups()

    comp_lookup = dict(zip(comps_df["name"], comps_df["id"]))
    team_lookup = dict(zip(teams_df["name"], teams_df["id"]))

    # Resolve names. Drop rows where either lookup fails.
    df["competition_id"] = df["League"].map(comp_lookup)
    df["team_id"] = df["Team"].map(team_lookup)

    unresolved_league = df[df["competition_id"].isna()]["League"].unique()
    unresolved_team = df[df["team_id"].isna()]["Team"].unique()
    if len(unresolved_league):
        logger.warning(f"Unresolved league names (skipping rows): {list(unresolved_league)}")
    if len(unresolved_team):
        logger.warning(f"Unresolved team names ({len(unresolved_team)} distinct, skipping rows) — first 20: {list(unresolved_team)[:20]}")

    resolved = df.dropna(subset=["competition_id", "team_id"]).copy()
    logger.info(f"Resolved {len(resolved)} / {len(df)} rows ({len(df) - len(resolved)} dropped)")

    # Build values list
    def to_val(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        return v

    values = [
        (
            int(row["competition_id"]),
            int(row["team_id"]),
            row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
            to_val(row["Attack"]),
            to_val(row["Defense"]),
            to_val(row["Overall"]),
            to_val(row["Movement"]),
            to_val(row["Inverse"]),
        )
        for _, row in resolved.iterrows()
    ]

    logger.info(f"Inserting {len(values)} rows in chunks of 500...")

    conn = await get_connection()
    try:
        sql = """
        INSERT IGNORE INTO team_ratings (
            competition_id, team_id, date,
            attack, defense, overall, movement, inverse,
            created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        """
        total_inserted = 0
        chunk_size = 500
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i + chunk_size]
            async with conn.cursor() as cur:
                affected = await cur.executemany(sql, chunk)
            await conn.commit()
            total_inserted += affected or 0
            logger.info(f"  chunk {i // chunk_size + 1}/{(len(values) + chunk_size - 1) // chunk_size}: {affected} rows affected")
        logger.info(f"Done. Total rows affected: {total_inserted}")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
