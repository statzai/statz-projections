"""Smoke-test for LeagueDataLoader.

Run for one league, print scope counts + sample rows from each scoped
table. Lets you verify the loader actually executes and produces sane
shapes before flipping the shadow flag on a real projection.

Usage:
    docker compose exec app python test_loader.py "La Liga"
    docker compose exec app python test_loader.py "Premier League"
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

from app.data_loader import LeagueDataLoader
from app.source_database import (
    source_init_db_pool, close_source_db_pool, get_source_connection,
    release_source_connection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_loader")


async def resolve_league_id(league_name: str) -> int:
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM competitions WHERE name = %s", (league_name,)
            )
            row = await cur.fetchone()
            if row is None:
                raise ValueError(f"League not found: {league_name}")
            return int(row[0])
    finally:
        release_source_connection(conn)


async def main(league_name: str) -> int:
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    await source_init_db_pool()
    try:
        league_id = await resolve_league_id(league_name)
        print(f"\n══ {league_name} (id={league_id}) ══\n")

        league_weightings_path = Path(__file__).parent / "app" / "data" / "League Weightings.xlsx"
        loader = LeagueDataLoader(
            league_id,
            league_weightings_xlsx_path=str(league_weightings_path),
        )

        import time
        t0 = time.time()
        await loader.load()
        elapsed = time.time() - t0

        print(f"\nLoaded in {elapsed:.1f}s\n")

        print("── Scope ──────────────────────────────────")
        print(f"  team_ids           : {len(loader.team_ids):>6}")
        print(f"  player_ids         : {len(loader.player_ids):>6}")
        print(f"  team_fixture_ids   : {len(loader.team_fixture_ids):>6}")
        print(f"  player_fixture_ids : {len(loader.player_fixture_ids):>6}")
        print(f"  fixture_ids (∪)    : {len(loader.fixture_ids):>6}")
        added = len(loader.player_fixture_ids) - len(set(loader.team_fixture_ids) & set(loader.player_fixture_ids))
        print(f"  player-only added  : {added:>6}  (cross-club + intl)")

        for attr in ("fixtures_df", "team_stats", "player_stats"):
            df = getattr(loader, attr)
            print(f"\n── {attr} ──────────────────────────────")
            if df is None or df.empty:
                print("  EMPTY")
                continue
            print(f"  rows    : {len(df):>7}")
            print(f"  cols    : {len(df.columns)}")
            print(f"  columns : {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")
            print(f"  sample (first 3 rows):")
            print(df.head(3).to_string(index=False))

        for ref in ("comps", "seasons", "comp_teams", "teams", "stats_types",
                    "transfermarkt_team_mappings", "promoted_team_ratings",
                    "projection_config", "team_ratings", "league_weightings"):
            df = getattr(loader, ref)
            n = "—" if df is None else (0 if df.empty else len(df))
            print(f"  {ref:<35} {n}")

        return 0
    finally:
        await close_source_db_pool()


if __name__ == "__main__":
    league = sys.argv[1] if len(sys.argv) > 1 else "La Liga"
    sys.exit(asyncio.run(main(league)))
