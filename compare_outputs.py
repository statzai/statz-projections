"""Snapshot + diff tool for projection output tables.

Captures the contents of all six projection-output tables (scoped to one
league) so we can compare CSV-mode vs DB-loader-mode runs. Used during
the Direct DB Query Migration to verify on-mode produces equivalent
projections to off-mode.

Tables captured:
  - fixture_projections      (per-fixture: home_goals, win/draw/away %, etc.)
  - team_projections         (per-team-per-fixture: shot expectations, etc.)
  - player_projections       (per-player-per-fixture-per-stat)
  - player_prop_projections  (per-player-prop)
  - league_projections       (predicted final standings — Power Rankings as table)
  - team_ratings             (Power Rankings: Attack/Defense/Overall/Movement)

Usage:
    # capture current DB state under a label
    docker compose exec statz-projection python compare_outputs.py snapshot \
        --league "La Liga" --label db_loader_v1

    # compare two captured snapshots column-by-column
    docker compose exec statz-projection python compare_outputs.py diff \
        --label-a csv_v1 --label-b db_loader_v1

Snapshot scope: fixtures kicking off in the next 5 days for the league,
plus today's team_ratings + league_projections row set. Matches what the
projection run actually wrote.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_outputs")

CAPTURE_ROOT = Path("/tmp/output_capture")

# (table, scope_strategy, key_columns, value_columns_for_diff)
# scope_strategy values:
#   "fixture_ids"  — WHERE fixture_id IN (fixtures upcoming for league)
#   "competition_id"  — WHERE competition_id = league_id
#   "today_ratings"   — WHERE competition_id = X AND date = today
TABLES = [
    {
        "name": "fixture_projections",
        "scope": "fixture_ids",
        "key": ["fixture_id"],
        "value_cols": [
            "home_goals", "away_goals",
            "home_win_percent", "away_win_percent", "draw_percent",
            "home_clean_sheet_percent", "away_clean_sheet_percent",
            "over_15_goals_percent", "over_25_goals_percent",
            "both_teams_shore_percent",
        ],
    },
    {
        "name": "team_projections",
        "scope": "fixture_ids",
        "key": ["fixture_id", "team_id"],
        "value_cols": [
            "goals", "shots_total", "shots_on_target", "corners",
            "fouls", "yellowcards", "tackles", "passes",
            "total_crosses", "offsides",
        ],
    },
    {
        "name": "player_projections",
        "scope": "fixture_ids",
        "key": ["fixture_id", "player_id", "stats_type_id"],
        "value_cols": ["start", "stats_value"],
    },
    {
        "name": "player_prop_projections",
        "scope": "fixture_ids",
        "key": ["fixture_id", "player_id", "stats_type_id", "prop"],
        "value_cols": ["projection_percent"],
    },
    {
        "name": "league_projections",
        "scope": "competition_id",
        "key": ["competition_id", "team_id"],
        "value_cols": [
            "position", "points", "goals_for", "goals_against",
            "goal_difference", "win_percent",
            "top_2_percent", "top_4_percent", "top_6_percent", "top_7_percent",
            "relegation_percent", "max_points", "min_points",
        ],
    },
    {
        "name": "team_ratings",
        "scope": "today_ratings",
        "key": ["competition_id", "team_id", "date"],
        "value_cols": [
            "attack", "defense", "overall",
            "attack_xg", "defense_xg", "overall_xg",
            "movement",
        ],
    },
]


async def resolve_league_id(league_name: str) -> int:
    from app.source_database import get_source_connection, release_source_connection
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


async def get_upcoming_fixture_ids(league_id: int, days: int = 5) -> list:
    """Same fixture window as projection_service.projections() — TODAY 00:00
    through TODAY+N days. Critical that this matches the projection's window
    exactly so snapshot includes every fixture the run wrote (incl. ones
    that have since kicked off and are no longer 'upcoming-from-now')."""
    from app.source_database import get_source_connection, release_source_connection
    conn = await get_source_connection()
    today_midnight = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_midnight + timedelta(days=days)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id FROM fixtures
                WHERE competition_id = %s
                  AND kickoff_datetime >= %s
                  AND kickoff_datetime <= %s
                """,
                (league_id, today_midnight, cutoff),
            )
            return [int(r[0]) for r in await cur.fetchall()]
    finally:
        release_source_connection(conn)


async def fetch_table(table_cfg: dict, league_id: int, fixture_ids: list, today: str) -> pd.DataFrame:
    """Fetch table contents from the PROJECTION DB (not source DB).

    The projection-output tables live on the writable DB (statz Laravel
    schema), accessed via `database.py` (the local pool), not the
    `source_database.py` read-only pool.
    """
    from app.database import get_connection, pool as db_pool
    table = table_cfg["name"]
    scope = table_cfg["scope"]
    if scope == "fixture_ids":
        if not fixture_ids:
            return pd.DataFrame()
        ph = ",".join(["%s"] * len(fixture_ids))
        sql = f"SELECT * FROM {table} WHERE fixture_id IN ({ph})"
        params = tuple(fixture_ids)
    elif scope == "competition_id":
        sql = f"SELECT * FROM {table} WHERE competition_id = %s"
        params = (league_id,)
    elif scope == "today_ratings":
        sql = f"SELECT * FROM {table} WHERE competition_id = %s AND date = %s"
        params = (league_id, today)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    from app import database as _db
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        if _db.pool is not None:
            _db.pool.release(conn)


async def cmd_snapshot(league: str, label: str):
    from app.source_database import source_init_db_pool, close_source_db_pool
    from app.database import init_db_pool, close_db_pool
    await source_init_db_pool()
    await init_db_pool()
    try:
        league_id = await resolve_league_id(league)
        fixture_ids = await get_upcoming_fixture_ids(league_id)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = CAPTURE_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = pd.DataFrame([{
            "league_name": league,
            "league_id": league_id,
            "captured_at": datetime.utcnow().isoformat(),
            "n_fixture_ids": len(fixture_ids),
        }])
        meta.to_parquet(out_dir / "_meta.parquet", index=False)
        pd.DataFrame({"fixture_id": fixture_ids}).to_parquet(
            out_dir / "_fixture_ids.parquet", index=False
        )

        for cfg in TABLES:
            df = await fetch_table(cfg, league_id, fixture_ids, today)
            df.to_parquet(out_dir / f"{cfg['name']}.parquet", index=False)
            logger.info(f"  {cfg['name']:<28} {len(df):>6} rows")

        print(f"\nSnapshot written to {out_dir}")
    finally:
        await close_source_db_pool()
        await close_db_pool()


def cmd_diff(label_a: str, label_b: str, tolerance: float = 0.01):
    """Compare two snapshots. For each table, outer-join on key cols and
    flag value-column rows whose absolute difference exceeds tolerance."""
    dir_a = CAPTURE_ROOT / label_a
    dir_b = CAPTURE_ROOT / label_b
    if not dir_a.exists():
        print(f"ERROR: snapshot not found: {dir_a}")
        return 2
    if not dir_b.exists():
        print(f"ERROR: snapshot not found: {dir_b}")
        return 2

    print(f"\nComparing snapshots:\n  A = {label_a}\n  B = {label_b}\n")

    any_drift = False
    for cfg in TABLES:
        table = cfg["name"]
        path_a = dir_a / f"{table}.parquet"
        path_b = dir_b / f"{table}.parquet"
        df_a = pd.read_parquet(path_a) if path_a.exists() else pd.DataFrame()
        df_b = pd.read_parquet(path_b) if path_b.exists() else pd.DataFrame()

        print(f"── {table} ─────────────────────────────")
        print(f"  A rows: {len(df_a):>6}    B rows: {len(df_b):>6}")

        if df_a.empty and df_b.empty:
            print("  (both empty — skipping)\n")
            continue

        key = cfg["key"]
        # Coerce key cols to a comparable type (int) where possible
        for k in key:
            for d in (df_a, df_b):
                if k in d.columns:
                    try:
                        d[k] = pd.to_numeric(d[k], errors="ignore")
                    except Exception:
                        pass

        # Set membership
        keys_a = set(map(tuple, df_a[key].itertuples(index=False, name=None))) if not df_a.empty else set()
        keys_b = set(map(tuple, df_b[key].itertuples(index=False, name=None))) if not df_b.empty else set()
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        in_both = keys_a & keys_b
        if only_a or only_b:
            print(f"  only in A: {len(only_a)}    only in B: {len(only_b)}    in both: {len(in_both)}")
            any_drift = any_drift or bool(only_a or only_b)
        else:
            print(f"  key sets identical ({len(in_both)} rows)")

        # Value diffs on rows in both
        if in_both and cfg.get("value_cols"):
            merged = df_a.merge(df_b, on=key, suffixes=("_a", "_b"), how="inner")
            value_diffs = []
            for col in cfg["value_cols"]:
                a_col = f"{col}_a"
                b_col = f"{col}_b"
                if a_col not in merged.columns or b_col not in merged.columns:
                    continue
                # Coerce to numeric for comparison
                merged[a_col] = pd.to_numeric(merged[a_col], errors="coerce")
                merged[b_col] = pd.to_numeric(merged[b_col], errors="coerce")
                diff = (merged[a_col] - merged[b_col]).abs()
                n_diff = int((diff > tolerance).sum())
                if n_diff > 0:
                    max_diff = float(diff.max())
                    mean_diff = float(diff.mean())
                    value_diffs.append((col, n_diff, max_diff, mean_diff))

            if value_diffs:
                any_drift = True
                print(f"  ⚠ value diffs (tolerance ±{tolerance}):")
                for col, n, mx, mn in value_diffs:
                    print(f"      {col:<32} {n:>5} rows differ  max={mx:.4f}  mean={mn:.4f}")
            else:
                print(f"  ✓ all values within tolerance (±{tolerance})")
        print()

    print("=" * 60)
    if any_drift:
        print("RESULT: drift detected — review per-table summaries above.")
        return 1
    print("RESULT: snapshots match within tolerance.")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_snap = sub.add_parser("snapshot")
    sp_snap.add_argument("--league", required=True)
    sp_snap.add_argument("--label", required=True)

    sp_diff = sub.add_parser("diff")
    sp_diff.add_argument("--label-a", required=True)
    sp_diff.add_argument("--label-b", required=True)
    sp_diff.add_argument("--tolerance", type=float, default=0.01)

    args = p.parse_args()

    if args.cmd == "snapshot":
        return asyncio.run(cmd_snapshot(args.league, args.label))
    elif args.cmd == "diff":
        return cmd_diff(args.label_a, args.label_b, args.tolerance)
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
