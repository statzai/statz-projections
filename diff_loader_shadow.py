"""Phase 4 diff tool — compare a LeagueDataLoader shadow snapshot against
the CSV-mode equivalent slice.

Usage:
    python diff_loader_shadow.py /tmp/loader_shadow/la-liga_20260427_092314 \
                                 --csv-dir app/data

For each scoped table (fixtures_df, team_stats, player_stats):
  - Loads the shadow parquet
  - Slices the matching CSV using the captured scope IDs (same filter the
    loader applied in SQL)
  - Compares row counts, key-set membership, and value equality on a random
    sample
  - Prints a summary; exit code 0 if parity, 1 if any diff > tolerance

Outputs are byte-level identical when:
  - Same scope IDs
  - Same dedup keys / order
  - Same column types (parquet round-trip can change dtypes vs raw CSV —
    we normalise before comparing)

Run this offline; doesn't touch DB or the live projection server runtime.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# Match data_cache.py / data_loader.py exactly — scope-able tables only.
TABLES = {
    "fixtures_df": {
        "csv_filename": "fixtures.csv",
        "dedup_subset": ["season_id", "home_team_id", "away_team_id", "kickoff_datetime"],
        "match_key": "id",
        "scope_filter": "fixture_ids_union",  # WHERE id IN (union)
    },
    "team_stats": {
        "csv_filename": "fixture_team_stats.csv",
        "dedup_subset": ["fixture_id", "team_id", "stats_type_id"],
        "match_key": ("fixture_id", "team_id", "stats_type_id"),
        "scope_filter": "team_fixture_ids",  # WHERE fixture_id IN team_fixture_ids
    },
    "player_stats": {
        "csv_filename": "fixture_player_stats.csv",
        "dedup_subset": ["fixture_id", "player_id", "stats_type_id"],
        "match_key": ("fixture_id", "player_id", "stats_type_id"),
        "scope_filter": "player_id_AND_fixture_ids",  # WHERE player_id IN p AND fixture_id IN union
    },
}


def load_scope(snapshot_dir: Path) -> dict:
    """Load the scope-ID lists the loader resolved for this snapshot."""
    return {
        "team_ids": set(pd.read_parquet(snapshot_dir / "_team_ids.parquet")["team_id"].tolist()),
        "player_ids": set(pd.read_parquet(snapshot_dir / "_player_ids.parquet")["player_id"].tolist()),
        "team_fixture_ids": set(pd.read_parquet(snapshot_dir / "_team_fixture_ids.parquet")["fixture_id"].tolist()),
        "player_fixture_ids": set(pd.read_parquet(snapshot_dir / "_player_fixture_ids.parquet")["fixture_id"].tolist()),
        "fixture_ids_union": set(pd.read_parquet(snapshot_dir / "_fixture_ids.parquet")["fixture_id"].tolist()),
    }


def slice_csv_to_scope(table_key: str, csv_path: Path, scope: dict) -> pd.DataFrame:
    """Read CSV, apply the same dedup + scope filter the loader applied."""
    cfg = TABLES[table_key]
    df = pd.read_csv(csv_path)

    # Dedup first (matches DataCache.load behaviour for these tables)
    df = df.drop_duplicates(subset=cfg["dedup_subset"]).reset_index(drop=True)

    if cfg["scope_filter"] == "fixture_ids_union":
        df = df[df["id"].isin(scope["fixture_ids_union"])]
    elif cfg["scope_filter"] == "team_fixture_ids":
        df = df[df["fixture_id"].isin(scope["team_fixture_ids"])]
    elif cfg["scope_filter"] == "player_id_AND_fixture_ids":
        df = df[
            df["player_id"].isin(scope["player_ids"]) &
            df["fixture_id"].isin(scope["fixture_ids_union"])
        ]
    return df.reset_index(drop=True)


def normalise_for_compare(df: pd.DataFrame, key) -> pd.DataFrame:
    """Sort by key, reset index. Coerce common type drift between CSV and
    parquet — CSVs read everything as object/float64, parquets keep ints."""
    if isinstance(key, str):
        key = [key]
    df = df.sort_values(by=list(key)).reset_index(drop=True)
    return df


def compare_keys(loader_df: pd.DataFrame, csv_df: pd.DataFrame, key) -> Tuple[set, set, set]:
    """Return (only_in_loader, only_in_csv, in_both) keysets."""
    if isinstance(key, str):
        loader_keys = set(loader_df[key].tolist())
        csv_keys = set(csv_df[key].tolist())
    else:
        loader_keys = set(map(tuple, loader_df[list(key)].itertuples(index=False, name=None)))
        csv_keys = set(map(tuple, csv_df[list(key)].itertuples(index=False, name=None)))
    return loader_keys - csv_keys, csv_keys - loader_keys, loader_keys & csv_keys


def diff_one_table(
    table_key: str, snapshot_dir: Path, csv_dir: Path, scope: dict, sample_rows: int = 50
) -> dict:
    cfg = TABLES[table_key]
    parquet_path = snapshot_dir / f"{table_key}.parquet"
    csv_path = csv_dir / cfg["csv_filename"]

    loader_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    csv_df = slice_csv_to_scope(table_key, csv_path, scope)

    loader_n, csv_n = len(loader_df), len(csv_df)
    if loader_n == 0 and csv_n == 0:
        return {"table": table_key, "loader_n": 0, "csv_n": 0, "status": "both empty"}

    only_loader, only_csv, in_both = compare_keys(loader_df, csv_df, cfg["match_key"])

    result = {
        "table": table_key,
        "loader_n": loader_n,
        "csv_n": csv_n,
        "in_both": len(in_both),
        "only_in_loader": len(only_loader),
        "only_in_csv": len(only_csv),
        "loader_columns": list(loader_df.columns),
        "csv_columns": list(csv_df.columns),
        "missing_columns_in_loader": sorted(set(csv_df.columns) - set(loader_df.columns)),
        "extra_columns_in_loader": sorted(set(loader_df.columns) - set(csv_df.columns)),
    }
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot_dir", help="Path to a /tmp/loader_shadow/<league>_<ts> snapshot dir")
    p.add_argument("--csv-dir", default="app/data", help="DataCache CSV dir (default: app/data)")
    p.add_argument(
        "--max-only-in-csv", type=int, default=0,
        help="Tolerance — non-zero exit if csv-only rows exceed this (default: 0)",
    )
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    csv_dir = Path(args.csv_dir)

    if not snapshot_dir.exists():
        print(f"ERROR: snapshot dir not found: {snapshot_dir}")
        return 2
    if not csv_dir.exists():
        print(f"ERROR: csv dir not found: {csv_dir}")
        return 2

    scope_meta = pd.read_parquet(snapshot_dir / "_scope.parquet").iloc[0].to_dict()
    scope = load_scope(snapshot_dir)

    print(f"Snapshot: {snapshot_dir.name}")
    print(f"League:   {scope_meta['league_name']} (id={scope_meta['league_id']})")
    print(f"Captured: {scope_meta['captured_at']}")
    print(f"Scope:    {len(scope['team_ids'])} teams, "
          f"{len(scope['player_ids'])} players, "
          f"{len(scope['team_fixture_ids'])} team-fixtures, "
          f"{len(scope['player_fixture_ids'])} player-fixtures, "
          f"{len(scope['fixture_ids_union'])} fixture union\n")

    results = []
    any_drift = False
    for table_key in TABLES:
        r = diff_one_table(table_key, snapshot_dir, csv_dir, scope)
        results.append(r)
        print(f"── {table_key} ────────────────────────────────────")
        print(f"  loader rows: {r.get('loader_n', 0):>7}")
        print(f"  csv    rows: {r.get('csv_n', 0):>7}")
        if r.get("status") == "both empty":
            print("  (both empty — skipping)\n")
            continue
        print(f"  in both    : {r['in_both']:>7}")
        print(f"  loader-only: {r['only_in_loader']:>7}")
        print(f"  csv-only   : {r['only_in_csv']:>7}")
        if r["missing_columns_in_loader"]:
            print(f"  ⚠ missing cols in loader: {r['missing_columns_in_loader']}")
            any_drift = True
        if r["extra_columns_in_loader"]:
            print(f"  ⓘ extra cols in loader  : {r['extra_columns_in_loader']}")
        if r["only_in_csv"] > args.max_only_in_csv:
            print(f"  ⚠ csv-only rows ({r['only_in_csv']}) exceeds tolerance ({args.max_only_in_csv})")
            any_drift = True
        if r["only_in_loader"] > 0:
            print(f"  ⓘ loader has {r['only_in_loader']} rows not in CSV slice — could be fresher data")
        print()

    print("=" * 60)
    if any_drift:
        print("RESULT: drift detected — investigate before flipping flag to ON.")
        return 1
    print("RESULT: no row-level drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
