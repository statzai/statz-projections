"""
One-off backfill script: read the existing *_model_dataset_with_history.parquet
and *_accuracy_dataset.parquet files on /app/app/data/ and insert all rows
into projection_model_dataset + projection_accuracy_dataset. Also registers
existing .sav model files in projection_models with is_active=1 so the new
DB-backed loader has something to load after cutover.

Run once after the Laravel migration has created the empty tables. Safe to
re-run (ON DUPLICATE KEY UPDATE preserves rows via (fixture_id, team_id)
natural key for model_dataset and (fixture_id) for accuracy_dataset).

Usage (from inside the projection container):
    docker compose exec statz-projection python3 seed_projection_datasets.py

Or from host via SSH:
    ssh ... 'docker exec statz-projection-statz-projection-1 python3 seed_projection_datasets.py'

CLI flags:
    --only {model,accuracy,models}   run only one section (default: all three)
    --dry-run                        parse and resolve but don't insert
"""
import argparse
import asyncio
import glob
import logging
import math
import os
import re
from datetime import datetime

import pandas as pd

from app.database import init_db_pool, get_connection
from app.source_database import source_init_db_pool, get_source_connection, release_source_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed")

DATA_DIR = "/app/app/data"
MODEL_BUILDS_DIR = "/app/app/model-builds"

# Parquet → DB column mapping for projection_model_dataset. All stat and
# history column names follow the simple rule: lowercase + space→underscore.
# Only identity columns need explicit mapping.
MODEL_DATASET_IDENTITY_MAP = {
    "id": "fixture_id",
    "comp_id": "competition_id",
    "Team": "team_name",
    "Opponent": "opponent_name",
    "Venue": "venue",
    "kickoff_datetime": "kickoff_datetime",
}

MODEL_DATASET_STATS = [
    "shots_total", "shots_on_target", "corners", "fouls", "yellowcards",
    "tackles", "passes", "successful_passes", "interceptions",
    "total_crosses", "offsides",
]

# Parquet → DB column mapping for projection_accuracy_dataset. Stats are
# generated programmatically below; only the irregular columns need explicit
# mapping.
ACCURACY_DATASET_IDENTITY_MAP = {
    "fixture_id": "fixture_id",
    "comp_id": "competition_id",
    "Home Team": "home_team_name",
    "Away Team": "away_team_name",
    "kickoff_datetime": "kickoff_datetime",
}
ACCURACY_DATASET_IRREGULAR_MAP = {
    "Over 1.5 Goals %": "over_15_goals_percent",
    "Over 2.5 Goals %": "over_25_goals_percent",
    "Both Teams Score %": "both_teams_score_percent",
    "Over 1.5": "over_15",
    "Over 2.5": "over_25",
    "BTTS": "btts",
}

ACCURACY_DATASET_STATS = [
    "goals", "shots_total", "shots_on_target", "corners", "fouls",
    "yellowcards", "tackles", "passes", "successful_passes",
    "total_crosses", "interceptions", "offsides",
]

ACCURACY_DATASET_SIMPLE_PERCENT_COLS = [
    "Home Odds %", "Draw Odds %", "Away Odds %",
    "Home Win %", "Draw %", "Away Win %",
    "Home Clean Sheet %", "Away Clean Sheet %",
]

ACCURACY_DATASET_OUTCOME_COLS = [
    "Home Win", "Draw", "Away Win", "Home Clean Sheet", "Away Clean Sheet",
]


# ──────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────

def to_val(v):
    """Convert pandas value to DB-safe python primitive."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def parse_percent(v):
    """'35.62%' → 35.62 (float). NaN/None → None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        s = v.replace("%", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return float(v)


def parse_bool(v):
    """Nullable bool from parquet values (True/False/None/NaN)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool,)):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    return None


# ──────────────────────────────────────────────────────────────────────────
#  Lookups
# ──────────────────────────────────────────────────────────────────────────

async def fetch_lookups():
    """Pull teams, fixtures (for season_id), comp_teams (for scope-resolve)."""
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM teams")
            teams = pd.DataFrame(await cur.fetchall(), columns=["id", "name"])
            await cur.execute("SELECT id, season_id, competition_id FROM fixtures")
            fixtures = pd.DataFrame(
                await cur.fetchall(),
                columns=["id", "season_id", "competition_id"],
            )
            await cur.execute(
                "SELECT competition_id, season_id, team_id FROM competition_season_teams"
            )
            comp_teams = pd.DataFrame(
                await cur.fetchall(),
                columns=["competition_id", "season_id", "team_id"],
            )
        return teams, fixtures, comp_teams
    finally:
        release_source_connection(conn)


def resolve_team_id_scoped(team_name, teams, scoped_team_ids):
    """Competition-scoped name → id. Returns None if unresolved."""
    if team_name is None or (isinstance(team_name, float) and math.isnan(team_name)):
        return None
    # Prefer a team_id within the comp's team pool.
    scoped = teams.loc[
        (teams["id"].isin(scoped_team_ids)) & (teams["name"] == team_name), "id"
    ]
    if not scoped.empty:
        return int(scoped.iloc[0])
    # Fallback to first global match.
    match = teams.loc[teams["name"] == team_name, "id"]
    if match.empty:
        return None
    return int(match.iloc[0])


# ──────────────────────────────────────────────────────────────────────────
#  projection_model_dataset seeder
# ──────────────────────────────────────────────────────────────────────────

def build_model_dataset_rows(df, teams, fixtures, comp_teams):
    """Transform one league's parquet → list of INSERT tuples.

    Some historical parquet files (Eredivisie, Liga Portugal, Super Lig,
    Champions League, Europa League) are contaminated: they contain rows
    from OTHER competitions but all labelled with the current comp_id,
    because the first projection run for those leagues read from the
    all_leagues fallback file and baked it into the league-specific file.

    To produce clean DB data we IGNORE the parquet's comp_id claim and
    use the fixture's ACTUAL competition_id from the DB fixtures table.
    Rows where the fixture doesn't exist in the DB (or has a different
    comp_id than the parquet claims) get their comp_id overridden from
    DB truth. This means a contaminated Eredivisie parquet's rows for
    MLS fixtures get inserted with competition_id=779 (MLS), not 72.
    """
    # Pre-compute (season_id, competition_id) lookup from the fixtures DB
    fix_by_id = {
        int(r["id"]): (int(r["season_id"]), int(r["competition_id"]))
        for _, r in fixtures.iterrows()
    }

    rows = []
    unresolved_team = 0
    unresolved_opponent = 0
    missing_fixture = 0
    contamination_skipped = 0

    for claimed_comp_id, comp_df in df.groupby("comp_id"):
        scoped_ids = comp_teams.loc[
            comp_teams["competition_id"] == int(claimed_comp_id), "team_id"
        ].unique()

        for _, row in comp_df.iterrows():
            fixture_id = int(row["id"])
            fix_info = fix_by_id.get(fixture_id)
            if fix_info is None:
                missing_fixture += 1
                continue
            actual_season_id, actual_comp_id = fix_info

            # Contamination check: if the parquet claims this row belongs
            # to comp X but the fixture actually belongs to comp Y, SKIP.
            # These are rows that got pooled into the wrong league's parquet
            # file (e.g. MLS fixtures sitting in Eredivisie parquet). The
            # correct data for that fixture lives in the other league's
            # parquet, so skipping here avoids overwriting it.
            if int(claimed_comp_id) != actual_comp_id:
                contamination_skipped += 1
                continue

            team_id = resolve_team_id_scoped(row["Team"], teams, scoped_ids)
            opponent_id = resolve_team_id_scoped(row["Opponent"], teams, scoped_ids)

            if team_id is None:
                unresolved_team += 1
                continue
            if opponent_id is None:
                unresolved_opponent += 1
                continue

            tup = [
                fixture_id,
                actual_comp_id,
                actual_season_id,
                team_id,
                opponent_id,
                to_val(row["Team"]),
                to_val(row["Opponent"]),
                to_val(row["Venue"]),
                to_val(row["kickoff_datetime"]),
            ]
            # 11 actuals in the order matching the INSERT sql below
            for stat in MODEL_DATASET_STATS:
                col = "Team " + stat.replace("_", " ").title()
                # title() capitalises each word; need to re-map 'On' etc.
                # simplest: use the exact parquet col by lookup
                col = _title_parquet_col(stat, history=False)
                tup.append(to_val(row[col]))
            # 22 history features: team_x_history, opponent_x_history_against (x 11 stats)
            for stat in MODEL_DATASET_STATS:
                th_col = _title_parquet_col(stat, history=True, opponent=False)
                oh_col = _title_parquet_col(stat, history=True, opponent=True)
                tup.append(to_val(row[th_col]))
                tup.append(to_val(row[oh_col]))
            rows.append(tuple(tup))

    return rows, {
        "unresolved_team": unresolved_team,
        "unresolved_opponent": unresolved_opponent,
        "missing_fixture": missing_fixture,
        "contamination_skipped": contamination_skipped,
    }


def _title_parquet_col(stat_snake, history=False, opponent=False):
    """'shots_on_target', history=True, opponent=False → 'Team Shots On Target History'"""
    words = stat_snake.replace("_", " ").split()
    title = " ".join(w.capitalize() for w in words)
    if history:
        if opponent:
            return f"Opponent {title} History Against"
        return f"Team {title} History"
    return f"Team {title}"


MODEL_DATASET_INSERT_SQL = """
INSERT INTO projection_model_dataset (
    fixture_id, competition_id, season_id, team_id, opponent_id,
    team_name, opponent_name, venue, kickoff_datetime,
    team_shots_total, team_shots_on_target, team_corners, team_fouls,
    team_yellowcards, team_tackles, team_passes, team_successful_passes,
    team_interceptions, team_total_crosses, team_offsides,
    team_shots_total_history, opponent_shots_total_history_against,
    team_shots_on_target_history, opponent_shots_on_target_history_against,
    team_corners_history, opponent_corners_history_against,
    team_fouls_history, opponent_fouls_history_against,
    team_yellowcards_history, opponent_yellowcards_history_against,
    team_tackles_history, opponent_tackles_history_against,
    team_passes_history, opponent_passes_history_against,
    team_successful_passes_history, opponent_successful_passes_history_against,
    team_interceptions_history, opponent_interceptions_history_against,
    team_total_crosses_history, opponent_total_crosses_history_against,
    team_offsides_history, opponent_offsides_history_against,
    created_at, updated_at
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    NOW(), NOW()
) ON DUPLICATE KEY UPDATE
    competition_id = VALUES(competition_id),
    season_id = VALUES(season_id),
    opponent_id = VALUES(opponent_id),
    team_name = VALUES(team_name),
    opponent_name = VALUES(opponent_name),
    venue = VALUES(venue),
    kickoff_datetime = VALUES(kickoff_datetime),
    team_shots_total = VALUES(team_shots_total),
    team_shots_on_target = VALUES(team_shots_on_target),
    team_corners = VALUES(team_corners),
    team_fouls = VALUES(team_fouls),
    team_yellowcards = VALUES(team_yellowcards),
    team_tackles = VALUES(team_tackles),
    team_passes = VALUES(team_passes),
    team_successful_passes = VALUES(team_successful_passes),
    team_interceptions = VALUES(team_interceptions),
    team_total_crosses = VALUES(team_total_crosses),
    team_offsides = VALUES(team_offsides),
    team_shots_total_history = VALUES(team_shots_total_history),
    opponent_shots_total_history_against = VALUES(opponent_shots_total_history_against),
    team_shots_on_target_history = VALUES(team_shots_on_target_history),
    opponent_shots_on_target_history_against = VALUES(opponent_shots_on_target_history_against),
    team_corners_history = VALUES(team_corners_history),
    opponent_corners_history_against = VALUES(opponent_corners_history_against),
    team_fouls_history = VALUES(team_fouls_history),
    opponent_fouls_history_against = VALUES(opponent_fouls_history_against),
    team_yellowcards_history = VALUES(team_yellowcards_history),
    opponent_yellowcards_history_against = VALUES(opponent_yellowcards_history_against),
    team_tackles_history = VALUES(team_tackles_history),
    opponent_tackles_history_against = VALUES(opponent_tackles_history_against),
    team_passes_history = VALUES(team_passes_history),
    opponent_passes_history_against = VALUES(opponent_passes_history_against),
    team_successful_passes_history = VALUES(team_successful_passes_history),
    opponent_successful_passes_history_against = VALUES(opponent_successful_passes_history_against),
    team_interceptions_history = VALUES(team_interceptions_history),
    opponent_interceptions_history_against = VALUES(opponent_interceptions_history_against),
    team_total_crosses_history = VALUES(team_total_crosses_history),
    opponent_total_crosses_history_against = VALUES(opponent_total_crosses_history_against),
    team_offsides_history = VALUES(team_offsides_history),
    opponent_offsides_history_against = VALUES(opponent_offsides_history_against),
    updated_at = NOW()
"""


async def seed_model_dataset(teams, fixtures, comp_teams, dry_run):
    files = sorted(glob.glob(f"{DATA_DIR}/*_model_dataset_with_history.parquet"))
    files = [f for f in files if "all_leagues" not in os.path.basename(f).lower()]
    logger.info(f"model_dataset: {len(files)} parquet files found")

    total_rows_in = 0
    total_rows_out = 0
    total_unresolved = {"unresolved_team": 0, "unresolved_opponent": 0,
                        "missing_fixture": 0, "contamination_skipped": 0}

    for path in files:
        league = os.path.basename(path).replace("_model_dataset_with_history.parquet", "")
        df = pd.read_parquet(path)
        total_rows_in += len(df)
        rows, stats = build_model_dataset_rows(df, teams, fixtures, comp_teams)
        for k in total_unresolved:
            total_unresolved[k] += stats[k]
        logger.info(
            f"  {league}: parquet={len(df)} resolved={len(rows)} "
            f"dropped(team={stats['unresolved_team']} opp={stats['unresolved_opponent']} "
            f"missing_fixture={stats['missing_fixture']} contaminated={stats['contamination_skipped']})"
        )
        if not dry_run and rows:
            await insert_chunks(MODEL_DATASET_INSERT_SQL, rows, label=f"[model {league}]")
        total_rows_out += len(rows)

    logger.info(
        f"model_dataset DONE — parquet total={total_rows_in} inserted={total_rows_out} "
        f"dropped: {total_unresolved}"
    )


# ──────────────────────────────────────────────────────────────────────────
#  projection_accuracy_dataset seeder
# ──────────────────────────────────────────────────────────────────────────

def build_accuracy_rows(df, teams, fixtures, comp_teams):
    """Same contamination treatment as model_dataset: use the fixture's
    ACTUAL competition_id from the DB, not the parquet's claim."""
    fix_by_id = {
        int(r["id"]): int(r["competition_id"])
        for _, r in fixtures.iterrows()
    }
    rows = []
    unresolved_home = 0
    unresolved_away = 0
    missing_fixture = 0
    contamination_skipped = 0

    for claimed_comp_id, comp_df in df.groupby("comp_id"):
        scoped_ids = comp_teams.loc[
            comp_teams["competition_id"] == int(claimed_comp_id), "team_id"
        ].unique()

        for _, row in comp_df.iterrows():
            fixture_id = int(row["fixture_id"])
            actual_comp_id = fix_by_id.get(fixture_id)
            if actual_comp_id is None:
                missing_fixture += 1
                continue
            # Same contamination rule as model_dataset.
            if int(claimed_comp_id) != actual_comp_id:
                contamination_skipped += 1
                continue

            home_id = resolve_team_id_scoped(row.get("Home Team"), teams, scoped_ids)
            away_id = resolve_team_id_scoped(row.get("Away Team"), teams, scoped_ids)
            if home_id is None:
                unresolved_home += 1
                continue
            if away_id is None:
                unresolved_away += 1
                continue

            tup = [
                fixture_id,
                actual_comp_id,
                home_id,
                away_id,
                to_val(row.get("Home Team")),
                to_val(row.get("Away Team")),
                to_val(row["kickoff_datetime"]),
            ]
            # 72 stat cols: for each stat, (total, total_projected, home, home_projected, away, away_projected)
            for stat in ACCURACY_DATASET_STATS:
                title = " ".join(w.capitalize() for w in stat.split("_"))
                for venue in ["Total", "Home", "Away"]:
                    tup.append(to_val(row.get(f"{venue} {title}")))
                    tup.append(to_val(row.get(f"{venue} Projected {title}")))
            # Odds (already decimal, not strings)
            tup.append(to_val(row.get("Home Odds %")))
            tup.append(to_val(row.get("Draw Odds %")))
            tup.append(to_val(row.get("Away Odds %")))
            # Projection %s (stored as '35.62%' strings in parquet → parse)
            tup.append(parse_percent(row.get("Home Win %")))
            tup.append(parse_percent(row.get("Draw %")))
            tup.append(parse_percent(row.get("Away Win %")))
            tup.append(parse_percent(row.get("Home Clean Sheet %")))
            tup.append(parse_percent(row.get("Away Clean Sheet %")))
            tup.append(parse_percent(row.get("Over 1.5 Goals %")))
            tup.append(parse_percent(row.get("Over 2.5 Goals %")))
            tup.append(parse_percent(row.get("Both Teams Score %")))
            # Outcomes
            tup.append(parse_bool(row.get("Home Win")))
            tup.append(parse_bool(row.get("Draw")))
            tup.append(parse_bool(row.get("Away Win")))
            tup.append(parse_bool(row.get("Home Clean Sheet")))
            tup.append(parse_bool(row.get("Away Clean Sheet")))
            tup.append(parse_bool(row.get("Over 1.5")))
            tup.append(parse_bool(row.get("Over 2.5")))
            tup.append(parse_bool(row.get("BTTS")))
            rows.append(tuple(tup))

    return rows, {
        "unresolved_home": unresolved_home,
        "unresolved_away": unresolved_away,
        "missing_fixture": missing_fixture,
        "contamination_skipped": contamination_skipped,
    }


def _build_accuracy_insert_sql():
    """Generate the INSERT sql programmatically so it stays in sync with table schema."""
    cols = [
        "fixture_id", "competition_id", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name", "kickoff_datetime",
    ]
    for stat in ACCURACY_DATASET_STATS:
        for venue in ["total", "home", "away"]:
            cols.append(f"{venue}_{stat}")
            cols.append(f"{venue}_projected_{stat}")
    cols.extend([
        "home_odds_percent", "draw_odds_percent", "away_odds_percent",
        "home_win_percent", "draw_percent", "away_win_percent",
        "home_clean_sheet_percent", "away_clean_sheet_percent",
        "over_15_goals_percent", "over_25_goals_percent", "both_teams_score_percent",
        "home_win", "draw", "away_win",
        "home_clean_sheet", "away_clean_sheet",
        "over_15", "over_25", "btts",
    ])
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    update_list = ",\n    ".join(f"{c} = VALUES({c})" for c in cols if c != "fixture_id")
    return f"""
INSERT INTO projection_accuracy_dataset (
    {col_list},
    created_at, updated_at
) VALUES ({placeholders}, NOW(), NOW())
ON DUPLICATE KEY UPDATE
    {update_list},
    updated_at = NOW()
"""


ACCURACY_DATASET_INSERT_SQL = _build_accuracy_insert_sql()


async def seed_accuracy_dataset(teams, fixtures, comp_teams, dry_run):
    files = sorted(glob.glob(f"{DATA_DIR}/*_accuracy_dataset.parquet"))
    files = [f for f in files if "all_leagues" not in os.path.basename(f).lower()]
    logger.info(f"accuracy_dataset: {len(files)} parquet files found")

    total_rows_in = 0
    total_rows_out = 0
    total_unresolved = {"unresolved_home": 0, "unresolved_away": 0,
                        "missing_fixture": 0, "contamination_skipped": 0}

    for path in files:
        league = os.path.basename(path).replace("_accuracy_dataset.parquet", "")
        df = pd.read_parquet(path)
        total_rows_in += len(df)
        rows, stats = build_accuracy_rows(df, teams, fixtures, comp_teams)
        for k in total_unresolved:
            total_unresolved[k] += stats[k]
        logger.info(
            f"  {league}: parquet={len(df)} resolved={len(rows)} "
            f"dropped(home={stats['unresolved_home']} away={stats['unresolved_away']} "
            f"missing_fixture={stats['missing_fixture']} contaminated={stats['contamination_skipped']})"
        )
        if not dry_run and rows:
            await insert_chunks(ACCURACY_DATASET_INSERT_SQL, rows, label=f"[acc {league}]")
        total_rows_out += len(rows)

    logger.info(
        f"accuracy_dataset DONE — parquet total={total_rows_in} inserted={total_rows_out} "
        f"dropped: {total_unresolved}"
    )


# ──────────────────────────────────────────────────────────────────────────
#  projection_models seeder (register existing .sav files)
# ──────────────────────────────────────────────────────────────────────────

PASS_STATS = {"Passes", "Successful Passes"}


async def seed_projection_models(comps_df, dry_run):
    """Register one row per .sav file with is_active=1 so the DB-backed
    loader has something to serve immediately after cutover."""
    rows = []
    for league_dir in sorted(os.listdir(MODEL_BUILDS_DIR)):
        full = os.path.join(MODEL_BUILDS_DIR, league_dir)
        if not os.path.isdir(full) or league_dir.startswith("__"):
            continue
        is_all_leagues = (league_dir == "All Leagues")
        competition_id = None
        if not is_all_leagues:
            match = comps_df.loc[comps_df["name"] == league_dir, "id"]
            if match.empty:
                # Brazil Serie A historical pair: league dir "Brazil Serie A" → comp id 648
                if league_dir == "Brazil Serie A":
                    competition_id = 648
                else:
                    logger.warning(f"  skipping '{league_dir}': no matching competition")
                    continue
            else:
                competition_id = int(match.iloc[0])

        for fn in sorted(os.listdir(full)):
            if not fn.endswith(".sav"):
                continue
            # Parse: "Brazil Serie A_Corners_model.sav" → stat "Corners"
            # or for All Leagues: "All_Leagues_Corners_model.sav" → stat "Corners"
            base = fn[:-len("_model.sav")]
            if is_all_leagues and base.startswith("All_Leagues_"):
                stat = base[len("All_Leagues_"):].replace("_", " ")
            elif base.startswith(f"{league_dir}_"):
                stat = base[len(league_dir) + 1:]
            else:
                logger.warning(f"  unexpected filename: {fn}")
                continue

            file_path = os.path.join(full, fn)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            algorithm = "fit_model" if stat in PASS_STATS else "grid_search"

            rows.append((
                competition_id,
                stat,
                algorithm,
                None,              # hyperparams_json — unknown for pre-existing .sav
                mtime,             # trained_at
                0,                 # trained_on_n_rows — unknown for pre-existing
                None,              # holdout_mae
                None,              # holdout_r2
                file_path,
                1,                 # is_active
                mtime,             # promoted_at = mtime (best guess)
                None,              # demoted_at
                "initial-seed",    # promotion_reason
            ))

    logger.info(f"projection_models: {len(rows)} .sav files to register")
    if dry_run or not rows:
        for r in rows[:5]:
            logger.info(f"  sample: {r}")
        return

    sql = """
    INSERT IGNORE INTO projection_models (
        competition_id, stat_name, algorithm, hyperparams_json,
        trained_at, trained_on_n_rows, holdout_mae, holdout_r2,
        file_path, is_active, promoted_at, demoted_at, promotion_reason,
        created_at, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """
    await insert_chunks(sql, rows, label="[models]")


# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 500


async def insert_chunks(sql, values, label=""):
    conn = await get_connection()
    try:
        total = 0
        n_chunks = (len(values) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(0, len(values), CHUNK_SIZE):
            chunk = values[i:i + CHUNK_SIZE]
            async with conn.cursor() as cur:
                affected = await cur.executemany(sql, chunk)
            await conn.commit()
            total += (affected or 0)
            logger.info(f"  {label} chunk {i // CHUNK_SIZE + 1}/{n_chunks}: {affected} rows affected")
        logger.info(f"  {label} total affected: {total}")
    finally:
        conn.close()


async def fetch_comps():
    conn = await get_source_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, name FROM competitions")
            rows = await cur.fetchall()
        return pd.DataFrame(rows, columns=["id", "name"])
    finally:
        release_source_connection(conn)


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

async def main(only, dry_run):
    await init_db_pool()
    await source_init_db_pool()

    logger.info("Fetching lookups (teams, fixtures, comp_teams, comps)...")
    teams, fixtures, comp_teams = await fetch_lookups()
    comps = await fetch_comps()
    logger.info(
        f"  teams={len(teams)} fixtures={len(fixtures)} "
        f"comp_teams={len(comp_teams)} competitions={len(comps)}"
    )

    if only in (None, "model"):
        logger.info("=== SEEDING projection_model_dataset ===")
        await seed_model_dataset(teams, fixtures, comp_teams, dry_run)

    if only in (None, "accuracy"):
        logger.info("=== SEEDING projection_accuracy_dataset ===")
        await seed_accuracy_dataset(teams, fixtures, comp_teams, dry_run)

    if only in (None, "models"):
        logger.info("=== SEEDING projection_models ===")
        await seed_projection_models(comps, dry_run)

    logger.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["model", "accuracy", "models"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.only, args.dry_run))
