"""
projection_models table — one row per trained .sav model version.

See database/migrations/2026_04_20_200002_create_projection_models_table.php
for schema. Key queries:

  insert_projection_model()   → INSERT a new version, return id. is_active
                                defaults to 0; Phase 5 promotion step sets
                                it to 1 after the guardrail clears.
  fetch_active_model()         → SELECT the currently-active model row for
                                a given (competition_id, stat_name). Used
                                by the retrain loop to compare holdout MAE
                                against the incumbent.
  promote_model()              → atomically flip is_active: demote incumbent,
                                activate new. Lives here so the transaction
                                boundary is obvious. Not yet called by the
                                dry-run retrain; Phase 5 wires it in.
"""
import asyncio
import json
import logging
import os
import shutil
from decimal import Decimal
from typing import Optional

import app.database as _db
from app.database import get_connection

logger = logging.getLogger("projection_model_repo")


async def insert_projection_model(
    competition_id: Optional[int],
    stat_name: str,
    algorithm: str,
    hyperparams: dict,
    trained_on_n_rows: int,
    holdout_mae: Optional[float],
    holdout_r2: Optional[float],
    file_path: str,
) -> int:
    """Insert a new model version with is_active=0. Returns the new row id.
    competition_id=None signals an "All Leagues" fallback model (NULL column)."""
    sql = """
        INSERT INTO projection_models (
            competition_id, stat_name, algorithm, hyperparams_json,
            trained_at, trained_on_n_rows, holdout_mae, holdout_r2,
            file_path, is_active, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s, 0, NOW(), NOW())
    """
    params = (
        competition_id, stat_name, algorithm, json.dumps(hyperparams),
        trained_on_n_rows, holdout_mae, holdout_r2, file_path,
    )
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(sql, params)
            new_id = cursor.lastrowid
            await conn.commit()
            return new_id
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def fetch_active_model(competition_id: Optional[int], stat_name: str) -> Optional[dict]:
    """Return the currently-active model row for (competition_id, stat_name),
    or None if none exists. competition_id IS NULL for "All Leagues"."""
    if competition_id is None:
        sql = (
            "SELECT id, holdout_mae, holdout_r2, file_path, trained_at "
            "FROM projection_models "
            "WHERE is_active = 1 AND stat_name = %s AND competition_id IS NULL "
            "LIMIT 1"
        )
        params = (stat_name,)
    else:
        sql = (
            "SELECT id, holdout_mae, holdout_r2, file_path, trained_at "
            "FROM projection_models "
            "WHERE is_active = 1 AND stat_name = %s AND competition_id = %s "
            "LIMIT 1"
        )
        params = (stat_name, int(competition_id))

    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await cursor.execute(sql, params)
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "holdout_mae": float(row[1]) if isinstance(row[1], Decimal) else row[1],
                "holdout_r2": float(row[2]) if isinstance(row[2], Decimal) else row[2],
                "file_path": row[3],
                "trained_at": row[4],
            }
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


def _unversioned_target_path(versioned_file_path: str, league_dir_name: str, stat_name: str) -> str:
    """Compute the fixed-filename .sav path that load_model() reads at
    projection runtime. Lives next to the versioned file in the same
    directory.

    Conventions match load_model() in statz_functions.py:
      Per-league:    {dir}/{league}_{stat}_model.sav
      All Leagues:   {dir}/All_Leagues_{stat}_model.sav  (note underscore)
    """
    if league_dir_name == "All Leagues":
        target_basename = f"All_Leagues_{stat_name}_model.sav"
    else:
        target_basename = f"{league_dir_name}_{stat_name}_model.sav"
    return os.path.join(os.path.dirname(versioned_file_path), target_basename)


async def promote_model(new_model_id: int, reason: str):
    """Atomically: demote incumbent, activate new, AND swap the .sav file
    that load_model() reads at projection runtime.

    Two-part atomicity:
      1. DB transaction: demote + activate.
      2. After commit: rename a pre-staged .tmp copy over the unversioned
         filename. os.replace is atomic on POSIX.

    Why both halves matter: load_model() in statz_functions.py reads a
    fixed filename ({league}_{stat}_model.sav) — it does NOT consult
    projection_models.is_active. Without the file swap, promotion would
    flip DB metadata but the runtime would keep loading the old model.
    Caught 2026-05-05 reviewing Phase 5 wiring before the first
    auto-promote firing.

    Failure handling:
      - Source .sav missing → raise (the new model's file_path was
        deleted — abort, don't promote)
      - DB transaction fails → rollback + clean up .tmp + raise
      - File rename fails after DB commit → log loud error, leave DB
        committed (recoverable: next promotion overwrites; rollback
        button can re-attempt)
    """
    conn = None
    target_tmp = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            # Fetch new model's (comp_id, stat, file_path) + competition name
            # so we can compute the unversioned-filename target.
            await cursor.execute(
                "SELECT pm.competition_id, pm.stat_name, pm.file_path, c.name AS comp_name "
                "FROM projection_models pm "
                "LEFT JOIN competitions c ON c.id = pm.competition_id "
                "WHERE pm.id = %s",
                (new_model_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise ValueError(f"promote_model: id={new_model_id} not found")
            comp_id, stat_name, file_path, comp_name = row

            # Verify the versioned source exists BEFORE touching the DB.
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"promote_model: id={new_model_id} source file_path "
                    f"{file_path!r} does not exist; refusing to promote"
                )

            league_dir_name = comp_name if comp_id is not None else "All Leagues"
            target_path = _unversioned_target_path(file_path, league_dir_name, stat_name)
            target_tmp = target_path + ".tmp"

            # Pre-stage the file copy as .tmp so the post-commit step is just
            # an atomic rename. If the .tmp copy itself fails (disk full /
            # permission), abort before any DB write.
            shutil.copy2(file_path, target_tmp)

            await conn.begin()
            try:
                # Demote any current incumbent
                if comp_id is None:
                    await cursor.execute(
                        "UPDATE projection_models SET is_active = 0, demoted_at = NOW() "
                        "WHERE is_active = 1 AND stat_name = %s AND competition_id IS NULL",
                        (stat_name,),
                    )
                else:
                    await cursor.execute(
                        "UPDATE projection_models SET is_active = 0, demoted_at = NOW() "
                        "WHERE is_active = 1 AND stat_name = %s AND competition_id = %s",
                        (stat_name, int(comp_id)),
                    )
                # Activate the new one
                await cursor.execute(
                    "UPDATE projection_models SET is_active = 1, "
                    "promoted_at = NOW(), promotion_reason = %s "
                    "WHERE id = %s",
                    (reason, new_model_id),
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

            # DB transaction succeeded. Atomically swap .tmp → unversioned
            # filename. POSIX os.replace is atomic; on failure we log loud
            # but don't roll back the DB (next promotion will overwrite).
            try:
                os.replace(target_tmp, target_path)
                target_tmp = None  # consumed
                logger.info(
                    f"[projection_models] promoted id={new_model_id} "
                    f"(comp={comp_id}, stat={stat_name}): {reason} | "
                    f"file: {os.path.basename(file_path)} -> {os.path.basename(target_path)}"
                )
            except Exception as rename_err:
                logger.error(
                    f"[projection_models] DB promoted id={new_model_id} but "
                    f"file rename failed ({target_tmp} -> {target_path}): "
                    f"{rename_err}. Runtime will keep loading the previous "
                    f"unversioned file until the next successful promotion.",
                    exc_info=True,
                )
                raise
    finally:
        # Clean up .tmp if it's still around (DB rollback or rename failure)
        if target_tmp:
            try:
                if os.path.exists(target_tmp):
                    os.remove(target_tmp)
            except Exception:
                pass
        if conn and _db.pool:
            _db.pool.release(conn)
