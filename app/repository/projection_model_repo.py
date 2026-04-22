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


async def promote_model(new_model_id: int, reason: str):
    """Atomically: demote incumbent, activate new. Invariant: at most one
    is_active=1 row per (competition_id, stat_name) at any moment.

    Not wired in Phase 4 (dry-run); Phase 5 calls this after the guardrail
    check. Added now so the transaction boundary lives alongside insert.
    """
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            # Fetch the new model's (comp_id, stat) so we know which incumbent to demote
            await cursor.execute(
                "SELECT competition_id, stat_name FROM projection_models WHERE id = %s",
                (new_model_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise ValueError(f"promote_model: id={new_model_id} not found")
            comp_id, stat_name = row

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
                logger.info(
                    f"[projection_models] promoted id={new_model_id} "
                    f"(comp={comp_id}, stat={stat_name}): {reason}"
                )
            except Exception:
                await conn.rollback()
                raise
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)
