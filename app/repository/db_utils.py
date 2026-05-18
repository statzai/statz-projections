import asyncio
import logging
import math
import aiomysql
import pandas as pd
import app.database as _db
from app.database import get_connection

logger = logging.getLogger("db_utils")

CHUNK_SIZE = 500
MAX_RETRIES = 3
QUERY_TIMEOUT = 120  # seconds per chunk before giving up and retrying

# Chunk size sizing notes:
# - max_allowed_packet on prod = 64 MB.
# - Widest table is model_dataset (~80 cols × ~50 chars = ~4 KB/row).
#   500 rows = ~2 MB per INSERT statement. ~32x packet-size headroom.
# - Bumped 50 → 500 on 2026-05-11 to claw back the +20s/league insert overhead
#   added by the player-prop expansion (8 markets vs 3 = 22 props vs 9 → 30k
#   rows for PL). Saves ~9 min off a full 23-league sweep.


def resolve_team_id(team_name, teams, competition_id=None, comp_teams=None):
    """
    Resolve a team name to its id via the `teams` dataframe, returning None
    if the name can't be mapped. Applies TEAM_NAME_FIXES so historical names
    like "Milan" map to their current counterparts ("AC Milan"). Used by the
    projection repos to populate `team_id` columns on insert, replacing the
    previous pattern of writing the name string directly.

    When `competition_id` and `comp_teams` are provided, the lookup is first
    restricted to teams registered in that competition (via the
    competition_season_teams mapping). This is the duplicate-name bug fix:
    without scoping, a flat name match could resolve "Liverpool" to FC
    Montevideo Uruguay (id=976) instead of Liverpool FC England (id=8).
    Same for "Everton" (FC England id=13 vs Viña del Mar Chile id=15064)
    and "Nacional" (CD Nacional Portugal vs Club Nacional Uruguay).

    If the scoped lookup misses (e.g. team not yet registered for this
    comp's current season), we fall through to the global lookup with a
    WARNING — so the insert still lands but the scope-pool gap is visible.

    If the GLOBAL lookup finds multiple matches, we log a WARNING and pick
    the first — same degraded behaviour as before the fix, but now visible.

    Returning None rather than raising lets insert rows with unresolvable
    names still land (with team_id NULL) instead of breaking the whole batch.
    """
    if team_name is None or (isinstance(team_name, float) and math.isnan(team_name)):
        return None
    try:
        from app.services.statz_functions import TEAM_NAME_FIXES
        name = TEAM_NAME_FIXES.get(team_name, team_name)

        # Scoped lookup first — restrict to teams registered in this competition.
        # If competition_id is None but comp_teams is non-empty, treat comp_teams
        # as already pre-filtered by the caller.
        if comp_teams is not None and not comp_teams.empty:
            if competition_id is not None:
                # Euro comps pass a LIST of domestic league IDs; scalar `==`
                # errored with "Lengths must match". .isin() handles both.
                if isinstance(competition_id, (list, tuple)) or (hasattr(competition_id, '__iter__') and not isinstance(competition_id, (str, int, float))):
                    comp_id_list = competition_id
                else:
                    comp_id_list = [competition_id]
                scoped_ids = comp_teams.loc[
                    comp_teams['competition_id'].isin(comp_id_list), 'team_id'
                ].unique()
            else:
                scoped_ids = comp_teams['team_id'].unique() if 'team_id' in comp_teams.columns else []

            if len(scoped_ids) > 0:
                scoped = teams.loc[
                    (teams['id'].isin(scoped_ids)) & (teams['name'] == name), 'id'
                ]
                if not scoped.empty:
                    return int(scoped.iloc[0])
            logger.warning(
                f"resolve_team_id({team_name!r}): no match within competition {competition_id} scope — falling back to global lookup"
            )

        match = teams.loc[teams['name'] == name, 'id']
        if match.empty:
            return None
        if len(match) > 1:
            logger.warning(
                f"resolve_team_id({team_name!r}): ambiguous — {len(match)} global matches ({list(match)}), picking first (comp_id={competition_id})"
            )
        return int(match.iloc[0])
    except Exception as e:
        logger.warning(f"resolve_team_id({team_name!r}) failed: {type(e).__name__}: {e}")
        return None


async def _get_fresh_connection(label: str, chunk_info: str):
    """Acquire a connection with logging."""
    logger.info(f"{label} {chunk_info} acquiring connection...")
    conn = await asyncio.wait_for(get_connection(), timeout=30)
    logger.info(f"{label} {chunk_info} connection acquired")
    return conn


async def fetch_all(sql: str, params: tuple = ()) -> list:
    """Run a SELECT and return all rows as a list of tuples. Self-contained:
    acquires and releases its own pooled connection."""
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await asyncio.wait_for(cursor.execute(sql, params), timeout=QUERY_TIMEOUT)
            return list(await cursor.fetchall())
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def execute(sql: str, params: tuple = ()) -> int:
    """Run a single write statement (DELETE/UPDATE/INSERT), commit, return the
    affected row count. Self-contained connection. For bulk inserts use
    execute_chunked instead."""
    conn = None
    try:
        conn = await asyncio.wait_for(get_connection(), timeout=30)
        async with conn.cursor() as cursor:
            await asyncio.wait_for(cursor.execute(sql, params), timeout=QUERY_TIMEOUT)
            await asyncio.wait_for(conn.commit(), timeout=30)
            return cursor.rowcount
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)


async def execute_chunked(sql: str, values: list, label: str = "") -> int:
    """
    Execute SQL in chunks of CHUNK_SIZE.
    Uses a single connection for all chunks (reuses it instead of acquire/release per chunk).
    On OperationalError or timeout, gets a fresh connection and retries the same chunk.
    Data errors (IntegrityError, etc.) are raised immediately.
    """
    if not values:
        return 0

    # Replace inf/-inf/nan with None so MySQL doesn't choke
    def _clean(v):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return None
        return v

    values = [tuple(_clean(v) for v in row) for row in values]

    total_rows = 0
    chunks = [values[i:i + CHUNK_SIZE] for i in range(0, len(values), CHUNK_SIZE)]
    logger.info(f"{label} inserting {len(values)} rows in {len(chunks)} chunk(s)")

    conn = await _get_fresh_connection(label, f"chunk 1/{len(chunks)}")
    chunk_idx = 0
    retries = 0

    try:
        while chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            chunk_info = f"chunk {chunk_idx + 1}/{len(chunks)}"
            try:
                async with conn.cursor() as cursor:
                    await asyncio.wait_for(cursor.executemany(sql, chunk), timeout=QUERY_TIMEOUT)
                    await asyncio.wait_for(conn.commit(), timeout=30)
                    total_rows += cursor.rowcount
                    logger.info(f"{label} {chunk_info} OK ({cursor.rowcount} rows)")
                chunk_idx += 1
                retries = 0  # reset retries on success
            except (aiomysql.OperationalError, asyncio.TimeoutError) as e:
                if retries >= MAX_RETRIES:
                    logger.error(f"{label} {chunk_info} FAILED after {MAX_RETRIES} retries: {type(e).__name__}: {e}")
                    raise
                retries += 1
                wait = 2 ** (retries - 1)  # 1s, 2s, 4s
                logger.warning(
                    f"{label} {chunk_info} retryable error (attempt {retries}/{MAX_RETRIES}), "
                    f"getting fresh connection in {wait}s: {type(e).__name__}: {e}"
                )
                try:
                    await asyncio.wait_for(conn.rollback(), timeout=5)
                except Exception:
                    pass
                if _db.pool:
                    _db.pool.release(conn)
                conn = None
                await asyncio.sleep(wait)
                conn = await _get_fresh_connection(label, chunk_info)
            except Exception as e:
                try:
                    await asyncio.wait_for(conn.rollback(), timeout=5)
                except Exception:
                    pass
                logger.error(f"{label} {chunk_info} data error (no retry): {type(e).__name__}: {e}")
                raise
    finally:
        if conn and _db.pool:
            _db.pool.release(conn)

    logger.info(f"{label} done — {total_rows} rows affected")
    return total_rows
