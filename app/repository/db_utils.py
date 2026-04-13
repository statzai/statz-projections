import asyncio
import logging
import math
import aiomysql
import app.database as _db
from app.database import get_connection

logger = logging.getLogger("db_utils")

CHUNK_SIZE = 50
MAX_RETRIES = 3
QUERY_TIMEOUT = 120  # seconds per chunk before giving up and retrying


async def _get_fresh_connection(label: str, chunk_info: str):
    """Acquire a connection with logging."""
    logger.info(f"{label} {chunk_info} acquiring connection...")
    conn = await asyncio.wait_for(get_connection(), timeout=30)
    logger.info(f"{label} {chunk_info} connection acquired")
    return conn


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
