import aiomysql
from typing import Optional

from app.config import Config

pool: Optional[aiomysql.Pool] = None

async def init_db_pool():
    global pool
    if pool is None:
        pool = await aiomysql.create_pool(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            db=Config.DB_NAME,
            minsize=Config.MIN_POOL_SIZE,
            maxsize=Config.MAX_POOL_SIZE,
            autocommit=False,
            connect_timeout=10,
            # Pin to UTC — see source_database.py for the rationale. Keeps the
            # write pool's NOW()/CURRENT_TIMESTAMP (and any DEFAULT timestamp
            # columns on projection tables) aligned with the UTC-stored
            # kickoffs, so reads and writes share one clock.
            init_command="SET time_zone = '+00:00'",
        )
    print(f"Pool is initialized: {pool}")

async def get_connection():
    if pool is None:
        raise RuntimeError("Database pool is not initialized. Call init_db_pool first.")
    conn = await pool.acquire()
    return conn

async def close_db_pool():
    if pool:
        pool.close()
        await pool.wait_closed()


async def check_connection(conn):
    try:
        await conn.execute('SELECT 1')
        return True
    except Exception as e:
        print(f"Connection is dead: {e}")
        return False
