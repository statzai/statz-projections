import logging
logger = logging.getLogger("projection")
from app.database import get_connection, pool


async def get_all_competition_season_team_in_batches(batch_size=10000):
    conn = None
    try:
        conn = await get_connection()
        async with conn.cursor() as cursor:
            sql = "SELECT * FROM fixture_projections"
            await cursor.execute(sql)

            while True:
                batch = await cursor.fetchmany(batch_size)
                if not batch:
                    break
                for row in batch:
                    yield row
    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        raise
    finally:
        if conn and pool:
            pool.release(conn)