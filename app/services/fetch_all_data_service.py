import asyncio
import logging
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path

import app.source_database as _src_db
from app.source_database import get_source_connection, check_source_connection, source_init_db_pool, release_source_connection

logger = logging.getLogger("fetch_data")


SEASON_FILTER_FPS = """
        fps.fixture_id IN (
            SELECT f.id FROM fixtures f
            WHERE f.season_id IN (
                SELECT season_id FROM (
                    SELECT competition_id, season_id,
                           ROW_NUMBER() OVER (PARTITION BY competition_id ORDER BY season_id DESC) AS rn
                    FROM competition_season_teams
                ) ranked
                WHERE rn <= 2
            )
        )
"""

SEASON_FILTER_FTS = """
        fts.fixture_id IN (
            SELECT f.id FROM fixtures f
            WHERE f.season_id IN (
                SELECT season_id FROM (
                    SELECT competition_id, season_id,
                           ROW_NUMBER() OVER (PARTITION BY competition_id ORDER BY season_id DESC) AS rn
                    FROM competition_season_teams
                ) ranked
                WHERE rn <= 2
            )
        )
"""


class FetchAllDataService:
    CURRENT_DIR = Path(__file__).resolve().parent
    APP_DIR = CURRENT_DIR.parent

    DATA_FOLDER_PATH = APP_DIR / "data"
    LAST_FETCH_FILE = DATA_FOLDER_PATH / "last_fetch.txt"

    MAX_RETRIES = 2

    @staticmethod
    def _read_last_fetch_time():
        try:
            if FetchAllDataService.LAST_FETCH_FILE.exists():
                ts = FetchAllDataService.LAST_FETCH_FILE.read_text().strip()
                if ts:
                    return ts
        except Exception as e:
            logger.warning(f"fetch-data: Could not read last_fetch.txt: {e}")
        return None

    @staticmethod
    def _save_last_fetch_time(ts: str):
        try:
            FetchAllDataService.LAST_FETCH_FILE.write_text(ts)
        except Exception as e:
            logger.error(f"fetch-data: Could not write last_fetch.txt: {e}")

    @staticmethod
    def _merge_csv(filepath, new_df, id_column='id'):
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                existing_df = pd.read_csv(filepath, low_memory=False)
                if not new_df.empty:
                    updated_ids = set(new_df[id_column].astype(str))
                    existing_df = existing_df[~existing_df[id_column].astype(str).isin(updated_ids)]
                    merged = pd.concat([existing_df, new_df], ignore_index=True)
                    return merged
                else:
                    return existing_df
            except Exception as e:
                logger.warning(f"fetch-data: Could not read existing CSV for merge, doing full write: {e}")
                return new_df
        return new_df

    async def _fetch_table(
        self,
        table_name: str,
        query_fn,
        filepath: Path,
        id_column: str = 'id',
        incremental: bool = False,
        results: dict = None,
    ):
        """
        Generic table fetch with retry logic.

        query_fn: async callable(conn) -> pd.DataFrame
        filepath: destination CSV path
        id_column: column used for merge deduplication
        incremental: if True, merge new rows into existing CSV; if False, full replace
        results: dict to record outcome ('ok', 'fallback', 'failed')
        """
        t_start = time.monotonic()
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            conn = None
            try:
                conn = await get_source_connection()
                df = await query_fn(conn)

                if incremental:
                    df = self._merge_csv(filepath, df, id_column=id_column)

                df.to_csv(filepath, index=False)
                elapsed = time.monotonic() - t_start
                logger.info(f"[{table_name}] OK — {len(df)} rows ({elapsed:.1f}s)")
                if results is not None:
                    results[table_name] = 'ok'
                return df

            except Exception as e:
                last_error = e
                elapsed = time.monotonic() - t_start
                logger.warning(
                    f"[{table_name}] FAILED — {e} (retry {attempt}/{self.MAX_RETRIES})"
                )
                # Reset pool before retry so we get a genuinely fresh connection
                if attempt < self.MAX_RETRIES:
                    try:
                        if _src_db.source_pool:
                            _src_db.source_pool.close()
                            try:
                                await asyncio.wait_for(
                                    _src_db.source_pool.wait_closed(), timeout=5
                                )
                            except asyncio.TimeoutError:
                                pass
                            _src_db.source_pool = None
                        await source_init_db_pool()
                    except Exception as pool_err:
                        logger.warning(f"[{table_name}] Pool reinit failed: {pool_err}")
            finally:
                if conn is not None and _src_db.source_pool:
                    try:
                        release_source_connection(conn)
                    except Exception:
                        pass

        # All retries exhausted
        csv_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        if csv_exists:
            elapsed = time.monotonic() - t_start
            logger.warning(
                f"[{table_name}] SKIPPED — using existing CSV (all retries failed: {last_error})"
            )
            if results is not None:
                results[table_name] = 'fallback'
            return None
        else:
            logger.error(
                f"[{table_name}] ABORTED — no fallback CSV and fetch failed: {last_error}"
            )
            if results is not None:
                results[table_name] = 'failed'
            raise RuntimeError(
                f"[{table_name}] No fallback CSV available and all retries failed: {last_error}"
            )

    # ------------------------------------------------------------------
    # Query functions — each accepts a connection and returns a DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    async def _query_competition_season_teams(conn):
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM competition_season_teams"
            )
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_competitions(conn):
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM competitions")
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_seasons(conn):
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM seasons")
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_stats_types(conn):
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM stats_types")
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_transfermarkt_team_mappings(conn):
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT c.name AS league_name, ttm.*
                FROM transfermarkt_team_mappings ttm
                JOIN competitions c ON c.id = ttm.competition_id
                WHERE ttm.to_name IS NOT NULL
            """)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_team_ratings(conn):
        """Historical team ratings per (league, team, date). Projection services
        join new computations onto this to compute Movement week-over-week."""
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT c.name AS League,
                       t.name AS Team,
                       tr.date AS Date,
                       tr.attack AS Attack,
                       tr.defense AS Defense,
                       tr.overall AS Overall,
                       tr.movement AS Movement,
                       tr.inverse AS Inverse,
                       tr.team_id,
                       tr.competition_id
                FROM team_ratings tr
                JOIN competitions c ON c.id = tr.competition_id
                JOIN teams t ON t.id = tr.team_id
            """)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_promoted_team_ratings(conn):
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT c.name AS league_name, ptr.*
                FROM promoted_team_ratings ptr
                JOIN competitions c ON c.id = ptr.competition_id
            """)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    async def _query_projection_config(conn):
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT c.name AS league_name,
                       ca.name AS league_above_name,
                       cb.name AS league_below_name,
                       cpc.*
                FROM competition_projection_config cpc
                JOIN competitions c ON c.id = cpc.competition_id
                LEFT JOIN competitions ca ON ca.id = cpc.league_above_id
                LEFT JOIN competitions cb ON cb.id = cpc.league_below_id
            """)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _make_fps_query(last_fetch):
        async def _query(conn):
            if last_fetch:
                where = f"WHERE ({SEASON_FILTER_FPS}) AND fps.updated_at > %s"
                params = (last_fetch,)
            else:
                where = f"WHERE {SEASON_FILTER_FPS}"
                params = ()
            sql = f"""
                SELECT fps.*
                FROM fixture_player_stats fps
                {where}
            """
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        return _query

    @staticmethod
    def _make_fts_query(last_fetch):
        async def _query(conn):
            if last_fetch:
                where = f"WHERE ({SEASON_FILTER_FTS}) AND fts.updated_at > %s"
                params = (last_fetch,)
            else:
                where = f"WHERE {SEASON_FILTER_FTS}"
                params = ()
            sql = f"""
                SELECT fts.*
                FROM fixture_team_stats fts
                {where}
            """
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        return _query

    @staticmethod
    def _make_fixtures_query(last_fetch):
        # Exclude cancelled fixtures (Sportmonks state_id = 10) to avoid
        # projecting ghost fixtures. NULL state_id is treated as scheduled
        # (default) and included. Matches the filter Laravel uses
        # elsewhere (e.g. FixtureService::getFixtures()).
        state_filter = "(state_id != 10 OR state_id IS NULL)"
        async def _query(conn):
            if last_fetch:
                sql = f"SELECT * FROM fixtures WHERE updated_at > %s AND {state_filter}"
                params = (last_fetch,)
            else:
                sql = f"SELECT * FROM fixtures WHERE {state_filter}"
                params = ()
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        return _query

    @staticmethod
    def _make_players_query(last_fetch):
        async def _query(conn):
            if last_fetch:
                sql = "SELECT * FROM players WHERE updated_at > %s"
                params = (last_fetch,)
            else:
                sql = "SELECT * FROM players"
                params = ()
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        return _query

    @staticmethod
    def _make_standings_query(last_fetch):
        async def _query(conn):
            if last_fetch:
                sql = "SELECT * FROM standings WHERE updated_at > %s"
                params = (last_fetch,)
            else:
                sql = "SELECT * FROM standings"
                params = ()
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
        return _query

    # ------------------------------------------------------------------
    # Teams uses LIMIT/OFFSET pagination — handled separately
    # ------------------------------------------------------------------

    async def _fetch_teams(self, filepath: Path, results: dict):
        table_name = "teams"
        t_start = time.monotonic()
        last_error = None
        BATCH_SIZE = 500

        for attempt in range(1, self.MAX_RETRIES + 1):
            conn = None
            try:
                conn = await get_source_connection()
                all_rows = []
                cols = None
                offset = 0
                while True:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT * FROM teams LIMIT %s OFFSET %s",
                            (BATCH_SIZE, offset),
                        )
                        batch = await cur.fetchall()
                        if cols is None:
                            cols = [d[0] for d in cur.description]
                    if not batch:
                        break
                    all_rows.extend(batch)
                    if len(batch) < BATCH_SIZE:
                        break
                    offset += BATCH_SIZE

                df = pd.DataFrame(all_rows, columns=cols) if cols else pd.DataFrame()
                df.to_csv(filepath, index=False)
                elapsed = time.monotonic() - t_start
                logger.info(f"[{table_name}] OK — {len(df)} rows ({elapsed:.1f}s)")
                results[table_name] = 'ok'
                return df

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{table_name}] FAILED — {e} (retry {attempt}/{self.MAX_RETRIES})"
                )
                if attempt < self.MAX_RETRIES:
                    try:
                        if _src_db.source_pool:
                            _src_db.source_pool.close()
                            try:
                                await asyncio.wait_for(
                                    _src_db.source_pool.wait_closed(), timeout=5
                                )
                            except asyncio.TimeoutError:
                                pass
                            _src_db.source_pool = None
                        await source_init_db_pool()
                    except Exception as pool_err:
                        logger.warning(f"[{table_name}] Pool reinit failed: {pool_err}")
            finally:
                if conn is not None:
                    try:
                        release_source_connection(conn)
                    except Exception:
                        pass

        csv_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        if csv_exists:
            elapsed = time.monotonic() - t_start
            logger.warning(
                f"[{table_name}] SKIPPED — using existing CSV (all retries failed: {last_error})"
            )
            results[table_name] = 'fallback'
            return None
        else:
            logger.error(
                f"[{table_name}] ABORTED — no fallback CSV and fetch failed: {last_error}"
            )
            results[table_name] = 'failed'
            raise RuntimeError(
                f"[{table_name}] No fallback CSV available and all retries failed: {last_error}"
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def import_all_tables(self):
        output_folder = FetchAllDataService.DATA_FOLDER_PATH
        logger.info(f"fetch-data: Starting export to {output_folder}")
        start = datetime.now()
        fetch_timestamp = start.strftime('%Y-%m-%d %H:%M:%S')

        last_fetch = FetchAllDataService._read_last_fetch_time()
        if last_fetch:
            logger.info(f"fetch-data: Incremental mode — fetching changes since {last_fetch}")
        else:
            logger.info("fetch-data: Full mode — no previous fetch timestamp found")

        # Reset pool at startup
        if _src_db.source_pool:
            _src_db.source_pool.close()
            try:
                await asyncio.wait_for(_src_db.source_pool.wait_closed(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("fetch-data: wait_closed timed out, forcing reinit")
            _src_db.source_pool = None
        await source_init_db_pool()

        results = {}
        f = output_folder

        # --- Small / always-full tables ---
        logger.info("[competition_season_teams] START")
        await self._fetch_table(
            "competition_season_teams",
            self._query_competition_season_teams,
            f / "competition_season_teams.csv",
            incremental=False,
            results=results,
        )

        logger.info("[competitions] START")
        await self._fetch_table(
            "competitions",
            self._query_competitions,
            f / "competitions.csv",
            incremental=False,
            results=results,
        )

        logger.info("[seasons] START")
        await self._fetch_table(
            "seasons",
            self._query_seasons,
            f / "seasons.csv",
            incremental=False,
            results=results,
        )

        logger.info("[stats_types] START")
        await self._fetch_table(
            "stats_types",
            self._query_stats_types,
            f / "stats_types.csv",
            incremental=False,
            results=results,
        )

        logger.info("[projection_config] START")
        await self._fetch_table(
            "projection_config",
            self._query_projection_config,
            f / "projection_config.csv",
            incremental=False,
            results=results,
        )

        logger.info("[promoted_team_ratings] START")
        await self._fetch_table(
            "promoted_team_ratings",
            self._query_promoted_team_ratings,
            f / "promoted_team_ratings.csv",
            incremental=False,
            results=results,
        )

        logger.info("[team_ratings] START")
        await self._fetch_table(
            "team_ratings",
            self._query_team_ratings,
            f / "team_ratings.csv",
            incremental=False,
            results=results,
        )

        logger.info("[transfermarkt_team_mappings] START")
        await self._fetch_table(
            "transfermarkt_team_mappings",
            self._query_transfermarkt_team_mappings,
            f / "transfermarkt_team_mappings.csv",
            incremental=False,
            results=results,
        )

        # --- Large incremental tables ---
        logger.info("[fixture_player_stats] START")
        await self._fetch_table(
            "fixture_player_stats",
            self._make_fps_query(last_fetch),
            f / "fixture_player_stats.csv",
            id_column='id',
            incremental=bool(last_fetch),
            results=results,
        )

        logger.info("[fixture_team_stats] START")
        await self._fetch_table(
            "fixture_team_stats",
            self._make_fts_query(last_fetch),
            f / "fixture_team_stats.csv",
            id_column='id',
            incremental=bool(last_fetch),
            results=results,
        )

        logger.info("[fixtures] START")
        await self._fetch_table(
            "fixtures",
            self._make_fixtures_query(last_fetch),
            f / "fixtures.csv",
            id_column='id',
            incremental=bool(last_fetch),
            results=results,
        )

        logger.info("[players] START")
        await self._fetch_table(
            "players",
            self._make_players_query(last_fetch),
            f / "players.csv",
            id_column='id',
            incremental=bool(last_fetch),
            results=results,
        )

        logger.info("[standings] START")
        await self._fetch_table(
            "standings",
            self._make_standings_query(last_fetch),
            f / "standings.csv",
            id_column='id',
            incremental=bool(last_fetch),
            results=results,
        )

        # --- Teams (paginated) ---
        logger.info("[teams] START")
        await self._fetch_teams(f / "teams.csv", results)

        # --- Summary ---
        ok = [t for t, s in results.items() if s == 'ok']
        fallback = [t for t, s in results.items() if s == 'fallback']
        failed = [t for t, s in results.items() if s == 'failed']

        elapsed_total = (datetime.now() - start).total_seconds()
        logger.info(
            f"fetch-data: Summary — OK: {ok or 'none'} | "
            f"Fallback: {fallback or 'none'} | "
            f"Failed: {failed or 'none'} | "
            f"Total: {elapsed_total:.1f}s"
        )

        if not failed:
            FetchAllDataService._save_last_fetch_time(fetch_timestamp)

        logger.info("fetch-data: COMPLETE")
