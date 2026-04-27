"""LeagueDataLoader — scoped per-league DB loader for projection runs.

Phase 2 skeleton (Direct DB Query Migration). Replaces the read paths of
DataCache for the 3 large tables (player_stats, team_stats, fixtures_df) by
querying only the rows needed for ONE league projection.

Scope (per `loader_scope_rules.md`):
  - team_ids = current + previous season squads of: target league + extras
    + league_above + league_below.
  - player_ids = players whose current_team_id ∈ team_ids.
  - team_fixture_ids = fixtures involving any team_id, last 2yr.
  - player_fixture_ids = fixtures any player_id appeared in, last 2yr —
    captures cross-club history (e.g. Marc Bernal's Barcelona stats while
    now at Palace) AND international stats (Saka for England).
  - fixture_ids = team_fixture_ids ∪ player_fixture_ids → drives
    fixtures_df so all merge keys resolve.

  Per-table scope:
  - fixtures_df: WHERE id IN fixture_ids (the union)
  - team_stats: WHERE fixture_id IN team_fixture_ids only — both teams'
    rows loaded (no team_id filter) so get_opp_stats sees opponents.
    Cross-club fixtures intentionally EXCLUDED — no projection path
    iterates team_stats for out-of-scope clubs.
  - player_stats: WHERE player_id IN player_ids AND fixture_id IN
    fixture_ids (union).

NOT YET WIRED IN. This file is a skeleton — Phase 3 adds shadow-mode hookup.

Output schema matches DataCache attribute-by-attribute (same column names,
same dedup keys, same fixtures_df bet365 LEFT JOIN, same team_ratings.Date
type) so projection services can swap source with no other changes.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from app.source_database import get_source_connection, release_source_connection

logger = logging.getLogger("data_loader")


# Match DataCache.fixtures_df bet365 column expectations exactly. Downstream
# code reads e.g. `bet365_home_odds_decimal` — these renames preserve those.
_BET365_COLS_NEEDED = [
    "fixture_id",
    "home_win_odd", "draw_odd", "away_win_odd",
    "btts_yes_odd",
    "over_1_5_odd", "over_2_5_odd",
]
_BET365_RENAMES = {
    "home_win_odd": "bet365_home_odds_decimal",
    "draw_odd": "bet365_draw_odds_decimal",
    "away_win_odd": "bet365_away_odds_decimal",
    "btts_yes_odd": "bet365_btts_yes_odds_decimal",
    "over_1_5_odd": "over_1_5_odds_decimal",
    "over_2_5_odd": "over_2_5_odds_decimal",
}

# Fixture history depth. Matches the ~2-season rolling window used by the
# old SEASON_FILTER_FPS/FTS in fetch_all_data_service.py. Calendar-based
# (vs season-id based) is simpler and slightly more inclusive — promoted
# teams keep their lower-league history naturally.
_FIXTURE_LOOKBACK_YEARS = 2


class LeagueDataLoader:
    """Loads scoped data for projecting ONE competition.

    Lifecycle: instantiate per projection run, call `load()`, read
    attributes, drop. Reference tables are loaded fresh each run for now;
    Phase 3+ may add a session-level cache for Run All Leagues bursts.
    """

    def __init__(
        self,
        league_id: int,
        *,
        # For Euro comps the scope spans multiple domestic top tiers. Caller
        # passes the list explicitly. None = single-league scope.
        extra_league_ids: Optional[Sequence[int]] = None,
        league_weightings_xlsx_path: Optional[str] = None,
    ):
        self.league_id = int(league_id)
        self.extra_league_ids: List[int] = [int(x) for x in (extra_league_ids or [])]
        self.league_weightings_xlsx_path = league_weightings_xlsx_path

        # Resolved scope (populated by _resolve_scope / _resolve_fixture_ids)
        self.team_ids: List[int] = []
        self.player_ids: List[int] = []
        self.team_fixture_ids: List[int] = []     # team-based set
        self.player_fixture_ids: List[int] = []   # player-based set (cross-club + intl)
        self.fixture_ids: List[int] = []          # UNION — drives fixtures_df

        # Scoped tables (the 3 big ones)
        self.player_stats: Optional[pd.DataFrame] = None
        self.team_stats: Optional[pd.DataFrame] = None
        self.fixtures_df: Optional[pd.DataFrame] = None

        # Reference tables (loaded in full — small)
        self.standings: Optional[pd.DataFrame] = None
        self.seasons: Optional[pd.DataFrame] = None
        self.comps: Optional[pd.DataFrame] = None
        self.comp_teams: Optional[pd.DataFrame] = None
        self.teams: Optional[pd.DataFrame] = None
        self.b365_odds: Optional[pd.DataFrame] = None
        self.stats_types: Optional[pd.DataFrame] = None
        self.league_weightings: Optional[pd.DataFrame] = None
        self.projection_config: Optional[pd.DataFrame] = None
        self.promoted_team_ratings: Optional[pd.DataFrame] = None
        self.transfermarkt_team_mappings: Optional[pd.DataFrame] = None
        self.team_ratings: Optional[pd.DataFrame] = None

        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Resolve scope → resolve fixture IDs → load tables.

        Reference tables and scope come first. Then fixture-ID resolution
        (two queries: team-based + player-based, UNION'd). Then the three
        scoped data loaders. Sequential on a single connection — queries
        are fast (milliseconds) on indexed tables, parallelism would just
        add pool churn."""
        conn = await get_source_connection()
        try:
            await self._load_reference_tables(conn)
            await self._resolve_scope(conn)
            await self._resolve_fixture_ids(conn)
            await self._load_fixtures(conn)
            await self._load_team_stats(conn)
            await self._load_player_stats(conn)
            self._load_local_files()
            self._loaded = True
            logger.info(
                "LeagueDataLoader loaded for comp_id=%s: "
                "%d teams, %d players, %d team_fixtures, %d player_fixtures, "
                "%d fixtures (union), %d team_stat rows, %d player_stat rows",
                self.league_id, len(self.team_ids), len(self.player_ids),
                len(self.team_fixture_ids), len(self.player_fixture_ids),
                len(self.fixture_ids),
                0 if self.team_stats is None else len(self.team_stats),
                0 if self.player_stats is None else len(self.player_stats),
            )
        finally:
            release_source_connection(conn)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Scope resolution ──────────────────────────────────────────────────

    async def _resolve_scope(self, conn) -> None:
        """Compute team_ids and player_ids.

        Team scope = (target_league + extras + league_above + league_below)
        × current 2 seasons. Players = current_team_id IN team_ids.
        Fixture-ID resolution is a separate step (`_resolve_fixture_ids`).
        """
        comp_ids = self._all_scope_comp_ids()

        # Add league_above / league_below from competition_projection_config
        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(comp_ids))
            await cur.execute(
                f"""
                SELECT league_above_id, league_below_id
                FROM competition_projection_config
                WHERE competition_id IN ({placeholders})
                """,
                tuple(comp_ids),
            )
            rows = await cur.fetchall()
        for above, below in rows:
            if above is not None:
                comp_ids.add(int(above))
            if below is not None:
                comp_ids.add(int(below))

        # Resolve team_ids: top 2 seasons per competition in scope.
        # SELECT DISTINCT inside the window function so each season gets
        # ranked once (not once per team-row). Same fix as SEASON_FILTER_FPS
        # — the bug that silently capped to 1 season for months.
        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(comp_ids))
            await cur.execute(
                f"""
                SELECT DISTINCT cst.team_id
                FROM competition_season_teams cst
                JOIN (
                    SELECT competition_id, season_id FROM (
                        SELECT competition_id, season_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY competition_id
                                   ORDER BY season_id DESC
                               ) AS rn
                        FROM (
                            SELECT DISTINCT competition_id, season_id
                            FROM competition_season_teams
                            WHERE competition_id IN ({placeholders})
                        ) cs
                    ) ranked
                    WHERE rn <= 2
                ) recent
                  ON recent.competition_id = cst.competition_id
                 AND recent.season_id = cst.season_id
                """,
                tuple(comp_ids),
            )
            self.team_ids = sorted({int(r[0]) for r in await cur.fetchall()})

        if not self.team_ids:
            logger.warning(
                "LeagueDataLoader: scope resolution returned 0 teams for comp_id=%s",
                self.league_id,
            )
            self.player_ids = []
            return

        # Resolve player_ids via current squad membership.
        async with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(self.team_ids))
            await cur.execute(
                f"""
                SELECT id FROM players
                WHERE current_team_id IN ({placeholders})
                """,
                tuple(self.team_ids),
            )
            self.player_ids = sorted({int(r[0]) for r in await cur.fetchall()})

    def _all_scope_comp_ids(self) -> set:
        ids = {self.league_id}
        ids.update(self.extra_league_ids)
        return ids

    # ── Fixture-ID resolution (two sources, UNION'd) ──────────────────────

    async def _resolve_fixture_ids(self, conn) -> None:
        """Resolve team_fixture_ids + player_fixture_ids, store union.

        Team-based: any in-scope team's fixtures in last 2yr.
        Player-based: any fixture an in-scope player appeared in (last 2yr) —
        captures cross-club history (e.g. Bernal's Barca games while now at
        Palace) AND international fixtures (e.g. Saka for England)."""
        cutoff = datetime.utcnow() - timedelta(days=365 * _FIXTURE_LOOKBACK_YEARS)

        # Team-based set
        if self.team_ids:
            team_ph = ",".join(["%s"] * len(self.team_ids))
            sql = f"""
                SELECT id FROM fixtures
                WHERE (home_team_id IN ({team_ph}) OR away_team_id IN ({team_ph}))
                  AND kickoff_datetime >= %s
            """
            params = tuple(self.team_ids) + tuple(self.team_ids) + (cutoff,)
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                self.team_fixture_ids = sorted({int(r[0]) for r in await cur.fetchall()})
        else:
            self.team_fixture_ids = []

        # Player-based set (cross-club + international)
        if self.player_ids:
            player_ph = ",".join(["%s"] * len(self.player_ids))
            sql = f"""
                SELECT DISTINCT fps.fixture_id
                FROM fixture_player_stats fps
                JOIN fixtures f ON f.id = fps.fixture_id
                WHERE fps.player_id IN ({player_ph})
                  AND f.kickoff_datetime >= %s
            """
            params = tuple(self.player_ids) + (cutoff,)
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                self.player_fixture_ids = sorted({int(r[0]) for r in await cur.fetchall()})
        else:
            self.player_fixture_ids = []

        self.fixture_ids = sorted(
            set(self.team_fixture_ids) | set(self.player_fixture_ids)
        )

    # ── Scoped table loaders ──────────────────────────────────────────────

    async def _load_fixtures(self, conn) -> None:
        """All fixtures referenced by team_stats OR player_stats. UNION
        ensures downstream merges (player_stats.fixture_id ↔ fixtures.id)
        always resolve. bet365 LEFT JOIN preserves DataCache column names."""
        if not self.fixture_ids:
            self.fixtures_df = pd.DataFrame()
            return

        fix_ph = ",".join(["%s"] * len(self.fixture_ids))
        b365_select = ", ".join(
            f"b365.{col} AS {_BET365_RENAMES.get(col, col)}"
            for col in _BET365_COLS_NEEDED if col != "fixture_id"
        )
        sql = f"""
            SELECT f.*, {b365_select}
            FROM fixtures f
            LEFT JOIN bet365_fixture_odds b365 ON b365.fixture_id = f.id
            WHERE f.id IN ({fix_ph})
        """
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(self.fixture_ids))
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)

        if not df.empty:
            df.drop_duplicates(
                subset=["season_id", "home_team_id", "away_team_id", "kickoff_datetime"],
                inplace=True,
            )
        self.fixtures_df = df

    async def _load_team_stats(self, conn) -> None:
        """Both teams' stats for in-scope-team fixtures only.

        No `team_id` filter: get_opp_stats needs opposing-team rows. Scope
        deliberately limited to team_fixture_ids — cross-club fixtures
        (Bernal at Barcelona) excluded since no projection path reads
        team_stats from out-of-scope clubs. See loader_scope_rules.md."""
        if not self.team_fixture_ids:
            self.team_stats = pd.DataFrame()
            return

        fix_ph = ",".join(["%s"] * len(self.team_fixture_ids))
        sql = f"""
            SELECT * FROM fixture_team_stats
            WHERE fixture_id IN ({fix_ph})
        """
        async with conn.cursor() as cur:
            await cur.execute(sql, tuple(self.team_fixture_ids))
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df.drop_duplicates(
                subset=["fixture_id", "team_id", "stats_type_id"],
                inplace=True,
            )
            # `value` is stored as VARCHAR in source DB; CSV mode inferred
            # it as float via pandas. Coerce here so downstream arithmetic
            # (home + away, > 2.5, etc.) doesn't string-concatenate.
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
        self.team_stats = df

    async def _load_player_stats(self, conn) -> None:
        """Player stats across UNION fixture set so cross-club + international
        history is captured. player_id filter ensures we only load rows for
        currently-in-scope players, not e.g. Real Madrid players from a
        Barcelona-vs-Real fixture pulled in via Bernal's history."""
        if not self.player_ids or not self.fixture_ids:
            self.player_stats = pd.DataFrame()
            return

        player_ph = ",".join(["%s"] * len(self.player_ids))
        fix_ph = ",".join(["%s"] * len(self.fixture_ids))
        sql = f"""
            SELECT * FROM fixture_player_stats
            WHERE player_id IN ({player_ph})
              AND fixture_id IN ({fix_ph})
        """
        params = tuple(self.player_ids) + tuple(self.fixture_ids)
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df.drop_duplicates(
                subset=["fixture_id", "player_id", "stats_type_id"],
                inplace=True,
            )
            # Same VARCHAR→numeric coercion as team_stats. CSV mode got
            # float for free via pandas type inference.
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
        self.player_stats = df

    # ── Reference tables (small; bulk-loaded each run for now) ────────────

    async def _load_reference_tables(self, conn) -> None:
        """Load all small reference tables. Mirrors fetch_all_data_service.py
        query shapes so column names match DataCache exactly."""
        self.comps = await self._sql_to_df(conn, "SELECT * FROM competitions")
        self.seasons = await self._sql_to_df(conn, "SELECT * FROM seasons")
        self.comp_teams = await self._sql_to_df(
            conn, "SELECT * FROM competition_season_teams"
        )
        self.teams = await self._sql_to_df(conn, "SELECT * FROM teams")
        self.standings = await self._sql_to_df(conn, "SELECT * FROM standings")
        self.stats_types = await self._sql_to_df(conn, "SELECT * FROM stats_types")

        self.transfermarkt_team_mappings = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name, ttm.*
            FROM transfermarkt_team_mappings ttm
            JOIN competitions c ON c.id = ttm.competition_id
            WHERE ttm.to_name IS NOT NULL
            """,
        )
        self.promoted_team_ratings = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name, ptr.*
            FROM promoted_team_ratings ptr
            JOIN competitions c ON c.id = ptr.competition_id
            """,
        )
        self.projection_config = await self._sql_to_df(
            conn,
            """
            SELECT c.name AS league_name,
                   ca.name AS league_above_name,
                   cb.name AS league_below_name,
                   cpc.*
            FROM competition_projection_config cpc
            JOIN competitions c ON c.id = cpc.competition_id
            LEFT JOIN competitions ca ON ca.id = cpc.league_above_id
            LEFT JOIN competitions cb ON cb.id = cpc.league_below_id
            """,
        )

        # team_ratings — column rename + Date conversion identical to
        # DataCache so projection services see no diff.
        self.team_ratings = await self._sql_to_df(
            conn,
            """
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
            """,
        )
        if not self.team_ratings.empty:
            self.team_ratings["Date"] = pd.to_datetime(self.team_ratings["Date"]).dt.date
            # MySQL DECIMAL → Python decimal.Decimal via aiomysql; CSV mode
            # gets float for free. Coerce so arithmetic in get_ratings
            # (Attack/Defense weighting, Movement subtraction) works.
            for col in ("Attack", "Defense", "Overall", "Movement"):
                if col in self.team_ratings.columns:
                    self.team_ratings[col] = pd.to_numeric(
                        self.team_ratings[col], errors="coerce"
                    )

        # b365_odds — DataCache keeps it as an empty frame; nothing critical
        # reads it directly. Same here for parity.
        self.b365_odds = pd.DataFrame()

    def _load_local_files(self) -> None:
        """League Weightings.xlsx is the only non-DB reference. Loaded if
        path supplied; otherwise empty frame (CSV-mode parity)."""
        if self.league_weightings_xlsx_path and os.path.exists(self.league_weightings_xlsx_path):
            self.league_weightings = pd.read_excel(self.league_weightings_xlsx_path)
        else:
            self.league_weightings = pd.DataFrame()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    async def _sql_to_df(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)


# ── Shadow-mode capture (Phase 3) ────────────────────────────────────────

SHADOW_OUTPUT_DIR = Path("/tmp/loader_shadow")


async def capture_shadow_snapshot(
    league_name: str,
    league_id: int,
    *,
    extra_league_ids: Optional[Sequence[int]] = None,
    league_weightings_xlsx_path: Optional[str] = None,
) -> Optional[Path]:
    """Run LeagueDataLoader for one league and dump its DataFrames to parquet.

    Phase 4's diff tool will compare these against CSV-mode equivalents.
    Failures swallowed and logged — this must NEVER break the surrounding
    projection (it's purely observational while we validate parity).

    Returns the output dir path on success, None on failure.
    """
    try:
        SHADOW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        league_slug = league_name.replace(" ", "-").replace(".", "").lower()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = SHADOW_OUTPUT_DIR / f"{league_slug}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        loader = LeagueDataLoader(
            league_id,
            extra_league_ids=extra_league_ids,
            league_weightings_xlsx_path=league_weightings_xlsx_path,
        )
        await loader.load()

        # Dump scope diagnostics + the actual ID lists so Phase 4's diff
        # tool can apply the exact same filter to CSV-mode DataFrames.
        scope_meta = pd.DataFrame([{
            "league_name": league_name,
            "league_id": league_id,
            "n_team_ids": len(loader.team_ids),
            "n_player_ids": len(loader.player_ids),
            "n_team_fixture_ids": len(loader.team_fixture_ids),
            "n_player_fixture_ids": len(loader.player_fixture_ids),
            "n_fixture_ids_union": len(loader.fixture_ids),
            "captured_at": ts,
        }])
        scope_meta.to_parquet(out_dir / "_scope.parquet", index=False)

        pd.DataFrame({"team_id": loader.team_ids}).to_parquet(
            out_dir / "_team_ids.parquet", index=False)
        pd.DataFrame({"player_id": loader.player_ids}).to_parquet(
            out_dir / "_player_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.team_fixture_ids}).to_parquet(
            out_dir / "_team_fixture_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.player_fixture_ids}).to_parquet(
            out_dir / "_player_fixture_ids.parquet", index=False)
        pd.DataFrame({"fixture_id": loader.fixture_ids}).to_parquet(
            out_dir / "_fixture_ids.parquet", index=False)

        # The 3 scoped tables — these are what Phase 4 diffs against
        # equivalent CSV slices to prove parity.
        for attr in ("fixtures_df", "team_stats", "player_stats"):
            df = getattr(loader, attr)
            if df is not None and not df.empty:
                df.to_parquet(out_dir / f"{attr}.parquet", index=False)

        logger.info(
            "[%s] shadow snapshot captured at %s "
            "(teams=%d players=%d fixtures=%d team_stats=%d player_stats=%d)",
            league_name, out_dir,
            len(loader.team_ids), len(loader.player_ids),
            len(loader.fixture_ids),
            0 if loader.team_stats is None else len(loader.team_stats),
            0 if loader.player_stats is None else len(loader.player_stats),
        )
        return out_dir
    except Exception as e:
        logger.warning(
            "[%s] shadow snapshot FAILED (non-fatal): %s",
            league_name, e, exc_info=True,
        )
        return None
