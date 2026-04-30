import logging
import os
import pandas as pd

logger = logging.getLogger("data_cache")


class DataCache:
    """
    Singleton cache for shared source data (CSV files).
    Loaded once per projection run, shared across all leagues.
    Eliminates redundant file reads when projecting multiple leagues.
    """

    def __init__(self):
        self.player_stats = None
        self.team_stats = None
        self.fixtures_df = None
        self.standings = None
        self.seasons = None
        self.comps = None
        self.comp_teams = None
        self.teams = None
        self.b365_odds = None
        self.stats_types = None
        self.league_weightings = None
        self.projection_config = None
        self.promoted_team_ratings = None
        self.transfermarkt_team_mappings = None
        self.team_ratings = None
        self._loaded = False

    def load(self, data_folder_path: str):
        """Load all shared source CSV files into memory."""
        logger.info("DataCache: loading source CSV files into memory...")
        path = data_folder_path

        self.player_stats = pd.read_csv(os.path.join(path, "fixture_player_stats.csv"))
        self.player_stats.drop_duplicates(
            subset=["fixture_id", "player_id", "stats_type_id"], inplace=True
        )

        self.team_stats = pd.read_csv(os.path.join(path, "fixture_team_stats.csv"))
        self.team_stats.drop_duplicates(
            subset=["fixture_id", "team_id", "stats_type_id"], inplace=True
        )

        self.standings = pd.read_csv(os.path.join(path, "standings.csv"))
        self.seasons = pd.read_csv(os.path.join(path, "seasons.csv"))
        self.comps = pd.read_csv(os.path.join(path, "competitions.csv"))
        self.comp_teams = pd.read_csv(os.path.join(path, "competition_season_teams.csv"))
        self.teams = pd.read_csv(os.path.join(path, "teams.csv"))

        self.fixtures_df = pd.read_csv(os.path.join(path, "fixtures.csv"))
        self.fixtures_df.drop_duplicates(
            subset=["season_id", "home_team_id", "away_team_id", "kickoff_datetime"],
            inplace=True,
        )

        # Bet365 odds moved out of the `fixtures` table into their own
        # `bet365_fixture_odds` table in the 2026-04-23 unification. The
        # 96+ references in projection_service / all_teams / euro_comp
        # read legacy names like `bet365_home_odds_decimal`, so we LEFT
        # JOIN by fixture_id and rename back to those names. Downstream
        # code needs no changes.
        #
        # Columns that exist on bet365_fixture_odds but aren't consumed
        # (fractionals, odds_ids, market_fids, btts_no_odd) are dropped
        # to keep the fixtures DataFrame lean — only the decimal cols
        # used by probability math survive.
        b365_cols_needed = [
            "fixture_id",
            "home_win_odd", "draw_odd", "away_win_odd",
            "btts_yes_odd",
            "over_1_5_odd", "over_2_5_odd",
        ]
        b365_renames = {
            "home_win_odd": "bet365_home_odds_decimal",
            "draw_odd": "bet365_draw_odds_decimal",
            "away_win_odd": "bet365_away_odds_decimal",
            "btts_yes_odd": "bet365_btts_yes_odds_decimal",
            "over_1_5_odd": "over_1_5_odds_decimal",
            "over_2_5_odd": "over_2_5_odds_decimal",
        }
        b365_path = os.path.join(path, "bet365_fixture_odds.csv")
        if os.path.exists(b365_path):
            b365 = pd.read_csv(b365_path, usecols=b365_cols_needed)
            b365 = b365.drop_duplicates(subset=["fixture_id"], keep="last")
            b365 = b365.rename(columns=b365_renames)
            # Drop any stale renamed columns already on fixtures_df (from
            # the pre-migration schema) so the merge populates them fresh.
            stale = [c for c in b365_renames.values() if c in self.fixtures_df.columns]
            if stale:
                self.fixtures_df = self.fixtures_df.drop(columns=stale)
            self.fixtures_df = self.fixtures_df.merge(
                b365, left_on="id", right_on="fixture_id", how="left"
            ).drop(columns=["fixture_id"])
            logger.info(
                f"DataCache: loaded bet365_fixture_odds.csv ({len(b365)} rows) "
                "and merged onto fixtures_df"
            )
        else:
            # First run after this change — fetch hasn't exported the CSV
            # yet. Populate the expected columns as NaN so downstream
            # probability calcs (1/odd) yield NaN cleanly rather than
            # KeyError-ing.
            for col in b365_renames.values():
                self.fixtures_df[col] = pd.NA
            logger.warning(
                "DataCache: bet365_fixture_odds.csv not found — "
                "bet365 decimal columns will be NaN until next fetch"
            )
        # Retain raw long-format reference used by nothing critical but
        # referenced by old code paths; harmless empty DataFrame if the
        # legacy CSV is no longer fetched.
        self.b365_odds = pd.DataFrame()

        self.stats_types = pd.read_csv(os.path.join(path, "stats_types.csv"))

        # League Weightings xlsx is dead post-2026-04-22 migration to
        # competition_projection_config (DataCache only used in off-mode
        # legacy path — slated for Phase 7 deletion 2026-05-05). Tolerate
        # missing file so the cache still loads if someone toggles off-mode.
        _lw_path = os.path.join(path, "League Weightings.xlsx")
        self.league_weightings = pd.read_excel(_lw_path) if os.path.exists(_lw_path) else pd.DataFrame()

        # DB-driven projection config (from competition_projection_config table).
        # Falls back gracefully if the CSV doesn't exist yet (first fetch hasn't run).
        config_path = os.path.join(path, "projection_config.csv")
        if os.path.exists(config_path):
            self.projection_config = pd.read_csv(config_path)
            logger.info(f"DataCache: loaded projection_config.csv ({len(self.projection_config)} rows)")
        else:
            self.projection_config = pd.DataFrame()
            logger.info("DataCache: projection_config.csv not found — using League Weightings.xlsx fallback")

        # DB-driven Transfermarkt team name mappings (sole source of truth).
        tm_path = os.path.join(path, "transfermarkt_team_mappings.csv")
        if os.path.exists(tm_path):
            self.transfermarkt_team_mappings = pd.read_csv(tm_path)
            logger.info(f"DataCache: loaded transfermarkt_team_mappings.csv ({len(self.transfermarkt_team_mappings)} rows)")
        else:
            self.transfermarkt_team_mappings = pd.DataFrame()
            logger.warning("DataCache: transfermarkt_team_mappings.csv not found — MV adjustment will run unmapped")

        # DB-driven promoted team ratings (replaces per-league xlsx files).
        promoted_path = os.path.join(path, "promoted_team_ratings.csv")
        if os.path.exists(promoted_path):
            self.promoted_team_ratings = pd.read_csv(promoted_path)
            logger.info(f"DataCache: loaded promoted_team_ratings.csv ({len(self.promoted_team_ratings)} rows)")
        else:
            self.promoted_team_ratings = pd.DataFrame()
            logger.info("DataCache: promoted_team_ratings.csv not found — using xlsx fallback")

        # DB-driven team ratings (replaces Team Ratings.parquet / Team Ratings.xlsx).
        # Single source of truth shared by all 4 projection services (domestic,
        # all-teams, euro-comp, premier-league-legacy).
        tr_path = os.path.join(path, "team_ratings.csv")
        if os.path.exists(tr_path):
            self.team_ratings = pd.read_csv(tr_path)
            self.team_ratings['Date'] = pd.to_datetime(self.team_ratings['Date']).dt.date
            logger.info(f"DataCache: loaded team_ratings.csv ({len(self.team_ratings)} rows)")
        else:
            self.team_ratings = pd.DataFrame(columns=['League', 'Team', 'Date', 'Attack', 'Defense', 'Overall', 'Movement', 'Inverse', 'team_id', 'competition_id'])
            logger.warning("DataCache: team_ratings.csv not found — all ratings will compute as first-entries with Movement=NULL")

        self._loaded = True
        logger.info("DataCache: all source data loaded successfully.")

    def invalidate(self):
        """Force reload on next projection run (call after /fetch-data)."""
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded
