import asyncio
import logging
import time
from scipy.stats import poisson
import warnings

from app.repository.fixtures_repo import insert_fixtures_async
from app.repository.team_repo import insert_teams_async
from app.repository.player_stat_repo import insert_players_stats_async
from app.repository.player_repo import insert_player_async, get_players_from_league
from app.data_loader import LeagueDataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from .statz_functions import *
from pathlib import Path
import os

logger = logging.getLogger("projection")


class EuroCompProjectionService:
    CURRENT_DIR = Path(__file__).resolve().parent
    APP_DIR = CURRENT_DIR.parent

    DATA_FOLDER_PATH = APP_DIR / "data"
    MODEL_FILE_PATH = APP_DIR / "model-builds"
    SAVE_FILE_PATH = APP_DIR / "projection-outputs"
    DAYS = 5

    EURO_COMPS = ['Champions League', 'Europa League', 'Conference League', 'Europa Conference League']

    # Top-5 league competition_ids — used as the baseline for the
    # Poisson goal averages in euro-comp projections (avg_home_goals /
    # avg_away_goals). Averaging across all 15 LEAGUE_COUNTRY_DICT
    # entries pulled the baseline toward smaller-league scoring rates
    # (Eliteserien/Allsvenskan etc.) that don't reflect realistic
    # scoring at the euro-comp level. Decision: 2026-05-27.
    TOP_5_LEAGUE_IDS = [8, 564, 384, 82, 301]  # PL, La Liga, Serie A, Bundesliga, Ligue 1

    # Domestic top tiers in scope for Euro-comp cross-league ratings.
    # Every league here MUST have:
    #   - team_ratings rows (so the rescale step has data to anchor on)
    #   - a competition_projection_config row (so the cross-league
    #     weighting / transfermarkt-code lookup resolves)
    # Confirmed 2026-05-21 for all 15 entries below.
    LEAGUE_COUNTRY_DICT = {
        'Premier League':       'England',
        'La Liga':               'Spain',
        'Serie A':               'Italy',
        'Bundesliga':            'Germany',
        'Ligue 1':               'France',
        'Eredivisie':            'Netherlands',
        'Liga Portugal':         'Portugal',
        'Scottish Premiership':  'Scotland',
        # added 2026-05-21 — every one has clubs in current UCL/UEL/UECL
        # and full team_ratings + projection_config coverage.
        'Austrian Bundesliga':   'Austria',
        'Belgian Pro League':    'Belgium',
        'Eliteserien':           'Norway',
        'Super League':          'Switzerland',
        'Super Lig':             'Turkey',
        'Superliga':             'Denmark',
        'Allsvenskan':           'Sweden',
    }

    @staticmethod
    def _read_df(path_no_ext: str) -> pd.DataFrame:
        parquet_path = f"{path_no_ext}.parquet"
        excel_path = f"{path_no_ext}.xlsx"
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(excel_path):
            return pd.read_excel(excel_path)
        raise FileNotFoundError(f"No data file found at {parquet_path} or {excel_path}")

    @staticmethod
    def is_euro_comp(league: str) -> bool:
        return league in EuroCompProjectionService.EURO_COMPS

    async def projections(self, league_request):
        league = league_request.league or 'Champions League'
        _start_time = time.time()
        logger.info(f'[{league}] START euro comp projections')

        data_folder_path = EuroCompProjectionService.DATA_FOLDER_PATH
        model_file_path = EuroCompProjectionService.MODEL_FILE_PATH
        save_file_path = EuroCompProjectionService.SAVE_FILE_PATH

        date_from = pd.to_datetime('today')
        date_to = date_from + pd.DateOffset(days=EuroCompProjectionService.DAYS)
        odds_weight = 0.5

        # Euro comp scope spans the comp itself + 8 domestic top-tiers
        # (LEAGUE_COUNTRY_DICT). Resolve IDs up-front via direct DB queries,
        # then pass them to LeagueDataLoader so the team scope covers all
        # relevant clubs.
        from app.services.projection_service import ProjectionService
        comp_id_for_load = await ProjectionService._resolve_league_id_db(league)
        domestic_ids = []
        for dom_league in EuroCompProjectionService.LEAGUE_COUNTRY_DICT.keys():
            domestic_ids.append(await ProjectionService._resolve_league_id_db(dom_league))
        league_weightings_path = os.path.join(data_folder_path, "League Weightings.xlsx")
        _loader = LeagueDataLoader(
            comp_id_for_load,
            extra_league_ids=domestic_ids,
            league_weightings_xlsx_path=league_weightings_path,
        )
        await _loader.load()
        source = _loader
        logger.info(f"[{league}] Data source: LeagueDataLoader (+8 domestic comps)")
        ProjectionService._current_source = source
        # Loader is per-call so mutation safety isn't a concern. _maybe_copy
        # kept as a no-op shim so call sites don't churn.
        def _maybe_copy(df):
            return df

        player_stats = _maybe_copy(source.player_stats)
        team_stats = _maybe_copy(source.team_stats)
        standings = _maybe_copy(source.standings)
        seasons = source.seasons
        comps = source.comps
        comp_teams = source.comp_teams
        teams = source.teams
        fixtures_df = _maybe_copy(source.fixtures_df)
        stats_types = source.stats_types

        # Players from LeagueDataLoader (DB-direct, scoped to comp + 8 domestic
        # comps). display_name already stripped upstream.
        players = source.players
        logger.info(f"[{league}] Loaded {len(players)} players from DB-loader")

        # Ratings Dataset — DB-sourced (cache or per-league loader).
        all_team_ratings = _maybe_copy(source.team_ratings)

        # League Weightings (for domestic rating calculations) — loader
        # populates this from competition_projection_config; xlsx fallback
        # only fires if the DB table is unexpectedly empty.
        _lw_xlsx = os.path.join(data_folder_path, "League Weightings.xlsx")
        league_weightings_df = source.league_weightings if (source.league_weightings is not None and not source.league_weightings.empty) else (pd.read_excel(_lw_xlsx) if os.path.exists(_lw_xlsx) else pd.DataFrame())

        # UEFA Coefficients — DB-sourced from competitions.uefa_coefficient_index
        # (backfilled 2026-04-22). Built into the same shape the legacy xlsx
        # had (League / Coefficient Index columns) so downstream lookups work
        # unchanged. xlsx is no longer read.
        # Cast to float: aiomysql reads MySQL DECIMAL → decimal.Decimal,
        # which doesn't divide with float (downstream `team_projections /
        # (diff + 1)` raises TypeError). xlsx path got float for free.
        uefa_coef = comps[['name', 'uefa_coefficient_index']].rename(
            columns={'name': 'League', 'uefa_coefficient_index': 'Coefficient Index'}
        ).dropna(subset=['Coefficient Index']).reset_index(drop=True)
        uefa_coef['Coefficient Index'] = pd.to_numeric(uefa_coef['Coefficient Index'], errors='coerce')

        comp_id = get_league_id(league, comps)
        league_ids = [get_league_id(l, comps) for l in EuroCompProjectionService.LEAGUE_COUNTRY_DICT.keys()]

        # Shadow capture deliberately SKIPPED for euro comps. Scope spans
        # the comp + 8 domestic top tiers → 10k+ players, 8M+ player_stat
        # rows in the loader DataFrame. Stacked on top of the already-loaded
        # 4GB DataCache it OOM-kills the gunicorn worker mid-run. Domestic
        # leagues still capture (smaller scope, no OOM risk). Euro-comp
        # parity is covered by the standalone test_on_mode.py CL smoke
        # test from Phase 5d.

        fixtures = fixtures_df[fixtures_df['competition_id'] == comp_id]
        current_season_id = get_season_id(comp_id, seasons, False)
        if current_season_id is None:
            # Typical for euro comps right after the final — Sportmonks
            # hasn't created the next season yet. Skip cleanly.
            raise RuntimeError(
                f"no current season in seasons table for competition_id={comp_id} — skipping"
            )
        stat_list = get_stat_list()

        logger.info(f'[{league}] Building cross-league ratings...')

        # ── Build cross-league ratings using UEFA coefficients ──

        def rescale_to_range(series, new_min=0.5, new_max=2.0):
            old_min = series.min()
            old_max = series.max()
            return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)

        ratings_df = pd.DataFrame()

        # CACHED-RATINGS PATH (2026-05-27): inner-league ratings are read
        # from the team_ratings DB table instead of recomputed per league.
        # The domestic projection cron writes fresh, post-MV, post-dial,
        # rescaled-to-mean-100 rows nightly — recomputing here was ~25s
        # per league × 15 leagues = ~6 min of wasted work every euro
        # comp run. Now we just pick the latest row per (competition_id,
        # team_id), apply the UEFA coefficient on top, and concat.
        #
        # Things that USED to happen in this loop and now don't, because
        # they're already baked into team_ratings:
        #   - get_ratings() weighted compute
        #   - promoted-team blend (handled by domestic projection)
        #   - market-value adjustment
        #   - team dials apply
        #   - per-league rescale-to-mean-100
        latest_ratings_by_id = {}
        if all_team_ratings is not None and not all_team_ratings.empty:
            # Pick latest row per (competition_id, team_id). Frame includes
            # all leagues so we filter as we iterate.
            sorted_tr = all_team_ratings.sort_values('Date', ascending=False)
            latest_ratings_by_id = sorted_tr.drop_duplicates(
                subset=['competition_id', 'team_id'], keep='first'
            )

        for league_name, country in EuroCompProjectionService.LEAGUE_COUNTRY_DICT.items():
            league_id = get_league_id(league_name, comps)

            if isinstance(latest_ratings_by_id, pd.DataFrame):
                league_rows = latest_ratings_by_id[latest_ratings_by_id['competition_id'] == league_id]
            else:
                league_rows = pd.DataFrame()
            if league_rows.empty:
                logger.warning(f"[{league}] {league_name}: no team_ratings rows in DB — skipping (run the domestic projection first to seed it)")
                continue

            ratings = league_rows[['Team', 'Attack', 'Defense', 'Overall',
                                   'Attack_xG', 'Defense_xG', 'Overall_xG']].copy()
            # Defensive — strip whitespace on team names so cross-league
            # joins downstream match cleanly (transfermarkt mapping uses
            # exact strings).
            ratings['Team'] = ratings['Team'].astype(str).str.strip()
            logger.info(f"[{league}] {league_name}: loaded {len(ratings)} teams from team_ratings cache")

            # Apply UEFA coefficient — same scaling applies to both the
            # indexed and the xG/game columns so euro-comp rankings stay
            # cross-league comparable.
            # Try DB first (competitions.uefa_coefficient_index, added
            # 2026-04-22 migration), fall back to League Coefficients.xlsx
            # for any league not yet backfilled in DB.
            comp_row = comps[comps['id'] == league_id]
            db_coef = comp_row['uefa_coefficient_index'].iloc[0] if (
                not comp_row.empty and 'uefa_coefficient_index' in comps.columns
                and pd.notna(comp_row['uefa_coefficient_index'].iloc[0])
            ) else None
            if db_coef is not None:
                coef = float(db_coef)
            else:
                xlsx_match = uefa_coef[uefa_coef['League'] == league_name]['Coefficient Index']
                if xlsx_match.empty:
                    logger.warning(f"[{league}] No UEFA coefficient for {league_name} in DB or xlsx — defaulting to 1.0")
                    coef = 1.0
                else:
                    coef = xlsx_match.values[0]
            ratings['League'] = league_name
            ratings['coef'] = coef
            ratings['Attack'] *= coef
            ratings['Defense'] /= coef
            ratings['Attack_xG'] *= coef
            ratings['Defense_xG'] /= coef
            ratings['Overall_xG'] = ratings['Attack_xG'] - ratings['Defense_xG']
            ratings_df = pd.concat([ratings_df, ratings], ignore_index=True)

        ratings_df['Overall'] = ratings_df['Attack'] - ratings_df['Defense']
        ratings_df.sort_values(by='Overall', ascending=False, inplace=True)
        ratings = ratings_df.copy()

        logger.info(f'[{league}] Ratings built for {len(ratings)} teams across {len(EuroCompProjectionService.LEAGUE_COUNTRY_DICT)} leagues')

        # Save UEFA-coefficient-adjusted ratings to the team_ratings DB table
        # under the euro comp's competition_id. This replaces the previous
        # no-op (euro_comp_service never wrote ratings → Champions League
        # and Europa League hadn't been updated since Mar 20).
        from app.repository.team_ratings_repo import insert_team_ratings_async
        await insert_team_ratings_async(
            ratings[['Team', 'Attack', 'Defense', 'Overall', 'Attack_xG', 'Defense_xG', 'Overall_xG']].copy(),
            league, comp_id, teams,
            comp_teams=comp_teams,
            # Ratings are written under the euro comp's id but cover teams
            # from all 8 domestic top tiers (Barcelona/Bayern/PSG aren't
            # in EL's competition_season_teams pool, so a comp_id-scoped
            # lookup misses them — ~11 fallback warnings/run pre-fix).
            lookup_competition_ids=league_ids + [comp_id],
        )

        # ── Fixture projections ──

        fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
        next_fix = fixtures[(fixtures['kickoff_datetime'] >= date_from) & (fixtures['kickoff_datetime'] <= date_to)]
        if hasattr(league_request, 'fixture_ids') and league_request.fixture_ids:
            next_fix = next_fix[next_fix['id'].isin(league_request.fixture_ids)]
            logger.info(f'[{league}] Filtered to {len(next_fix)} fixtures by IDs')
        # Carry `neutral_venue` through — read at projection time by
        # make_round_goal_prediction + get_team_round_predictions to
        # disable home-advantage bias for finals at neutral grounds.
        # Defaults to False if the source DF doesn't have the column
        # yet (legacy fixtures pre-migration 2026-05-27).
        _has_neutral = 'neutral_venue' in next_fix.columns
        _cols = ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id',
                 'bet365_home_odds_decimal', 'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']
        if _has_neutral:
            _cols.append('neutral_venue')
        next_fix = next_fix[_cols]
        if not _has_neutral:
            next_fix['neutral_venue'] = False
        next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
        next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
        next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
        next_fix = drop_placeholder_fixtures(next_fix, league)
        next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
        next_fix.reset_index(drop=True, inplace=True)

        # Drop fixtures where teams don't have ratings
        drop_indices = []
        for i in range(len(next_fix)):
            home_team = next_fix['home_team'][i]
            away_team = next_fix['away_team'][i]
            if home_team not in ratings['Team'].values or away_team not in ratings['Team'].values:
                logger.warning(f'[{league}] Skipping fixture: {home_team} vs {away_team} — team not in ratings')
                drop_indices.append(i)
        next_fix = next_fix.drop(drop_indices).reset_index(drop=True)

        if len(next_fix) == 0:
            logger.info(f"[{league}] No fixtures to project"); logger.info(f"[{league}] DONE euro comp projections (nothing to do)")
            return

        logger.info(f'[{league}] Projecting {len(next_fix)} fixtures...')

        # Goal averages from the top-5 leagues only (PL, La Liga, Serie A,
        # Bundesliga, Ligue 1). All 15 LEAGUE_COUNTRY_DICT entries get
        # ratings, but smaller leagues' goal rates don't reflect realistic
        # euro-comp scoring — using them in the Poisson baseline pulled
        # PSG/Arsenal/etc.'s projections toward Nordic / Austrian averages.
        # NaN-filter keeps the math safe for between-season leagues whose
        # team_stats might return None.
        _goal_avg_pool = [lid for lid in EuroCompProjectionService.TOP_5_LEAGUE_IDS if lid in league_ids]
        avg_home_goals_list = [get_home_goal_avg(lid, team_stats, fixtures_df, stats_types) for lid in _goal_avg_pool]
        avg_away_goals_list = [get_away_goal_avg(lid, team_stats, fixtures_df, stats_types) for lid in _goal_avg_pool]
        avg_home_goals_list = [v for v in avg_home_goals_list if v is not None and not np.isnan(v)]
        avg_away_goals_list = [v for v in avg_away_goals_list if v is not None and not np.isnan(v)]
        avg_home_goals = np.mean(avg_home_goals_list) if avg_home_goals_list else 1.5
        avg_away_goals = np.mean(avg_away_goals_list) if avg_away_goals_list else 1.2
        logger.info(f"[{league}] Goal averages: avg_home={avg_home_goals:.3f} avg_away={avg_away_goals:.3f} (from {len(avg_home_goals_list)} top-5 leagues)")

        score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)

        boost = 1.1
        score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []

        for i in range(len(score_preds)):
            bookie_margin = 1 + (score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[i, 'Away Odds %'] - 100) / 100
            score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
            score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)

            home_goals = score_preds['Home Goals'][i]
            away_goals = score_preds['Away Goals'][i]

            if pd.isna(score_preds['Home Odds %'][i]) == False:
                home_win_prob, draw_prob, away_win_prob = get_result_probs(home_goals, away_goals, boost)
                adjusted_home_win_prob = home_win_prob + ((score_preds['Home Odds %'][i] - home_win_prob) * odds_weight)
                adjusted_draw_prob = draw_prob + ((score_preds['Draw Odds %'][i] - draw_prob) * odds_weight)
                adjusted_away_win_prob = away_win_prob + ((score_preds['Away Odds %'][i] - away_win_prob) * odds_weight)
                new_home_goals, new_away_goals = find_inputs_for_probs(home_goals, away_goals, adjusted_home_win_prob,
                                                                       adjusted_draw_prob, adjusted_away_win_prob, boost)
                score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
                score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)
            else:
                new_home_goals = home_goals
                new_away_goals = away_goals
                adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = get_result_probs(home_goals, away_goals, boost)
                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)

            x = np.arange(0, 9)
            y = np.arange(0, 9)
            X, Y = np.meshgrid(x, y)
            Z = poisson.pmf(X, new_home_goals) * poisson.pmf(Y, new_away_goals)

            home_win.append(f"{adjusted_home_win_prob:.2f}%")
            draw.append(f"{adjusted_draw_prob:.2f}%")
            away_win.append(f"{adjusted_away_win_prob:.2f}%")
            home_clean.append(f"{home_clean_sheet * 100:.2f}%")
            away_clean.append(f"{away_clean_sheet * 100:.2f}%")
            over_1_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1]) * 100
            over_2_goals = (1 - Z[0, 0] - Z[1, 0] - Z[0, 1] - Z[2, 0] - Z[0, 2] - Z[1, 1]) * 100
            both_teams_score_prob = (1 - Z[0, :].sum() - Z[:, 0].sum() + Z[0, 0]) * 100
            over_1.append(f"{over_1_goals:.2f}%")
            over_2.append(f"{over_2_goals:.2f}%")
            btts.append(f"{both_teams_score_prob:.2f}%")

        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
        score_preds['Home Win %'] = home_win
        score_preds['Draw %'] = draw
        score_preds['Away Win %'] = away_win
        score_preds['Home Clean Sheet %'] = home_clean
        score_preds['Away Clean Sheet %'] = away_clean
        score_preds['Over 1.5 Goals %'] = over_1
        score_preds['Over 2.5 Goals %'] = over_2
        score_preds['Both Teams Score %'] = btts
        score_preds['Home Goals'] = score_preds['Home Goals'].round(2)
        score_preds['Away Goals'] = score_preds['Away Goals'].round(2)

        logger.info(f'[{league}] Fixture projections complete')

        # Save fixture projections
        await insert_fixtures_async(score_preds, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        # ── Team projections ──

        logger.info(f'[{league}] Building team projections...')

        # Load pre-trained models (no retraining)
        models = load_all_models(stat_list, str(model_file_path))

        team_projections = get_team_round_predictions(
            next_fix, stat_list, fixtures_df, team_stats, teams, stats_types, models,
            ratings=ratings, comp_id=league_ids, games=50,
            # comp_teams must cover all 8 domestic leagues (league_ids), not
            # just the euro comp itself — downstream get_team_id scopes via
            # .isin(league_ids) and would miss every PSG/Bayern/Atletico/etc
            # if comp_teams was filtered to comp_id alone (~880 fallback
            # warnings per nightly run pre-fix).
            comp_teams=comp_teams[comp_teams['competition_id'].isin(league_ids + [comp_id])],
        )

        team_projections.drop(
            columns=['Team ' + stat + ' History' for stat in stat_list] +
                    ['Opponent ' + stat + ' History Against' for stat in stat_list],
            inplace=True
        )

        # Adjust team stats by UEFA coefficient difference
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            team_league = ratings.loc[ratings['Team'] == team, 'League'].values
            if len(team_league) == 0:
                continue
            team_league = team_league[0]
            league_rating = uefa_coef[uefa_coef['League'] == team_league]['Coefficient Index'].values[0]

            opponent = team_projections['Opponent'].iloc[i]
            opp_league = ratings.loc[ratings['Team'] == opponent, 'League'].values
            if len(opp_league) == 0:
                continue
            opp_league = opp_league[0]
            opp_league_rating = uefa_coef[uefa_coef['League'] == opp_league]['Coefficient Index'].values[0]

            diff = (opp_league_rating - league_rating)
            team_projections.at[team_projections.index[i], 'Shots Total'] = (team_projections['Shots Total'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Shots On Target'] = (team_projections['Shots On Target'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Corners'] = (team_projections['Corners'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Passes'] = (team_projections['Passes'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Successful Passes'] = (team_projections['Successful Passes'].iloc[i] / (diff + 1)).round(2)
            team_projections.at[team_projections.index[i], 'Total Crosses'] = (team_projections['Total Crosses'].iloc[i] / (diff + 1)).round(2)

        # Bake goals into shots projections
        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == comp_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total', stats_types)].copy()
        league_shots['Date'] = league_shots['fixture_id'].map(fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(league_shots['Date'])).dt.days // 7
        league_shots['Weight'] = 0.9 ** (league_shots['Weeks Since Kickoff'] - 5)
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots['value']
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots['Weight'].sum()

        league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target', stats_types)].copy()
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(league_shots_on_target['Date'])).dt.days // 7
        league_shots_on_target['Weight'] = 0.9 ** (league_shots_on_target['Weeks Since Kickoff'] - 5)
        league_shots_on_target.loc[league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target['value']
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target['Weight'].sum()

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            try:
                team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
            except:
                team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
            goals.append(team_pred)
            assists.append((team_pred * 0.82).round(2))
            projected_shots = team_projections['Shots Total'].iloc[i]
            projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

            adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                team_pred, projected_shots, projected_shots_on_target,
                avg_shots_per_goal, avg_shots_on_target_per_goal
            )
            team_projections.at[team_projections.index[i], 'Shots Total'] = adjusted_shots
            team_projections.at[team_projections.index[i], 'Shots On Target'] = adjusted_shots_on_target

        team_projections['Goals'] = goals
        team_projections['Assists'] = assists

        saves = []
        for i in range(len(team_projections)):
            opp = team_projections['Opponent'].iloc[i]
            try:
                opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
            except:
                opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
            saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)

        team_projections['Saves'] = saves
        team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists', 'Key Passes'] +
            stat_list + ['Fouls Drawn', 'Saves']
        ]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)

        # Save team projections
        team_projections_save = team_projections.copy()
        team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1, inplace=True)
        team_projections_save = team_projections_save.round(2)
        await insert_teams_async(team_projections_save, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        logger.info(f'[{league}] Team projections complete')

        # ── Player projections ──

        logger.info(f'[{league}] Building player projections...')

        pl_projections = distribute_team_predictions_to_players(
            player_stats, team_stats, team_projections, stats_types, fixtures_df, players, teams, comps, 0.97,
            competition_id=comp_id, comp_teams=comp_teams,
        )

        player_pos = []
        player_saves = []
        for player, team in pl_projections[['Player', 'Team']].values:
            pos = get_player_position(player, team, players, teams, comp_id, comp_teams)
            if pos == 'GK':
                player_saves.append(team_projections[team_projections['Team'] == team]['Saves'].values[0])
            else:
                player_saves.append(0)
            player_pos.append(pos)
        pl_projections['Position'] = player_pos
        pl_projections['Saves'] = player_saves

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Goals', 'Assists', 'Shots Total', 'Shots On Target', 'Key Passes', 'Passes', 'Accurate Passes',
             'Interceptions', 'Tackles', 'Total Crosses', 'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']
        ]
        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # Predict starters
        pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])].copy()
        pred_starters = pred_starters[pred_starters['stats_type_id'] == 11]

        start = []
        for i in range(len(pl_projections)):
            team = pl_projections['Team'].iloc[i]
            player_name = pl_projections['Player'].iloc[i]
            try:
                player_id = get_player_id(player_name, players, team, teams, comp_id, comp_teams)
            except:
                start.append('No')
                continue
            team_starters = pred_starters[pred_starters['team_id'] == get_team_id(team, teams, comp_id, comp_teams)]
            if player_id in team_starters['player_id'].values:
                start.append('Yes')
            else:
                start.append('No')
        pl_projections['Start?'] = start

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'Start?', 'Goals', 'Assists', 'Shots Total', 'Shots On Target', 'Key Passes', 'Passes',
             'Accurate Passes', 'Interceptions', 'Tackles', 'Total Crosses', 'Yellow Cards', 'Offsides',
             'Fouls', 'Fouls Drawn', 'Saves']
        ]
        pl_projections = pl_projections.round(2)
        pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)

        # Save player projections
        await insert_player_async(pl_projections, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        logger.info(f'[{league}] Player projections complete')

        # ── Player stat props ──

        logger.info(f'[{league}] Building player stat props...')

        pl_projections.rename(columns={'Fouls': 'Fouls Committed'}, inplace=True)

        perc_stats = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn',
                      'Goals', 'Tackles', 'Shots Total', 'Offsides']
        lines = [1, 2, 3]

        player_stat_probs = get_poisson_probs(pl_projections, perc_stats, lines)
        # Note: 'Yellowcards' is renamed to 'Yellow Cards' upstream of this point.
        if 'Yellow Cards' in pl_projections.columns:
            yellow_probs = get_poisson_probs(pl_projections, ['Yellow Cards'], [1])
            player_stat_probs = pd.concat([player_stat_probs, yellow_probs], ignore_index=True)
        player_stat_probs = player_stat_probs.round(2)

        # Save player stat props
        await insert_players_stats_async(player_stat_probs, teams=teams, competition_id=comp_id, comp_teams=comp_teams)

        _elapsed = round(time.time() - _start_time, 1)
        logger.info(f'[{league}] DONE euro comp projections in {_elapsed}s')
