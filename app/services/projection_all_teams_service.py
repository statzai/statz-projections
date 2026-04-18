import logging
import time
from app.services.projection_service import ProjectionService
from app.services.euro_comp_projection_service import EuroCompProjectionService
from app.models.requests.league_request import LeagueRequest
from scipy.stats import poisson
import warnings
from app.repository.fixtures_repo import insert_fixtures_async
from app.repository.team_repo import insert_teams_async
from app.repository.predicted_table_repo import insert_predicted_table_async
from app.repository.player_stat_repo import insert_players_stats_async
from app.repository.player_repo import insert_player_async
from app.repository.fpl_repo import insert_fpl_projections_async
from app.repository.fanteam_repo import insert_fanteam_projections_async
from app.repository.opta_repo import insert_opta_projections_async
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from .statz_functions import *
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
from fastapi import Response


logger = logging.getLogger("projection")

class ProjectionAllTeams:
    DEFAULT_LEAGUES = [
        "Championship",
        "Premier League",
        "La Liga",
        "Serie A",
        "Campeonato Brasileiro",
        "League One",
        "League Two",
        "Ligue 1",
        "Bundesliga",
        "Champions League",
        "Europa League",
    ]

    async def projectionAllTeams(self, leagues=None):
        if leagues is None:
            leagues = ProjectionAllTeams.DEFAULT_LEAGUES

        _total_start = time.time()
        _league_times = {}

        # Load all shared source data ONCE before the loop (eliminates 11x redundant file reads)
        data_folder_path = ProjectionService.DATA_FOLDER_PATH
        if not ProjectionService._cache.is_loaded():
            ProjectionService._cache.load(str(data_folder_path))
        logger.info("All-leagues: shared source data loaded into cache")

        _euro_comp_service = EuroCompProjectionService()

        for league in leagues:
            try:
                # Delegate euro comps to dedicated service
                if EuroCompProjectionService.is_euro_comp(league):
                    logger.info(f"[{league}] Delegating to EuroCompProjectionService")
                    _start_time = time.time()
                    request = LeagueRequest(league=league)
                    await _euro_comp_service.projections(request)
                    _league_times[league] = round(time.time() - _start_time, 1)
                    logger.info(f"[{league}] DONE in {_league_times[league]}s")
                    continue

                logger.info(f"[{league}] START projections"); _start_time = time.time()
            # - Fixture Player Stats
            # - Fixture Team Stats
            # - Standings
            # - Fixtures
            # - Players
            # - Competitions
            # - Teams
            # - Seasons
            # - Stats Types
            # - Competition Season Teams
            # - Bet365 Odds


                data_folder_path = ProjectionService.DATA_FOLDER_PATH

                model_file_path = ProjectionService.MODEL_FILE_PATH

                save_file_path = ProjectionService.SAVE_FILE_PATH

                # Only thing to change here is the league, unless league is MLS as you need to specify the date range of fixtures you want to project as well.

                # In[4]:

                # league = league_request.league or 'Championship'
                # league = 'Brazil Serie A'

                date_from = pd.to_datetime('today')
                date_to = date_from + pd.DateOffset(days=ProjectionService.DAYS)

                # In[5]:

                league_dashed = league.replace(' ', '-').replace('.', '').lower()
                league_weightings_df = ProjectionService._cache.league_weightings

                league_row = league_weightings_df[league_weightings_df['League'] == league]
                if len(league_row) > 0:
                    league_below = league_row['League Below'].values[0]
                    league_above = league_row['League Above'].values[0]
                    league_below_attack_weight = league_row['League Below Attack Weight'].values[0]
                    league_below_defense_weight = league_row['League Below Defense Weight'].values[0]
                    league_above_attack_weight = league_row['League Above Attack Weight'].values[0]
                    league_above_defense_weight = league_row['League Above Defense Weight'].values[0]
                    country_code = league_row['code'].values[0]
                    div = league_row['div'].values[0]
                    weightings = [league_above_attack_weight, league_above_defense_weight, league_below_attack_weight,
                                  league_below_defense_weight]
                    mv_beta = league_row['mv_beta'].values[0]
                    odds_beta = league_row['odds_beta'].values[0]
                else:
                    # League not in League Weightings (e.g. Champions League, Europa League)
                    league_below = None
                    league_above = None
                    league_below_attack_weight = 1.0
                    league_below_defense_weight = 1.0
                    league_above_attack_weight = 1.0
                    league_above_defense_weight = 1.0
                    country_code = None
                    div = None
                    weightings = [1.0, 1.0, 1.0, 1.0]
                    mv_beta = 0.0
                    odds_beta = 1.0

                # In[6]:

                if league == 'Premier League':
                    xG = True
                    fpl = True
                elif league == 'Championship':
                    xG = True
                    fpl = False
                else:
                    xG = False
                    fpl = False

                # # **Load Data**
                #
                # Get Data from Previous Weeks Results + Next Weeks Fixtures (maybe add something to check the data has actually been uploaded)

                # In[10]:

                ## THIS IS ALL NEW - LOAD IN MODEL, ACCURACY AND RATINGS DATASETS

                # Model Dataset
                model_dataset_all = ProjectionService._read_df(os.path.join(data_folder_path, "all_leagues_model_dataset_with_history"))
                model_dataset_league = ProjectionService._read_df_with_fallback(os.path.join(data_folder_path, f"{league}_model_dataset_with_history"), os.path.join(data_folder_path, "all_leagues_model_dataset_with_history"))

                # Accuracy Dataset
                projection_accuracy_dataset_league = ProjectionService._read_df_with_fallback(os.path.join(data_folder_path, f"{league}_accuracy_dataset"), os.path.join(data_folder_path, "all_leagues_accuracy_dataset"))
                projection_accuracy_dataset_all = ProjectionService._read_df(os.path.join(data_folder_path, "all_leagues_accuracy_dataset"))

                # Ratings Dataset — DB-sourced via DataCache (was parquet).
                all_team_ratings = ProjectionService._cache.team_ratings.copy()

                # In[9]: Use shared cache (loaded once before loop)
                player_stats = ProjectionService._cache.player_stats.copy()
                team_stats = ProjectionService._cache.team_stats.copy()
                standings = ProjectionService._cache.standings.copy()
                seasons = ProjectionService._cache.seasons
                comps = ProjectionService._cache.comps
                comp_teams = ProjectionService._cache.comp_teams
                teams = ProjectionService._cache.teams
                players = pd.read_csv(os.path.join(data_folder_path, "players.csv"))
                players['display_name'] = players['display_name'].str.strip()
                fixtures_df = ProjectionService._cache.fixtures_df.copy()
                b365_odds = ProjectionService._cache.b365_odds
                stats_types = ProjectionService._cache.stats_types




                if league == 'Campeonato Brasileiro':
                    league_id = 648
                else:
                    league_id = get_league_id(league, comps)
                fixtures = fixtures_df[fixtures_df['competition_id'] == league_id]
                league_standings = standings[standings['competition_id'] == league_id]
                if pd.notna(league_above):
                    league_above_id = get_league_id(league_above, comps)
                else:
                    league_above_id = None
                if pd.notna(league_below):
                    league_below_id = get_league_id(league_below, comps)
                else:
                    league_below_id = None
                previous_season_id = get_season_id(league_id, seasons, True)
                current_season_id = get_season_id(league_id, seasons, False)
                standings = standings[standings['season_id'] == current_season_id]
                matches_played = standings['played'].mode().values[
                    0]  # NEW - This gets the number of matches played so far in the current season
                season_fixtures = fixtures[
                    fixtures['season_id'] == current_season_id]  # NEW - This gets the fixtures for the current season only
                total_matches = (season_fixtures['home_team_id'].value_counts() + season_fixtures[
                    'away_team_id'].value_counts()).mean().round(
                    0)  # NEW - This calculates the total number of matches in the season
                if league == 'League Two':
                    previous_season_id_below = 23846
                else:
                    previous_season_id_below = get_season_id(league_below_id, seasons, True) if league_below_id else None
                previous_season_id_above = get_season_id(league_above_id, seasons, True) if league_above_id else None
                stat_list = get_stat_list()

                # ## **Get Previous Weeks Data**

                # ## For Model Dataset

                # In[ ]:

                ## THIS IS ALL NEW - FILL IN ANY MISSING TEAM STATS IN MODEL DATASET

                model_dataset_league['comp_id'] = league_id
                previous_fixtures = model_dataset_league[model_dataset_league.isnull().any(axis=1)]
                logger.info(f"[{league}] Filling missing stats in model dataset ({len(previous_fixtures)} rows)...")
                for i in range(len(previous_fixtures)):
                    if i > 0 and i % 50 == 0:
                        logger.info(f"[{league}] model dataset fill progress: {i}/{len(previous_fixtures)}")
                    fixture_id = previous_fixtures.iloc[i]['id']
                    team = previous_fixtures.iloc[i]['Team']
                    team_id = get_team_id(previous_fixtures.iloc[i]['Team'], teams)
                    fixture_stats = team_stats[team_stats['fixture_id'] == fixture_id]
                    for stat in stat_list:
                        if stat == 'Goals':
                            continue
                        team_df = fixture_stats[fixture_stats['stats_type_id'] == get_stat_id(stat, stats_types)]
                        team_stat_df = team_df[team_df['team_id'] == team_id]
                        stat_value = team_stat_df['value'].values[0] if not team_stat_df.empty else 0
                        model_dataset_league.loc[(model_dataset_league['id'] == fixture_id) & (
                                    model_dataset_league['Team'] == team), 'Team ' + stat] = stat_value
                        model_dataset_all.loc[(model_dataset_all['id'] == fixture_id) & (
                                    model_dataset_all['Team'] == team), 'Team ' + stat] = stat_value

                # ## For Accuracy Dataset

                # In[ ]:

                ## THIS IS ALL NEW - FILL IN ANY MISSING TEAM STATS IN ACCURACY DATASET

                previous_accuracy_fixtures = projection_accuracy_dataset_league[
                    projection_accuracy_dataset_league.isnull().any(axis=1)]
                previous_accuracy_fixtures = previous_accuracy_fixtures[
                    previous_accuracy_fixtures['kickoff_datetime'] < pd.to_datetime('today')]
                logger.info(f"[{league}] Filling missing stats in accuracy dataset ({len(previous_accuracy_fixtures)} rows)...")
                for i in range(len(previous_accuracy_fixtures)):
                    if i > 0 and i % 50 == 0:
                        logger.info(f"[{league}] accuracy dataset fill progress: {i}/{len(previous_accuracy_fixtures)}")
                    fixture_id = previous_accuracy_fixtures.iloc[i]['fixture_id']
                    home_team_id = get_team_id(previous_accuracy_fixtures.iloc[i]['Home Team'], teams)
                    away_team_id = get_team_id(previous_accuracy_fixtures.iloc[i]['Away Team'], teams)
                    fixture_stats = team_stats[team_stats['fixture_id'] == fixture_id]
                    for stat in stat_list:
                        fixture_stat_df = fixture_stats[fixture_stats['stats_type_id'] == get_stat_id(stat, stats_types)]
                        home_team_stat_df = fixture_stat_df[fixture_stat_df['team_id'] == home_team_id]
                        away_team_stat_df = fixture_stat_df[fixture_stat_df['team_id'] == away_team_id]
                        home_stat_value = home_team_stat_df['value'].values[0] if not home_team_stat_df.empty else 0
                        away_stat_value = away_team_stat_df['value'].values[0] if not away_team_stat_df.empty else 0
                        # Update stat values for both datasets
                        for ds in [projection_accuracy_dataset_league, projection_accuracy_dataset_all]:
                            ds.loc[ds['fixture_id'] == fixture_id, 'Home ' + stat] = home_stat_value
                            ds.loc[ds['fixture_id'] == fixture_id, 'Away ' + stat] = away_stat_value
                            ds.loc[ds['fixture_id'] == fixture_id, 'Total ' + stat] = home_stat_value + away_stat_value

                        # Only for 'Goals', update result columns
                        if stat == 'Goals':
                            home_win = home_stat_value > away_stat_value
                            draw = home_stat_value == away_stat_value
                            over_2_5 = (home_stat_value + away_stat_value) > 2.5
                            over_1_5 = (home_stat_value + away_stat_value) > 1.5
                            btts = home_stat_value > 0 and away_stat_value > 0
                            away_cs = home_stat_value == 0
                            home_cs = away_stat_value == 0

                            for ds in [projection_accuracy_dataset_league, projection_accuracy_dataset_all]:
                                ds.loc[ds['fixture_id'] == fixture_id, 'Home Win'] = 'Y' if home_win else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Draw'] = 'Y' if draw else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Away Win'] = 'Y' if not home_win and not draw else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Over 2.5'] = 'Y' if over_2_5 else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Over 1.5'] = 'Y' if over_1_5 else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'BTTS'] = 'Y' if btts else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Away Clean Sheet'] = 'Y' if away_cs else 'N'
                                ds.loc[ds['fixture_id'] == fixture_id, 'Home Clean Sheet'] = 'Y' if home_cs else 'N'

                # ## **Re-Train Models**

                # In[ ]:

                ## THIS IS ALL NEW - RE-TRAIN AND SAVE MODELS

                league_training_dataset = model_dataset_league.dropna().copy()
                league_training_dataset = league_training_dataset[league_training_dataset['Team Passes'] > 0]
                league_training_dataset.reset_index(drop=True, inplace=True)
                all_league_training_dataset = model_dataset_all.dropna().copy()
                all_league_training_dataset = all_league_training_dataset[all_league_training_dataset['Team Passes'] > 0]
                all_league_training_dataset.reset_index(drop=True, inplace=True)

                for stat in stat_list:
                    if stat == 'Goals':
                        continue

                    # Putanja do modela po ligi
                    file_path = os.path.join(model_file_path, league, f"{league}_{stat}_model.sav")

                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            model = pickle.load(f)
                        logger.info(f"[{league}] Model loaded: {stat}")
                    else:
                        logger.info(f"[{league}] Training model: {stat}...")
                        predictors = ['Team ' + stat + ' History', 'Opponent ' + stat + ' History Against']
                        target = 'Team ' + stat
                        X = league_training_dataset[predictors]
                        y = league_training_dataset[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y)

                        if stat in ['Passes', 'Successful Passes']:
                            model = fit_model(X_train, y_train)
                        else:
                            model = grid_search(X_train, y_train)

                        # Snimanje modela
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'wb') as f:
                            pickle.dump(model, f)
                        logger.info(f"[{league}] Model trained and saved: {stat}")

                    # Isto za model svih liga
                    folder_path = os.path.join(model_file_path, "All Leagues")
                    os.makedirs(folder_path, exist_ok=True)
                    file_path_all = os.path.join(folder_path, f"All_Leagues_{stat}_model.sav")

                    if os.path.exists(file_path_all):
                        with open(file_path_all, 'rb') as f:
                            model_all = pickle.load(f)
                        logger.info(f"[{league}] All-leagues model loaded: {stat}")
                    else:
                        logger.info(f"[{league}] Training all-leagues model: {stat}...")
                        X_all = all_league_training_dataset[predictors]
                        y_all = all_league_training_dataset[target]
                        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all)
                        model_all = grid_search(X_train_all, y_train_all)

                        with open(file_path_all, 'wb') as f:
                            pickle.dump(model_all, f)
                        logger.info(f"[{league}] All-leagues model trained and saved: {stat}")

                    # ## **Re-Calculate Accuracy**

                # ## Team Stat Accuracy

                # In[ ]:

                ## THIS IS ALL NEW - CALCULATE AND SAVE PROJECTION ACCURACY

                logger.info(f"[{league}] Step: calculating projection accuracy")

                cols = ['Home {}', 'Away {}', 'Total {}', 'Total Projected {}', 'Home Projected {}', 'Away Projected {}']
                metrics = [
                    ('Fixture Error', lambda df, s: df[f'Total Projected {s}'] - df[f'Total {s}']),
                    ('Home Team Error', lambda df, s: df[f'Home Projected {s}'] - df[f'Home {s}']),
                    ('Away Team Error', lambda df, s: df[f'Away Projected {s}'] - df[f'Away {s}']),
                ]
                abs_metrics = [
                    ('Fixture Abs Error', 'Fixture Error'),
                    ('Home Team Abs Error', 'Home Team Error'),
                    ('Away Team Abs Error', 'Away Team Error'),
                ]

                def calc_errors(df, stat):
                    d = {name: func(df, stat) for name, func in metrics}
                    for name, base in abs_metrics:
                        d[name] = d[base].abs()
                    return d

                def summarize(df, stat):
                    d = calc_errors(df, stat)
                    return {
                        'Stat': stat,
                        'Fixture Error': d['Fixture Error'].mean(),
                        'Home Team Error': d['Home Team Error'].mean(),
                        'Away Team Error': d['Away Team Error'].mean(),
                        'Fixture Abs Error': d['Fixture Abs Error'].mean(),
                        'Home Team Abs Error': d['Home Team Abs Error'].mean(),
                        'Away Team Abs Error': d['Away Team Abs Error'].mean(),
                    }

                projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all.dropna().copy()
                projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all_copy[
                    projection_accuracy_dataset_all_copy['Total Passes'] > 0]
                projection_accuracy_dataset_all_copy.reset_index(drop=True, inplace=True)
                projection_accuracy_dataset_league_copy = projection_accuracy_dataset_league.dropna().copy()
                projection_accuracy_dataset_league_copy = projection_accuracy_dataset_league_copy[
                    projection_accuracy_dataset_league_copy['Total Passes'] > 0]
                projection_accuracy_dataset_league_copy.reset_index(drop=True, inplace=True)
                accuracy_df_league = pd.DataFrame(
                    [summarize(projection_accuracy_dataset_league_copy, stat) for stat in stat_list])
                accuracy_df_all = pd.DataFrame([summarize(projection_accuracy_dataset_all_copy, stat) for stat in stat_list])
                accuracy_df_league = accuracy_df_league.round(2)
                accuracy_df_all = accuracy_df_all.round(2)
                # accuracy_df_league.to_csv(rf"{data_folder_path}\{league} Projection Accuracy.csv", index=False)
                # accuracy_df_all.to_csv(rf"{data_folder_path}\All Leagues Projection Accuracy.csv", index=False)

                # Za league
                file_path_league = os.path.join(data_folder_path, f"{league} Projection Accuracy.csv")
                accuracy_df_league.to_csv(file_path_league, index=False)

                # Za sve lige
                file_path_all = os.path.join(data_folder_path, "All Leagues Projection Accuracy.csv")
                accuracy_df_all.to_csv(file_path_all, index=False)

                # In[ ]:
                logger.info(f"[{league}] Step: projection accuracy saved")
                ## THIS IS ALL NEW - ADD ABSOLUTE ERROR COLUMNS TO ACCURACY DATASET

                for stat in stat_list:
                    # Calculate absolute errors
                    for prefix in ['Total', 'Home', 'Away']:
                        abs_err_col = f"{prefix} {stat} Absolute Error"
                        proj_col = f"{prefix} Projected {stat}"
                        actual_col = f"{prefix} {stat}"
                        projection_accuracy_dataset_all_copy[abs_err_col] = (
                                    projection_accuracy_dataset_all_copy[proj_col] - projection_accuracy_dataset_all_copy[
                                actual_col]).abs()
                        # Move the absolute error column next to projected column
                        cols = list(projection_accuracy_dataset_all_copy.columns)
                        if abs_err_col in cols and proj_col in cols:
                            idx = cols.index(proj_col) + 1
                            cols.remove(abs_err_col)
                            cols.insert(idx, abs_err_col)
                            projection_accuracy_dataset_all_copy = projection_accuracy_dataset_all_copy[cols]

                # projection_accuracy_dataset_all_copy.to_excel(rf"{data_folder_path}\Accuracy Dataset with Errors.xlsx",
                #                                               index=False)
                ProjectionService._write_df(projection_accuracy_dataset_all_copy, os.path.join(data_folder_path, "Accuracy Dataset with Errors"))

                # ## **Team Ratings**
                #
                # Team Ratings are calculated by combining a weighted average of Actual Goals (30%) and Expected Goals (70%) over the last 50 games.

                # In[ ]:

                ## UPDATED - Added new input: previous_team_rating (using the team_ratings dataset)
                ## UPDATED - Change weight to 0.95 and games to 30

                logger.info(f"[{league}] Step: calculating team ratings...")
                _t = time.time()
                ratings = get_ratings(league_id=league_id, previous_team_ratings=all_team_ratings,
                                      current_season_id=current_season_id,
                                      all_season_ids=[current_season_id, previous_season_id, previous_season_id_above,
                                                      previous_season_id_below],
                                      comp_teams=comp_teams, teams_df=teams, fixtures_df=fixtures_df, team_stats=team_stats,
                                      stats_types=stats_types, weight=0.96, games=30, weightings=weightings)
                ratings.to_csv(f"{save_file_path}/{league} Get Ratings.csv", index=False)
                logger.info(f"[{league}] Step: team ratings calculated ({time.time()-_t:.1f}s)")
                # In[12]:

                team_mapping = {
                    'Bournemouth': 'AFC Bournemouth',
                    'Wimbledon': 'AFC Wimbledon',
                    'Esporte Clube Juventude': 'Juventude',
                    'Sport Club Corinthians Paulista': 'Corinthians',
                    'Esporte Clube Bahia': 'Bahia',
                    'Clube de Regatas Vasco da Gama': 'Vasco da Gama',
                    'CR Flamengo': 'Flamengo',
                    'Fluminense Football Club': 'Fluminense',
                    'Sport Club do Recife': 'Sport Recife',
                    'Sport Club Internacional': 'Internacional',
                    'Botafogo de Futebol e Regatas': 'Botafogo',
                    'Grêmio Foot-Ball Porto Alegrense': 'Grêmio',
                    'Cruzeiro Esporte Clube': 'Cruzeiro',
                    'Sociedade Esportiva Palmeiras': 'Palmeiras',
                    'Clube Atlético Mineiro': 'Atlético Mineiro',
                    'Esporte Clube Vitória': 'Vitória',
                    'São Paulo Futebol Clube': 'São Paulo',
                    'Fortaleza Esporte Clube': 'Fortaleza',
                    'Santos': 'Santos',
                    'Red Bull Bragantino': 'Bragantino',
                    'Mirassol Futebol Clube (SP)': 'Mirassol',
                    'Ceará Sporting Club': 'Ceará',
                    'Associação Chapecoense de Futebol': 'Chapecoense',
                    'Club Athletico Paranaense': 'Athletico PR',
                    'Coritiba Foot Ball Club': 'Coritiba',
                    'Clube do Remo (PA)': 'Remo',
                    'D.C. United': 'DC United',
                    'Orlando City SC': 'Orlando City',
                    'San Jose Earthquakes': 'SJ Earthquakes',
                    'Sporting Kansas City': 'Sporting KC',
                    'New York Red Bulls': 'New York RB',
                    'Red Bull New York': 'New York RB',
                    'Los Angeles Galaxy': 'LA Galaxy',
                    'New England Revolution': 'New England',
                    'Real Salt Lake City': 'Real Salt Lake',
                    'Los Angeles': 'Los Angeles FC',
                    'Inter Miami CF': 'Inter Miami',
                    'St. Louis CITY': 'St. Louis City',
                    'Nashville': 'Nashville SC',
                    'Montréal': 'CF Montréal',
                    'Barcelona': 'FC Barcelona',
                    'Getafe CF': 'Getafe',
                    'Villarreal CF': 'Villarreal',
                    'Atlético de Madrid': 'Atlético Madrid',
                    'Athletic Bilbao': 'Athletic Club',
                    'Real Betis Balompié': 'Real Betis',
                    'Valencia CF': 'Valencia',
                    'Levante UD': 'Levante',
                    'CA Osasuna': 'Osasuna',
                    'Espanyol Barcelona': 'Espanyol',
                    'RCD Mallorca': 'Mallorca',
                    'Elche CF': 'Elche',
                    'Paris Saint-Germain': 'Paris Saint Germain',
                    'AS Monaco': 'Monaco',
                    'OGC Nice': 'Nice',
                    'Olympique Lyon': 'Olympique Lyonnais',
                    'Stade Rennais': 'Rennes',
                    'RC Lens': 'Lens',
                    'Aston Villa ': 'Aston Villa FC',
                    'RC Strasbourg Alsace': 'Strasbourg',
                    'Stade Brestois 29': 'Brest',
                    'AJ Auxerre': 'Auxerre',
                    'Le Havre AC': 'Le Havre',
                    'S Napoli': 'Napoli',
                    'Inter Milan': 'Inter',
                    'Bologna 1909': 'Bologna',
                    'A Fiorentina': 'Fiorentina',
                    'Como 1907': 'Como',
                    'Pisa Sporting Club': 'Pisa',
                    'Cagliari Calcio': 'Cagliari',
                    'Genoa C': 'Genoa',
                    'Parma Calcio 1913': 'Parma',
                    'Udinese Calcio': 'Udinese',
                    'LO Lille': 'LOSC Lille',
                    'Angers O': 'Angers SCO',
                    'Bayern Munich': 'FC Bayern München',
                    '1.FSV Mainz 05': 'FSV Mainz 05',
                    'Freiburg': 'SC Freiburg',
                    'SV Werder Bremen': 'Werder Bremen',
                    '1. Union Berlin': 'FC Union Berlin',
                    '1. Köln': 'FC Köln',
                    'Augsburg': 'FC Augsburg',
                    '1. Heidenheim 1846': 'Heidenheim',
                    'TSG 1899 Hoffenheim': 'TSG Hoffenheim',
                    'PSV Eindhoven': 'PSV',
                    'Feyenoord Rotterdam': 'Feyenoord',
                    'Ajax Amsterdam': 'Ajax',
                    'AZ Alkmaar': 'AZ',
                    'Utrecht': 'FC Utrecht',
                    'Twente Enschede': 'FC Twente',
                    'Heerenveen': 'SC Heerenveen',
                    'Groningen': 'FC Groningen',
                    'Excelsior Rotterdam': 'Excelsior',
                    'N Breda': 'NAC Breda',
                    'Volendam': 'FC Volendam',
                    'Heart of Midlothian': 'Hearts',
                    'SL Benfica': 'Benfica',
                    'Braga': 'Sporting Braga',
                    'CD Santa Clara': 'Santa Clara',
                    'GD Estoril Praia': 'Estoril',
                    'Vitória Guimarães': 'Vitória SC',
                    'CD Tondela': 'Tondela',
                    'CD Nacional': 'Nacional',
                    'Avs Futebol': 'AVS',
                }

                # In[13]:

                try:
                    # second_ratings = pd.read_excel(rf"{data_folder_path}\{league} Promoted Team Ratings.xlsx")
                    second_ratings = pd.read_excel(f"{data_folder_path}/{league} Promoted Team Ratings.xlsx")
                    second_ratings = second_ratings[['Team', 'Attack', 'Defense']]
                    second_ratings['Attack'] = (second_ratings['Attack']) * league_below_attack_weight
                    second_ratings['Defense'] = (second_ratings[
                        'Defense']) / league_below_defense_weight  # UPDATED - divide instead of multiply
                    promoted_teams = second_ratings['Team'].unique()
                    old_weight = 0.85 ** matches_played  # NEW - uses matches played so far in season
                    new_weight = 1 - old_weight  # NEW - opposite of old weight
                    ratings_copy = ratings.copy()  # NEW - This was just to stop warnings in my program so not necessary for functionality
                    second_ratings['New Attack'] = second_ratings['Team'].map(ratings_copy.set_index('Team')[
                                                                                  'Attack'])  # NEW - This maps the new attack rating from get_ratings function
                    second_ratings['New Defense'] = second_ratings['Team'].map(ratings_copy.set_index('Team')[
                                                                                   'Defense'])  # NEW - This maps the new defense rating from get_ratings function
                    second_ratings['Attack'] = (second_ratings['Attack'] * old_weight) + (
                                second_ratings['New Attack'] * new_weight)  # NEW - This calculates the updated attack rating
                    second_ratings['Defense'] = (second_ratings['Defense'] * old_weight) + (
                                second_ratings['New Defense'] * new_weight)  # NEW - This calculates the updated defense rating
                    second_ratings = second_ratings[['Team', 'Attack', 'Defense']]  # NEW - This drops the temporary columns
                    ratings = ratings[~ratings['Team'].isin(promoted_teams)]
                    ratings = pd.concat([ratings, second_ratings], ignore_index=True)
                    ratings.dropna(inplace=True)
                    ratings.reset_index(drop=True, inplace=True)
                except:
                    pass

                # In[ ]:

                # ratings['Attack'] = (ratings['Attack'] / ratings['Attack'].mean()) * 100
                # ratings['Defense'] = (ratings['Defense'] / ratings['Defense'].mean()) * 100
                # ratings['Overall'] = (ratings['Attack'] + ratings['Defense']) / 2
                # ratings.sort_values('Overall', ascending=False, inplace=True)
                # ratings.reset_index(drop=True, inplace=True)

                # In[15]:

                ## NEW - Function to rescale market values

                def rescale_to_range(series, new_min=0.5, new_max=2.0):
                    old_min = series.min()
                    old_max = series.max()
                    return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)

                # In[ ]:

                if country_code is not None:
                    try:
                        market_values = get_market_value(league_dashed, div, country_code)
                        market_values['MV Index'] = market_values['Market Value'].astype(float) / market_values['Market Value'].astype(
                            float).median()
                        market_values['MV Index'] = np.log1p(market_values['MV Index'])
                        market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()
                        max = market_values['MV Index'].max() if market_values['MV Index'].max() < 2.0 else 2.0  # NEW - Cap max at 2.0
                        min = market_values['MV Index'].min() if market_values[
                                                                     'MV Index'].min() > 0.5 else 0.5  # NEW - Floor min at 0.5
                        market_values['MV Index'] = rescale_to_range(market_values['MV Index'], min,
                                                                     max)  # NEW - Rescale to new range to avoid outliers
                        market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()  # NEW - Re-normalize
                        market_values['Team'] = market_values['Team'].replace(team_mapping)
                        market_values['Team'] = market_values['Team'].str.strip()

                        ratings['Team'] = ratings['Team'].str.strip()
                        ratings['MV Index'] = ratings['Team'].map(market_values.set_index('Team')['MV Index'])
                        ratings['MV Index Reverse'] = (
                                    ratings['MV Index'].mean() / ratings['MV Index'])  # NEW - Inverse MV Index (for defence)
                        ratings['MV Index Reverse'] = ratings['MV Index Reverse'] / ratings[
                            'MV Index Reverse'].mean()  # NEW - Normalize

                        teams_to_map = ratings.loc[ratings['MV Index'].isna(), 'Team']  # NEW - Identify any teams not mapped

                        if len(teams_to_map) > 0:
                            market_values_not_mapped = market_values[~market_values['Team'].isin(ratings['Team'])]
                            logger.warning(f"[{league}] Team mapping missing — using neutral MV Index (1.0) for: {teams_to_map.tolist()} | Unmatched Transfermarkt names: {market_values_not_mapped['Team'].tolist()}")
                            ratings['MV Index'] = ratings['MV Index'].fillna(1.0)

                        total_match_perc = 38 / total_matches
                        mv_beta = league_weightings_df[league_weightings_df['League'] == league]['mv_beta'].values[0]
                        mv_beta = (mv_beta * (0.95 ** (matches_played * total_match_perc)))

                        ratings['MV Attack Underperformance'] = (ratings['MV Index'] - ratings['Attack'] / ratings[
                            'Attack'].mean()) * mv_beta
                        ratings['MV Attack Underperformance %'] = ratings['MV Attack Underperformance'] / ratings['Attack']
                        ratings['MV Defense Underperformance'] = (ratings['MV Index Reverse'] - ratings['Defense'] / ratings[
                            'Defense'].mean()) * mv_beta
                        ratings['MV Defense Underperformance %'] = ratings['MV Defense Underperformance'] / ratings['Defense']
                        ratings['Attack'] = ratings['Attack'] * (1 + ratings['MV Attack Underperformance %'])
                        ratings['Defense'] = ratings['Defense'] * (1 + ratings['MV Defense Underperformance %'])
                        ratings.drop(columns=['MV Defense Underperformance', 'MV Attack Underperformance', 'MV Index',
                                              'MV Defense Underperformance %', 'MV Attack Underperformance %', 'MV Index Reverse'],
                                     inplace=True)
                        logger.info(f"[{league}] Step: market value adjustments applied")
                    except Exception as _mv_err:
                        logger.warning(f"[{league}] Market value block failed: {_mv_err} — skipping MV adjustment")

                # Snapshot post-MV, pre-rescale ratings in xG/game units.
                ratings['Attack_xG'] = ratings['Attack']
                ratings['Defense_xG'] = ratings['Defense']
                ratings['Overall_xG'] = ratings['Attack'] - ratings['Defense']

                # In[17]:

                # Readjust so that 100 is the mean for Attack, Defense, and Overall
                for col in ['Attack', 'Defense']:
                    ratings[col] = ratings[col] / ratings[col].mean() * 100
                ratings['Overall'] = ratings['Attack'] - ratings['Defense']  # UPDATED - Overall is now Attack minus Defense
                ratings.sort_values('Overall', ascending=False, inplace=True)
                ratings.reset_index(drop=True, inplace=True)
                ratings = ratings.round(1)
                ratings['Rank'] = ratings.index + 1
                # Movement = rank change vs most recent snapshot at least 7 days old.
                # See projection_service.py for rationale (matchday-cadence proxy).
                from datetime import timedelta
                cutoff = pd.to_datetime('today').date() - timedelta(days=7)
                old_league = all_team_ratings[all_team_ratings['League'] == league]
                old_week_ago = old_league[old_league['Date'] <= cutoff]

                if len(old_week_ago) > 0:
                    old_ratings = old_week_ago[old_week_ago['Date'] == old_week_ago['Date'].max()].copy()
                    old_ratings.reset_index(drop=True, inplace=True)
                    old_ratings['Rank'] = old_ratings.index + 1
                    for i in range(len(ratings)):
                        team = ratings.loc[i, 'Team']
                        old_rank_vals = old_ratings.loc[old_ratings['Team'] == team, 'Rank'].values
                        old_rank = old_rank_vals[0] if len(old_rank_vals) > 0 else ratings.loc[i, 'Rank']
                        new_rank = ratings.loc[i, 'Rank']
                        ratings.loc[i, 'Movement'] = old_rank - new_rank
                else:
                    ratings['Movement'] = 0
                    logger.info(f"[{league}] No ratings snapshot older than 7 days — movement set to 0")
                ratings = ratings[['Team', 'Attack', 'Defense', 'Overall', 'Attack_xG', 'Defense_xG', 'Overall_xG', 'Movement']]

                # In[ ]:

                ## NEW - Update and save ratings to the all_team_ratings dataset

                ratings['Date'] = pd.to_datetime('today').date()
                ratings['League'] = league
                from app.repository.team_ratings_repo import insert_team_ratings_async
                await insert_team_ratings_async(
                    ratings, league, league_id, ProjectionService._cache.teams
                )

                logger.info(f"[{league}] Step: team ratings saved to DB")


                all_team_ratings[all_team_ratings['League'] == league].to_csv(f"{save_file_path}/{league} Team Ratings.csv", index=False)

                # ## **Make Predictions for Next Fixture Round**
                #
                # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

                # In[18]:

                fixtures['kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
                next_fix = fixtures[(fixtures['kickoff_datetime'] >= date_from) & (fixtures['kickoff_datetime'] <= date_to)]
                next_fix = next_fix[
                    ['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
                     'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
                next_fix['home_team'] = next_fix['home_team_id'].apply(lambda x: get_team(x, teams))
                next_fix['away_team'] = next_fix['away_team_id'].apply(lambda x: get_team(x, teams))
                next_fix = next_fix.drop(columns=['home_team_id', 'away_team_id'])
                next_fix.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
                next_fix.reset_index(drop=True, inplace=True)

                # In[ ]:

                avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
                avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)

                logger.info(f"[{league}] avg_home_goals={avg_home_goals:.3f}, avg_away_goals={avg_away_goals:.3f}")
            

                score_preds = make_round_goal_prediction(next_fix, ratings, avg_home_goals, avg_away_goals)
                score_preds.to_csv(f"{save_file_path}/{league} Score preds.csv")
                # boost = get_draw_boost(ratings, avg_home_goals, avg_away_goals, get_draw_perc(league_id, fixtures))
                boost = 1.1  # NEW - Set draw boost to fixed value
                score_preds['Home Odds %'] = ((1 / next_fix['bet365_home_odds_decimal']) * 100)
                score_preds['Draw Odds %'] = ((1 / next_fix['bet365_draw_odds_decimal']) * 100)
                score_preds['Away Odds %'] = ((1 / next_fix['bet365_away_odds_decimal']) * 100)
                next_fix.to_csv(f"{save_file_path}/{league} Next Fix.csv", index=False)

                home_win = []
                draw = []
                away_win = []
                home_clean = []
                away_clean = []
                over_1 = []
                over_2 = []
                btts = []
                for i in range(len(score_preds)):
                    bookie_margin = 1 + (
                                score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
                            i, 'Away Odds %'] - 100) / 100
                    score_preds.loc[i, 'Home Odds %'] = (score_preds.loc[i, 'Home Odds %'] / bookie_margin).round(2)
                    score_preds.loc[i, 'Draw Odds %'] = (score_preds.loc[i, 'Draw Odds %'] / bookie_margin).round(2)
                    score_preds.loc[i, 'Away Odds %'] = (score_preds.loc[i, 'Away Odds %'] / bookie_margin).round(2)
                    home_goals = score_preds['Home Goals'][i]
                    away_goals = score_preds['Away Goals'][i]
                    if pd.isna(score_preds['Home Odds %'][i]) == False:
                        home_win_prob, draw_prob, away_win_prob = get_result_probs(home_goals, away_goals, boost)
                        adjusted_home_win_prob = home_win_prob + ((score_preds['Home Odds %'][i] - home_win_prob) * odds_beta)
                        adjusted_draw_prob = draw_prob + ((score_preds['Draw Odds %'][i] - draw_prob) * odds_beta)
                        adjusted_away_win_prob = away_win_prob + ((score_preds['Away Odds %'][i] - away_win_prob) * odds_beta)
                        new_home_goals, new_away_goals = find_inputs_for_probs(home_goals, away_goals, adjusted_home_win_prob,
                                                                               adjusted_draw_prob, adjusted_away_win_prob,
                                                                               boost)
                        score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
                        score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
                        home_clean_sheet = poisson.pmf(0, new_away_goals)
                        away_clean_sheet = poisson.pmf(0, new_home_goals)
                    else:
                        new_home_goals = home_goals
                        new_away_goals = away_goals
                        adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = get_result_probs(home_goals,
                                                                                                              away_goals, boost)
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

                # score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)
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
                score_preds_with_odds = score_preds.copy()  # NEW - Create a copy with odds included
                score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'],
                                 inplace=True)  # NEW - Drop odds from main predictions dataframe

                # score_preds.to_csv(rf"{save_file_path}\{league} Fixtures.csv", index=False)
                score_preds.to_csv(f"{save_file_path}/{league} Fixtures.csv", index=False)

                logger.info(f"[{league}] Inserting fixtures into DB ({len(score_preds)} rows)...")
                _t = time.time()
                await insert_fixtures_async(score_preds, teams=teams)
                logger.info(f"[{league}] Fixtures inserted ({time.time()-_t:.1f}s)")

                # In[ ]:

                ## NEW - Update accuracy dataset with new predictions

                score_preds_with_odds.rename(
                    columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
                    inplace=True)
                score_preds_with_odds['Total Projected Goals'] = score_preds_with_odds['Home Projected Goals'] + \
                                                                 score_preds_with_odds['Away Projected Goals']
                score_preds_with_odds['comp_id'] = league_id
                projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_with_odds],
                                                               ignore_index=True)
                score_preds_with_odds.rename(
                    columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
                    inplace=True)
                score_preds_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

                # In[ ]:

                ## NEW - 4+ STAR BETS SECTION

                # ## **4+ Star Bets**

                # In[ ]:

                # NEW - Load previous best bets file and append new best bets

                # best_bets = pd.read_excel(rf"{data_folder_path}\Best Bets.xlsx")
                best_bets = ProjectionService._read_df(f"{data_folder_path}/Best Bets")

                new_best_bets = pd.DataFrame()
                for i in range(len(score_preds)):
                    fix_id = score_preds.loc[i, 'id']
                    date = score_preds.loc[i, 'kickoff_datetime']
                    date = date.strftime('%d-%m')
                    fix = fixtures_df[fixtures_df['id'] == fix_id]
                    home_win = float(score_preds.loc[i, 'Home Win %'].strip('%')) / 100
                    draw = float(score_preds.loc[i, 'Draw %'].strip('%')) / 100
                    away_win = float(score_preds.loc[i, 'Away Win %'].strip('%')) / 100
                    over_1_5_goals = float(score_preds.loc[i, 'Over 1.5 Goals %'].strip('%')) / 100
                    over_2_5_goals = float(score_preds.loc[i, 'Over 2.5 Goals %'].strip('%')) / 100
                    btts = float(score_preds.loc[i, 'Both Teams Score %'].strip('%')) / 100

                    home_win_odds = 1 / fix['bet365_home_odds_decimal'].values[0]
                    draw_odds = 1 / fix['bet365_draw_odds_decimal'].values[0]
                    away_win_odds = 1 / fix['bet365_away_odds_decimal'].values[0]
                    over_1_5_goals_odds = 1 / fix['over_1_5_odds_decimal'].values[0]
                    over_2_5_goals_odds = 1 / fix['over_2_5_odds_decimal'].values[0]
                    btts_odds = 1 / fix['bet365_btts_yes_odds_decimal'].values[0]

                    home_win_edge = home_win - home_win_odds
                    draw_edge = draw - draw_odds
                    away_win_edge = away_win - away_win_odds
                    over_1_5_goals_edge = over_1_5_goals - over_1_5_goals_odds
                    over_2_5_goals_edge = over_2_5_goals - over_2_5_goals_odds
                    btts_edge = btts - btts_odds

                    home_win_edge_rating = (home_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
                    draw_edge_rating = (draw_edge - (-0.1)) * 5 / (0.1 - (-0.1))
                    away_win_edge_rating = (away_win_edge - (-0.1)) * 5 / (0.1 - (-0.1))
                    over_1_5_goals_edge_rating = (over_1_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
                    over_2_5_goals_edge_rating = (over_2_5_goals_edge - (-0.1)) * 5 / (0.1 - (-0.1))
                    btts_edge_rating = (btts_edge - (-0.1)) * 5 / (0.1 - (-0.1))

                    home_win_prob_rating = (home_win) * 5 / (0.9)
                    draw_prob_rating = (draw) * 5 / (0.9)
                    away_win_prob_rating = (away_win) * 5 / (0.9)
                    over_1_5_goals_prob_rating = (over_1_5_goals) * 5 / (0.9)
                    over_2_5_goals_prob_rating = (over_2_5_goals) * 5 / (0.9)
                    btts_prob_rating = (btts) * 5 / (0.9)

                    home_win_total_rating = (home_win_edge_rating * 0.7 if home_win_edge_rating > 0 else 0) + (
                        home_win_prob_rating * 0.3 if home_win_prob_rating < 5 else 5 * 0.3)
                    draw_total_rating = (draw_edge_rating * 0.7 if draw_edge_rating > 0 else 0) + (
                        draw_prob_rating * 0.3 if draw_prob_rating < 5 else 5 * 0.3)
                    away_win_total_rating = (away_win_edge_rating * 0.7 if away_win_edge_rating > 0 else 0) + (
                        away_win_prob_rating * 0.3 if away_win_prob_rating < 5 else 5 * 0.3)
                    over_1_5_goals_total_rating = (
                                                      over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                                                      over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
                    over_2_5_goals_total_rating = (
                                                      over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
                                                      over_2_5_goals_prob_rating * 0.3 if over_2_5_goals_prob_rating < 5 else 5 * 0.3)
                    btts_total_rating = (btts_edge_rating * 0.7 if btts_edge_rating > 0 else 0) + (
                        btts_prob_rating * 0.3 if btts_prob_rating < 5 else 5 * 0.3)

                    for bet_type in ['Home Win', 'Draw', 'Away Win', 'Over 1.5 Goals', 'Over 2.5 Goals', 'BTTS']:
                        edge = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge']
                        edge_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_edge_rating']
                        prob_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_prob_rating']
                        total_rating = locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_total_rating']
                        if total_rating >= 4.0:
                            new_best_bets = pd.concat([new_best_bets, pd.DataFrame({
                                'Date': [date],
                                'Competition': [league],
                                'Home Team': [score_preds.loc[i, 'Home Team']],
                                'Away Team': [score_preds.loc[i, 'Away Team']],
                                'Bet Type': [bet_type],
                                'Rating': [round(total_rating, 1) if total_rating < 5 else 5.0],
                                'Edge %': [round(edge * 100, 2)],
                                'Price': [
                                    round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                            })], ignore_index=True)

                best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
                best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                          inplace=True)
                # best_bets.to_excel(rf"{data_folder_path}\Best Bets.xlsx", index=False)
                ProjectionService._write_df(best_bets, f"{data_folder_path}/Best Bets")

                # # **League Projections**
                logger.info(f"[{league}] Step: predicted table simulation complete")
                # In[ ]:

                if league != 'Major League Soccer':
                    season_fixtures = fixtures.copy()
                    today = pd.to_datetime('today')
                    season_fixtures['kickoff_datetime'] = pd.to_datetime(season_fixtures['kickoff_datetime'])
                    season_fixtures = season_fixtures[season_fixtures['kickoff_datetime'] >= today]
                    season_fixtures.loc[:, 'home_team'] = season_fixtures['home_team_id'].map(teams.set_index('id')['name'])
                    season_fixtures.loc[:, 'away_team'] = season_fixtures['away_team_id'].map(teams.set_index('id')['name'])
                    season_fixtures.sort_values(by='kickoff_datetime', inplace=True)
                    season_fixtures.reset_index(drop=True, inplace=True)

                    season_score_preds = make_round_goal_prediction(season_fixtures, ratings, avg_home_goals, avg_away_goals)

                    for i in range(len(season_score_preds)):
                        home_goals = season_score_preds['Home Goals'][i]
                        away_goals = season_score_preds['Away Goals'][i]

                    season_score_preds['Home Goals'] = season_score_preds['Home Goals'].round(2)
                    season_score_preds['Away Goals'] = season_score_preds['Away Goals'].round(2)
                    season_score_preds = season_score_preds.dropna(subset=['Home Goals', 'Away Goals'])

                    current_standings = standings.copy()
                    current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
                    current_standings.rename(
                        columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'}, inplace=True)
                    current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
                    current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
                    current_standings.reset_index(drop=True, inplace=True)
                    current_standings[['Points', 'Goals For', 'Goals Against', 'Goal Difference']] = current_standings[['Points', 'Goals For', 'Goals Against', 'Goal Difference']].fillna(0)
                    current_standings = current_standings.astype(
                        {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
                    current_league_table = {
                        team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                        team, points, gf, ga, gd in current_standings.values}

                    logger.info(f"[{league}] Step: running season simulation (10000 sims)...")
                    _t = time.time()
                    avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)
                    logger.info(f"[{league}] Step: season simulation complete ({time.time()-_t:.1f}s)")

                    avg_table_with_probs = get_avg_table_with_probs(league, avg_table, all_tables)
                    avg_table_with_probs_and_point_limits = get_avg_table_with_probs_and_point_limits(avg_table_with_probs,
                                                                                                      all_tables)
                    # avg_table_with_probs_and_point_limits.to_csv(rf"{save_file_path}\{league} Predicted Table.csv", index=False)
                    avg_table_with_probs_and_point_limits.to_csv(f"{save_file_path}/{league} Predicted Table.csv", index=False)
                    logger.info(f"[{league}] Inserting predicted table into DB ({len(avg_table_with_probs_and_point_limits)} rows)...")
                    _t = time.time()
                    await insert_predicted_table_async(avg_table_with_probs_and_point_limits, teams, comps, league)
                    logger.info(f"[{league}] Predicted table inserted ({time.time()-_t:.1f}s)")

                # # **Team Projections**
                #
                # Getting each Teams stat projections using the models

                # In[20]:

                stat_list = get_stat_list()

                # In[21]:

                models = load_all_models(stat_list, model_file_path, league)  # UPDATED - New League Parameter

                # In[22]:

                if next_fix.empty:
                    logger.info(f"[{league}] No upcoming fixtures in next {ProjectionService.DAYS} days, skipping")
                    continue

                todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

                # In[ ]:

                team_projections = get_team_round_predictions(next_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                              models, ratings=ratings,
                                                              league_weightings=[league_above_attack_weight,
                                                                                 league_above_defense_weight,
                                                                                 league_below_attack_weight,
                                                                                 league_below_defense_weight],
                                                              season_id=[current_season_id, previous_season_id,
                                                                         previous_season_id_above, previous_season_id_below],
                                                              games=50)
                team_projections.to_csv(f"{save_file_path}/{league} team projections.csv")
                # In[ ]:

                ## NEW - Add historical stats to the model dataset and drop them from team projections afterwards

                new_rows = []

                for i in range(len(team_projections)):
                    team_df = team_projections.iloc[[i]]
                    new_row = {}
                    new_row['id'] = team_df['fixture_id'].values[0]
                    new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
                    new_row['comp_id'] = league_id
                    new_row['Team'] = team_df['Team'].values[0]
                    new_row['Opponent'] = team_df['Opponent'].values[0]
                    new_row['Venue'] = team_df['Venue'].values[0]
                    for stat in stat_list:
                        new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                        new_row['Opponent ' + stat + ' History Against'] = \
                        team_df['Opponent ' + stat + ' History Against'].values[0]
                    new_rows.append(new_row)

                model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
                model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
                model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
                model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)

                ProjectionService._write_df(model_dataset_league, f"{data_folder_path}/{league}_model_dataset_with_history")
                ProjectionService._write_df(model_dataset_all, f"{data_folder_path}/all_leagues_model_dataset_with_history")

                # model_dataset_league.to_excel(rf"{data_folder_path}\{league}_model_dataset_with_history.xlsx", index=False)
                # model_dataset_all.to_excel(rf"{data_folder_path}\all_leagues_model_dataset_with_history.xlsx", index=False)

                team_projections.drop(
                    columns=['Team ' + stat + ' History' for stat in stat_list] + ['Opponent ' + stat + ' History Against' for
                                                                                   stat in stat_list], inplace=True)

                # In[ ]:

                avg_goals = (avg_home_goals + avg_away_goals) / 2

                league_team_stats = team_stats[
                    team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

                league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total',
                                                                                                   stats_types)].copy()  # NEW - all team shots for specific league
                league_shots['Date'] = league_shots['fixture_id'].map(
                    fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
                league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
                    league_shots['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
                league_shots['Weight'] = 0.9 ** (
                            league_shots['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
                league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
                league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots[
                    'value']  # NEW - calculate weighted shots
                avg_shots = league_shots['Weighted Shots'].sum() / league_shots[
                    'Weight'].sum()  # UPDATED - new formula for average shots

                league_shots_on_target = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots On Target',
                                                                                                             stats_types)].copy()  # NEW - all team shots on target for specific league
                league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
                    fixtures_df.set_index('id')['kickoff_datetime'])  # NEW - map fixture dates
                league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
                    league_shots_on_target['Date'])).dt.days // 7  # NEW - calculate weeks since kickoff
                league_shots_on_target['Weight'] = 0.9 ** (
                            league_shots_on_target['Weeks Since Kickoff'] - 5)  # NEW - apply weighting to more recent matches
                league_shots_on_target.loc[
                    league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1  # NEW - full weight for last 5 weeks
                league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target[
                    'value']  # NEW - calculate weighted shots on target
                avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target[
                    'Weight'].sum()  # UPDATED - new formula for average shots on target

                avg_shots_per_goal = avg_shots / avg_goals
                avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

                # In[ ]:

                # if 'team_projections' in globals():
                goals = []
                assists = []
                for i in range(len(team_projections)):
                    team = team_projections['Team'].iloc[i]
                    opp = team_projections['Opponent'].iloc[i]
                    # try:
                    #    team_pred = score_preds[score_preds['Home Team'] == team]['Home Goals'].values[0]
                    # except:
                    #    team_pred = score_preds[score_preds['Away Team'] == team]['Away Goals'].values[0]
                    fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[
                        i]]  # NEW - Get the fixture from score_preds
                    team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
                    fixture['Away Goals'].values[
                        0]  # UPDATED - new way to get team prediction that handles teams having multiple matches in a round
                    opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
                    fixture['Home Goals'].values[
                        0]  # UPDATED - new way to get opponent prediction that handles teams having multiple matches in a round
                    goals.append(team_pred)
                    assists.append((team_pred * 0.82).round(2))
                    projected_shots = team_projections['Shots Total'].iloc[i]
                    projected_shots_on_target = team_projections['Shots On Target'].iloc[i]

                    adjusted_shots, adjusted_shots_on_target = adjust_shots_projection(
                        team_pred,
                        projected_shots,
                        projected_shots_on_target,
                        avg_shots_per_goal,
                        avg_shots_on_target_per_goal
                    )
                    team_projections.at[i, 'Shots Total'] = adjusted_shots
                    team_projections.at[i, 'Shots On Target'] = adjusted_shots_on_target

                team_projections['Goals'] = goals
                team_projections['Assists'] = assists

                saves = []
                for i in range(len(team_projections)):
                    # opp = team_projections['Opponent'].iloc[i]
                    # try:
                    #    opp_pred = score_preds[score_preds['Home Team'] == opp]['Home Goals'].values[0]
                    # except:
                    #    opp_pred = score_preds[score_preds['Away Team'] == opp]['Away Goals'].values[0]
                    # saves.append(team_projections[team_projections['Team'] == opp]['Shots On Target'].values[0] - opp_pred)
                    fixture_id = team_projections['fixture_id'].iloc[i]  # NEW - Get fixture ID
                    fixture_team_projections = team_projections[
                        team_projections['fixture_id'] == fixture_id]  # NEW - Get both teams' projections for the fixture
                    fixture_team_projections = fixture_team_projections.drop(
                        i)  # NEW - Drop the current team to get the opponent projections
                    saves.append(
                        fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[
                            0])  # UPDATED - New way to calculate saves based on opponent projections that handles teams having multiple matches in a round

                team_projections['Saves'] = saves
                team_projections['Saves'] = team_projections['Saves'].round(2)  # NEW - Round saves to 2 decimal places
                team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
                team_projections = team_projections[
                    ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
                     'Key Passes'] + stat_list + ['Fouls Drawn', 'Saves']]
                team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)
            
            
                # print(team_projections['Assists', 'Key Passes'])
                # In[ ]:

                # team_projections_save = team_projections.copy()
                # team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1,
                #                            inplace=True)  # UPDATED - No longer dropping interceptions and accurate passes

                team_projections_save = team_projections.copy()
            
                team_projections_save.drop(
                    ['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'],
                    axis=1,
                    inplace=True,
                    errors='ignore'  # <- ovo sprečava KeyError ako kolona ne postoji
                )

                team_projections_save = team_projections_save.round(2)

                # team_projections_save.to_csv(rf"{save_file_path}\{league} Team.csv", index=False)
                team_projections_save.to_csv(f"{save_file_path}/{league} Team.csv", index=False)
                logger.info(f"[{league}] Inserting team projections into DB ({len(team_projections_save)} rows)...")
                _t = time.time()
                await insert_teams_async(team_projections_save, teams=teams)
                logger.info(f"[{league}] Team projections inserted ({time.time()-_t:.1f}s)")

                team_projections_save.rename(columns={'Accurate Passes': 'Successful Passes'},
                                             inplace=True)  # NEW - Rename back for consistency with other datasets

                # In[ ]:

                ## NEW - Update projection accuracy dataset

                for fixture_id in team_projections_save['fixture_id'].unique():
                    fixture_projections = team_projections_save[team_projections_save['fixture_id'] == fixture_id]
                    for stat in stat_list:
                        projection_accuracy_dataset_league.loc[
                            projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Home Projected ' + stat] = \
                        fixture_projections.loc[fixture_projections['Venue'] == 'H', stat].values[0]
                        projection_accuracy_dataset_league.loc[
                            projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Away Projected ' + stat] = \
                        fixture_projections.loc[fixture_projections['Venue'] == 'A', stat].values[0]
                        projection_accuracy_dataset_league.loc[
                            projection_accuracy_dataset_league['fixture_id'] == fixture_id, 'Total Projected ' + stat] = \
                        fixture_projections[stat].sum()

                projection_accuracy_dataset_league.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
                projection_accuracy_dataset_league.reset_index(drop=True, inplace=True)
                # projection_accuracy_dataset_league.to_excel(rf"{data_folder_path}\{league}_accuracy_dataset.xlsx", index=False)
                ProjectionService._write_df(projection_accuracy_dataset_league, f"{data_folder_path}/{league}_accuracy_dataset")

                projection_accuracy_dataset_all = pd.concat(
                    [projection_accuracy_dataset_all, projection_accuracy_dataset_league], ignore_index=True)
                projection_accuracy_dataset_all.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
                projection_accuracy_dataset_all.reset_index(drop=True, inplace=True)
                # projection_accuracy_dataset_all.to_excel(rf"{data_folder_path}\all_leagues_accuracy_dataset.xlsx", index=False)
                ProjectionService._write_df(projection_accuracy_dataset_all, f"{data_folder_path}/all_leagues_accuracy_dataset")

                #
                # # **Player Projections**
                #
                # Distributing the above dataframe's values to each player based on the % of teams total

                # In[ ]:

                # UPDATED: Removed xG parameter, added comps parameter and added season_id paramter

                logger.debug(f"[{league}] season_ids: {[current_season_id, previous_season_id, previous_season_id_above, previous_season_id_below]}")

                logger.info(f"[{league}] Starting player projections...")
                _t = time.time()
                pl_projections = distribute_team_predictions_to_players(player_stats, team_stats, team_projections, stats_types,
                                                                        fixtures_df, players, teams, comps, 0.97,
                                                                        season_id=[current_season_id, previous_season_id,
                                                                                   previous_season_id_above,
                                                                                   previous_season_id_below])
                logger.info(f"[{league}] Player projections computed - {len(pl_projections)} players ({time.time()-_t:.1f}s)")

                player_pos = []
                saves = []
                for i in range(len(pl_projections[['fixture_id', 'Player',
                                                   'Team']].values)):  # UPDATED - Using fixture_id for iteration as well
                    player = pl_projections['Player'].iloc[i]  # NEW - Get player name
                    team = pl_projections['Team'].iloc[i]  # NEW - Get team name
                    pos = get_player_position(player, team, players, teams)
                    if pos == 'GK':
                        team_projections_fix = team_projections[
                            team_projections['fixture_id'] == pl_projections['fixture_id'].iloc[
                                i]]  # NEW - Get the team projections for the fixture
                        saves.append(team_projections_fix[team_projections_fix['Team'] == team]['Saves'].values[0])
                    else:
                        saves.append(0)
                    player_pos.append(pos)
                pl_projections['Position'] = player_pos
                pl_projections['Saves'] = saves

                pl_projections = pl_projections[
                    ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
                     'Assists', 'Key Passes', 'Accurate Passes', 'Goals',
                     'Shots Total',
                     'Shots On Target',  'Passes',  'Interceptions', 'Tackles', 'Total Crosses',
                     'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]

                pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

                # ## **Predicted Lineups**
                #
                # Which players are predicted to play?

                # In[ ]:

                pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])].copy()
                pred_starters = pred_starters[pred_starters['stats_type_id'] == 11]
                logger.info(f"[{league}] Player projections: {len(pl_projections)} rows")
                start = []
                for i in range(len(pl_projections)):
                    team = pl_projections['Team'][i]
                    player_name = pl_projections['Player'][i]
                    try:
                        player_id = get_player_id(player_name, players, team)
                    except:
                        start.append('No')
                        continue
                    team_starters = pred_starters[pred_starters['team_id'] == get_team_id(team, teams)]
                    if player_id in team_starters['player_id'].values:
                        start.append('Yes')
                    else:
                        start.append('No')
                pl_projections['Start?'] = start
                pl_projections = pl_projections[
                    ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?', 'Shots Total',
                      'Goals', 'Assists', 'Key Passes', 'Accurate Passes',
                     'Shots On Target', 'Passes', 'Interceptions', 'Tackles', 'Total Crosses',
                     'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]
                pl_projections = pl_projections.round(2)

                # In[ ]:

                # pl_projections.sort_values(by='Goals', ascending=False, inplace=True)
                pl_projections.reset_index(drop=True, inplace=True)
                pl_projections = pl_projections.round(2)
                # pl_projections.to_csv(rf"{save_file_path}\{league} Player.csv", index=False)
                pl_projections.to_csv(f"{save_file_path}/{league} Player.csv", index=False)
                logger.info(f"[{league}] Inserting player projections into DB ({len(pl_projections)} rows)...")
                _t = time.time()
                await insert_player_async(pl_projections, teams=teams)
                logger.info(f"[{league}] Player projections inserted ({time.time()-_t:.1f}s)")

                # ## **FPL Points** (Premier League only)
                if fpl:
                    try:
                        fpl_file = os.path.join(data_folder_path, "PL Fantasy Players.xlsx")
                        pl_players = pd.read_excel(fpl_file)
                        pl_projections['Player'] = pl_projections['Player'].str.strip()
                        pl_projections['FPL Position'] = pl_projections['Player'].map(pl_players.set_index('Player')['FPL Position'])

                        # Compute extra stats per player (Clearances, Blocked Shots, Ball Recovery averages)
                        for _col in ['CBIT Hit Rate', 'CBIT Average', 'Clearances Average', 'Blocked Shots Average',
                                     'Ball Recovery Average', 'Tackles Won Average', 'Full Match Hit Rate']:
                            if _col not in pl_projections.columns:
                                pl_projections.loc[:, _col] = 0
                        for _player in pl_projections['Player'].unique():
                            _team = pl_projections[pl_projections['Player'] == _player]['Team'].values[0]
                            _pos = pl_projections[pl_projections['Player'] == _player]['FPL Position'].values[0]
                            try:
                                _cbit, _cbit_avg, _clr, _blk, _rec, _twon, _fmhr = get_extra_stats(
                                    _player, _pos, _team, teams, players, player_stats, fixtures_df, stats_types,
                                    weight=0.96, mins=50, games=50)
                                _mask = (pl_projections['Player'] == _player) & (pl_projections['Team'] == _team)
                                pl_projections.loc[_mask, 'CBIT Hit Rate'] = _cbit
                                pl_projections.loc[_mask, 'CBIT Average'] = _cbit_avg
                                pl_projections.loc[_mask, 'Clearances Average'] = _clr
                                pl_projections.loc[_mask, 'Blocked Shots Average'] = _blk
                                pl_projections.loc[_mask, 'Ball Recovery Average'] = _rec
                                pl_projections.loc[_mask, 'Tackles Won Average'] = _twon
                                pl_projections.loc[_mask, 'Full Match Hit Rate'] = _fmhr
                            except Exception:
                                continue
                        logger.info(f"[{league}] FPL: extra stats computed for {len(pl_projections['Player'].unique())} players")

                        fpl_points_dict_gk = {'Goals': 10, 'Assists': 3, 'Clean Sheet': 4, 'Saves': 1, 'Penalties Saved': 5, 'Goals Conceded': -1, 'Yellow Card': -1}
                        fpl_points_dict_def = {'Goals': 6, 'Assists': 3, 'Clean Sheet': 4, 'Goals Conceded': -1, 'Yellow Card': -1}
                        fpl_points_dict_mid = {'Goals': 5, 'Assists': 3, 'Clean Sheet': 1, 'Yellow Card': -1}
                        fpl_points_dict_fwd = {'Goals': 4, 'Assists': 3, 'Yellow Card': -1}

                        fpl_bonus_dict_gk = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Saves': 2.66, 'Penalties Saved': 8, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.5, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Goals Conceded': -4, 'Yellow Card': -3}
                        fpl_bonus_dict_def = {'Goals': 12, 'Winning Goal': 3, 'Assists': 9, 'Clean Sheet': 12, 'Clearances, Blocks & Interceptions': 0.5, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Goals Conceded': -4, 'Yellow Card': -3}
                        fpl_bonus_dict_mid = {'Goals': 18, 'Winning Goal': 3, 'Assists': 9, 'Clearances, Blocks & Interceptions': 0.5, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Yellow Card': -3}
                        fpl_bonus_dict_fwd = {'Goals': 24, 'Winning Goal': 3, 'Assists': 9, 'Key Passes': 1, 'Big Chances Created': 3, 'Successful Dribbles': 1, 'Clearance Offline': 9, 'Big Chances Missed': -3, 'Clearances, Blocks & Interceptions': 0.5, 'Recoveries': 0.33, 'Tackles Won': 2, 'Fouls Drawn': 1, 'Shots On Target': 2, 'Shots Off Target': -1, 'Offsides': -1, 'Fouls': -1, '70-79% Passes Completed': 2, '80-89% Passes Completed': 4, '90%+ Passes Completed': 6, 'Yellow Card': -3}

                        for _col in ['CBIT Hit Rate', 'Clearances Average', 'Blocked Shots Average', 'Interceptions', 'Ball Recovery Average']:
                            if _col not in pl_projections.columns:
                                pl_projections.loc[:, _col] = 0
                        fpl_point_df = get_fpl_points(pl_projections, score_preds, fpl_points_dict_gk, fpl_points_dict_def, fpl_points_dict_mid, fpl_points_dict_fwd)
                        bps_df = bonus_points_score(pl_projections, score_preds, fpl_bonus_dict_gk, fpl_bonus_dict_def, fpl_bonus_dict_mid, fpl_bonus_dict_fwd)
                        bonus = get_bonus_points(bps_df, score_preds, expo_factor=0.1)

                        fpl_df = fpl_point_df.merge(bonus, on=['Player', 'Team', 'Opponent'], how='left', suffixes=('', '_Bonus'))
                        fpl_df['FPL Points'] = fpl_df['PTS'] + fpl_df['Bonus Points'].fillna(0)
                        fpl_df = fpl_df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'FPL Points']].copy()
                        fpl_df = fpl_df.round(2)

                        logger.info(f"[{league}] Inserting FPL projections into DB ({len(fpl_df)} rows)...")
                        _t = time.time()
                        await insert_fpl_projections_async(fpl_df)
                        logger.info(f"[{league}] FPL projections inserted ({time.time()-_t:.1f}s)")
                    except Exception as e:
                        logger.warning(f"[{league}] FPL computation failed (skipping): {e}", exc_info=True)

                # ## **OPTA Points** (Premier League only)
                if fpl:
                    try:
                        opta_points_dict = {
                            'Goals': 10, 'Assists': 6, 'Shots Off': 2, 'Shots On Target': 4,
                            'Passes': 0.2, 'Interceptions': 2, 'Tackles': 2, 'Blocked Shots': 2,
                            'Total Crosses': 0.2, 'Yellow Cards': -2, 'Fouls': -1, 'Fouls Drawn': 1,
                            'Saves': 5, 'Offsides': -1, 'Goals Conceded': -1, 'Penalties Saved': 5
                        }
                        for _col in ['Blocked Shots Average']:
                            if _col not in pl_projections.columns:
                                pl_projections.loc[:, _col] = 0
                        opta_df = get_opta_points(pl_projections, score_preds, opta_points_dict)
                        opta_df = opta_df[['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'PTS', 'Floor PTS']].copy()
                        logger.info(f"[{league}] Inserting OPTA projections into DB ({len(opta_df)} rows)...")
                        _t = time.time()
                        await insert_opta_projections_async(opta_df)
                        logger.info(f"[{league}] OPTA projections inserted ({time.time()-_t:.1f}s)")
                    except Exception as e:
                        logger.warning(f"[{league}] OPTA computation failed (skipping): {e}", exc_info=True)

                # ## **FanTeam Points** (Premier League only)
                if fpl:
                    try:
                        fanteam_points_dict_gk = {
                            'Goals': 8, 'Assists': 3, 'Shots On Target': 1, 'Saves': 0.5,
                            'Penalties Saved': 5, 'Clean Sheet': 4, 'Win': 0.3, 'Lose': -0.3,
                            'Goals Conceded': -1, 'Yellow Card': -1
                        }
                        fanteam_points_dict_def = {
                            'Goals': 6, 'Assists': 3, 'Shots On Target': 0.6, 'Clean Sheet': 4,
                            'Win': 0.3, 'Lose': -0.3, 'Goals Conceded': -1, 'Yellow Card': -1
                        }
                        fanteam_points_dict_mid = {
                            'Goals': 5, 'Assists': 3, 'Shots On Target': 0.4, 'Clean Sheet': 1,
                            'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1
                        }
                        fanteam_points_dict_fwd = {
                            'Goals': 4, 'Assists': 3, 'Shots On Target': 0.4,
                            'Win': 0.3, 'Lose': -0.3, 'Yellow Card': -1, 'Full Match': 1
                        }
                        fanteam_mapping_file = os.path.join(data_folder_path, "Fanteam Mapping.xlsx")
                        fanteam_mapping = pd.read_excel(fanteam_mapping_file)
                        pl_projections['FanTeam Position'] = pl_projections['Player'].map(
                            pl_players.set_index('Player')['FanTeam Position'])
                        pl_projections['FanTeam ID'] = pl_projections['player_id'].map(
                            fanteam_mapping.set_index('SM Player ID')['FanTeam PlayerID'])
                        fanteam_data_file = os.path.join(data_folder_path, "Fanteam Data.csv")
                        if os.path.exists(fanteam_data_file):
                            fanteam_csv = pd.read_csv(fanteam_data_file)
                            pl_projections['Lineup'] = pl_projections['FanTeam ID'].map(fanteam_csv.set_index('PlayerID')['Lineup'])
                            pl_projections['Price'] = pl_projections['FanTeam ID'].map(fanteam_csv.set_index('PlayerID')['Price'])
                            ft_temp = pl_projections[pl_projections['Lineup'].isin(['expected', 'possible'])]
                        else:
                            pl_projections['Price'] = 0
                            ft_temp = pl_projections.copy()
                        ft_temp = ft_temp[ft_temp['FanTeam Position'].notna()].reset_index(drop=True)
                        fanteam_df = get_fanteam_points(ft_temp, score_preds, fanteam_points_dict_gk,
                                                        fanteam_points_dict_def, fanteam_points_dict_mid, fanteam_points_dict_fwd)
                        fanteam_df.dropna(inplace=True)
                        logger.info(f"[{league}] Inserting FanTeam projections into DB ({len(fanteam_df)} rows)...")
                        _t = time.time()
                        await insert_fanteam_projections_async(fanteam_df)
                        logger.info(f"[{league}] FanTeam projections inserted ({time.time()-_t:.1f}s)")
                    except Exception as e:
                        logger.warning(f"[{league}] FanTeam computation failed (skipping): {e}", exc_info=True)

                # ## **Player Stat Probabilities**
                #
                # Using Poisson Distribution to get the likelihood of players acheiving certain statistics.

                # In[ ]:

                pl_projections.rename(columns={'Fouls': 'Fouls Committed'}, inplace=True)

                # In[ ]:

                perc_stats = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn']
                lines = [1, 2, 3]

                # In[ ]:

                player_stat_probs = get_poisson_probs(pl_projections, perc_stats, lines)
                player_stat_probs = player_stat_probs.round(2)
                # player_stat_probs.to_csv(rf"{save_file_path}\{league} Player Stat Probabilities.csv", index=False)
                player_stat_probs.to_csv(f"{save_file_path}/{league} Player Stat Probabilities.csv", index=False)
                # await insert_players_stats_async(pl_projections)
                logger.info(f"[{league}] Inserting player stat probabilities into DB...")
                _t = time.time()
                await insert_players_stats_async(player_stat_probs, teams=teams)
                logger.info(f"[{league}] Player stat probs inserted ({time.time()-_t:.1f}s)")
                _league_elapsed = (time.time() - _start_time) / 60
                _league_times[league] = _league_elapsed
                logger.info(f"[{league}] COMPLETE - {_league_elapsed:.1f} min")
            except Exception as e:
                logger.error(f"[{league}] FAILED - skipping: {e}", exc_info=True)
                _league_times[league] = "FAILED"

        _total_elapsed = (time.time() - _total_start) / 60
        logger.info("=" * 60)
        logger.info("ALL LEAGUES COMPLETE - SUMMARY:")
        for _l, _t in _league_times.items():
            _t_str = f"{_t:.1f} min" if isinstance(_t, float) else _t
            logger.info(f"  {_l:<30} {_t_str}")
        logger.info(f"  {'TOTAL':<30} {_total_elapsed:.1f} min")
        logger.info("=" * 60)
