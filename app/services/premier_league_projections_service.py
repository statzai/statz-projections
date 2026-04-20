#!/usr/bin/env python
# coding: utf-8
from app.repository.fanteam_repo import insert_fanteam_projections_async
from app.repository.fpl_repo import insert_fpl_projections_async
from app.repository.opta_repo import insert_opta_projections_async
# In[1]:


# %pip install pandas==2.2.1
# %pip install numpy==1.26.4
# %pip install scipy
# %pip install pickle
# %pip install warnings
# %pip install scikit-learn==1.2.2


# In[2]:
from app.repository.player_stat_repo import insert_players_stats_async
from scipy.stats import poisson
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from pathlib import Path
from app.services.statz_functions import get_poisson_probs, get_dream11_points, get_draftkings_points, \
    get_fanteam_points, get_fpl_points, bonus_points_score, get_bonus_points, distribute_team_predictions_to_players, \
    adjust_shots_projection, get_stat_id, get_team_round_predictions, get_stat_list, \
    get_avg_table_with_probs_and_point_limits, get_avg_table_with_probs, sim_multiple_seasons, \
    make_round_goal_prediction, get_away_goal_avg, get_home_goal_avg, get_team, get_stage_id, get_market_value, \
    get_ratings, get_team_id, get_league_id, get_season_id, get_opta_points, get_result_probs, find_inputs_for_probs, \
    get_player_id, get_extra_stats, get_player_position


# # **DOWNLOAD THE FOLLOWING CSV FILES**
#
# - Fixture Player Stats
# - Fixture Team Stats
# - Standings
# - Fixtures
# - Players
# - Competitions
# - Teams
# - Stats Types
# - Competition Season Teams

# In[3]:


# CHANGE THIS TO WHERE YOU SAVED THE DATA

# data_folder_path = r"C:\Users\George\Documents\Statz.ai\Data"

# CHANGE THIS TO WHERE YOU SAVED THE MODEL BUILDS

# model_file_path = r"C:\Users\George\Documents\Statz.ai\Notebooks\Projections\Model Builds"

# CHANGE THIS TO WHERE YOU WANT OUTPUTS SAVED

# save_file_path = r"C:\Users\George\Documents\Statz.ai\Projections to Server Version 2\Projection Outputs"

# CHANGE THIS TO WHERE THE MODEL DATASETS ARE SAVED

# model_dataset_file_path = r"C:\Users\George\Documents\Statz.ai\Projections to Server Version 2\Model Datasets"

# CHANGE THIS TO WHERE THE ACCURACY DATASETS ARE SAVED

# accuracy_dataset_file_path = r"C:\Users\George\Documents\Statz.ai\Projections to Server Version 2\Accuracy Datasets"


class PremierLeagueProjectionsService:

    async def projections(self):
        CURRENT_DIR = Path(__file__).resolve().parent
        APP_DIR = CURRENT_DIR.parent

        data_folder_path = APP_DIR / "data"
        model_file_path = APP_DIR / "model-builds"
        save_file_path = APP_DIR / "projection-outputs"
        # model_dataset_file_path = APP_DIR / "Model Datasets"
        # accuracy_dataset_file_path = APP_DIR / "Accuracy Datasets"

        # # **Inputs**
        #
        # Only thing to change here is the league

        # In[4]:


        league = 'Premier League'

        fanteam_csv_imported = 'Yes'

        # In[5]:


        league_dashed = league.replace(' ', '-').replace('.', '').lower()

        file_path = os.path.join(data_folder_path, "League Weightings.xlsx")
        league_weightings_df = pd.read_excel(file_path)
        league_below = league_weightings_df[league_weightings_df['League'] == league]['League Below'].values[0]
        league_above = league_weightings_df[league_weightings_df['League'] == league]['League Above'].values[0]
        league_below_attack_weight = \
        league_weightings_df[league_weightings_df['League'] == league]['League Below Attack Weight'].values[0]
        league_below_defense_weight = \
        league_weightings_df[league_weightings_df['League'] == league]['League Below Defense Weight'].values[0]
        league_above_attack_weight = \
        league_weightings_df[league_weightings_df['League'] == league]['League Above Attack Weight'].values[0]
        league_above_defense_weight = \
        league_weightings_df[league_weightings_df['League'] == league]['League Above Defense Weight'].values[0]
        country_code = league_weightings_df[league_weightings_df['League'] == league]['code'].values[0]
        div = league_weightings_df[league_weightings_df['League'] == league]['div'].values[0]
        weightings = [league_above_attack_weight, league_above_defense_weight, league_below_attack_weight,
                      league_below_defense_weight]
        mv_beta = league_weightings_df[league_weightings_df['League'] == league]['mv_beta'].values[0]
        odds_beta = league_weightings_df[league_weightings_df['League'] == league]['odds_beta'].values[0]

        # # **Load Data**
        #
        # Get Data from Previous Weeks Results + Next Weeks Fixtures (maybe add something to check the data has actually been uploaded)

        # In[6]:


        # Model Dataset
        file_path = os.path.join(data_folder_path, "all_leagues_model_dataset_with_history.xlsx")
        model_dataset_all = pd.read_excel(file_path)
        file_path = os.path.join(data_folder_path, f"{league}_model_dataset_with_history.xlsx")
        model_dataset_league = pd.read_excel(file_path)

        # Accuracy Dataset
        file_path = os.path.join(data_folder_path, f"{league}_accuracy_dataset.xlsx")
        projection_accuracy_dataset_league = pd.read_excel(file_path)
        file_path = os.path.join(data_folder_path, "all_leagues_accuracy_dataset.xlsx")
        projection_accuracy_dataset_all = pd.read_excel(file_path)

        # Ratings Dataset
        file_path = os.path.join(data_folder_path, "Team Ratings.xlsx")
        all_team_ratings = pd.read_excel(file_path)
        all_team_ratings['Date'] = all_team_ratings['Date'].dt.date

        # In[7]:

        file_path = os.path.join(data_folder_path, "fixture_player_stats.csv")
        player_stats = pd.read_csv(file_path)
        player_stats.drop_duplicates(subset=['fixture_id', 'player_id', 'stats_type_id'], inplace=True)
        file_path = os.path.join(data_folder_path, "fixture_team_stats.csv")
        team_stats = pd.read_csv(file_path)
        team_stats.drop_duplicates(subset=['fixture_id', 'team_id', 'stats_type_id'], inplace=True)
        file_path = os.path.join(data_folder_path, "standings.csv")
        standings = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "seasons.csv")
        seasons = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "competitions.csv")
        comps = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "competition_season_teams.csv")
        comp_teams = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "teams.csv")
        teams = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "players.csv")
        players = pd.read_csv(file_path)
        players['display_name'] = players['display_name'].str.strip()
        file_path = os.path.join(data_folder_path, "fixtures.csv")
        fixtures_df = pd.read_csv(file_path)
        fixtures_df.drop_duplicates(
            subset=['season_id', 'home_team_id', 'away_team_id', 'home_team_goals', 'away_team_goals', 'kickoff_datetime'],
            inplace=True)
        file_path = os.path.join(data_folder_path, "bet365_odds.csv")
        b365_odds = pd.read_csv(file_path)
        b365_odds = b365_odds.drop_duplicates(subset=['fixture_id', 'name'], keep='last')
        fixtures_df['over_1_5_odds_decimal'] = fixtures_df['id'].map(
            b365_odds[b365_odds['name'] == 'OVER_1_5'].set_index('fixture_id')['odd_decimal'])
        fixtures_df['over_2_5_odds_decimal'] = fixtures_df['id'].map(
            b365_odds[b365_odds['name'] == 'OVER_2_5'].set_index('fixture_id')['odd_decimal'])
        file_path = os.path.join(data_folder_path, "stats_types.csv")
        stats_types = pd.read_csv(file_path)
        league_id = get_league_id(league, comps)
        fixtures = fixtures_df[fixtures_df['competition_id'] == league_id]
        league_standings = standings[standings['competition_id'] == league_id]
        league_above_id = None
        league_below_id = get_league_id(league_below, comps)
        previous_season_id = get_season_id(league_id, seasons, True)
        current_season_id = get_season_id(league_id, seasons, False)
        standings = standings[standings['season_id'] == current_season_id]
        matches_played = standings['played'].max()
        previous_season_id_below = get_season_id(league_below_id, seasons, True) if league_below_id else None
        previous_season_id_above = get_season_id(league_above_id, seasons, True) if league_above_id else None
        stat_list = get_stat_list()


        # In[8]:


        def change_player_team(players_df, player_id, new_team_id):
            players_df.loc[players_df['id'] == player_id, 'current_team_id'] = new_team_id
            return players_df


        # In[9]:


        transferred_players_dict = {
            '37288979': '8',
            '530762': '19',
            '194167': '19',
            '25217662': '19',
            '37592228': '19',
            '31609': '19',
            '186606': '19',
            '538472': '52',
            '28543553': '52',
            '37575228': '236',
            '23269659': '78',
            '37423141': '27',
            '37337041': '27',
            '37316840': '18',
            '28912976': '18',
            '37590847': '13',
            '540613': '13',
            '2178735': '71',
            '460159': '71',
            '173737': '71',
            '28575686': '9',
            '21072805': '9',
            '529689': '14',
            '28912805': '20',
            '19978623': '63',
            '37555816': '3',
            '37590634': '3',
            '37630179': '3',
            '1887': '3',
            '37397144': '3',
            '5640849': '6',
            '160072': '6',
            '37685630': '1',
            '9438': '20',
            '911': '27'}

        # In[10]:


        df = pd.DataFrame()
        df['Player ID'] = transferred_players_dict.keys()
        df['Team ID'] = transferred_players_dict.values()
        df['Team ID'] = df['Team ID'].astype(int)
        df['Player ID'] = df['Player ID'].astype(int)
        df['Player Name'] = df['Player ID'].map(players.set_index('id')['display_name'])

        # In[11]:


        for player_id, new_team_id in df[['Player ID', 'Team ID']].itertuples(index=False):
            players = change_player_team(players, player_id, new_team_id)

        # ## **Get Previous Weeks Data**

        # ## For Model Dataset

        # In[12]:


        model_dataset_league['comp_id'] = league_id
        previous_fixtures = model_dataset_league[model_dataset_league.isnull().any(axis=1)]
        for i in range(len(previous_fixtures)):
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
                model_dataset_all.loc[
                    (model_dataset_all['id'] == fixture_id) & (model_dataset_all['Team'] == team), 'Team ' + stat] = stat_value

        # ## For Accuracy Dataset

        # In[13]:


        previous_accuracy_fixtures = projection_accuracy_dataset_league[projection_accuracy_dataset_league.isnull().any(axis=1)]
        for i in range(len(previous_accuracy_fixtures)):
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

        # In[14]:


        def grid_search(trainX, trainY):
            param_grid = {
                'alpha': np.arange(0, 1, 0.1),
                'max_iter': [100, 200, 500],
                'fit_intercept': [True, False]
            }
            pr = PoissonRegressor()
            gs = GridSearchCV(pr, param_grid, cv=5, scoring='neg_mean_squared_error')
            model = gs.fit(trainX, trainY)
            return model


        def fit_model(trainX, trainY):
            model = PoissonRegressor(solver='newton-cholesky')
            model.fit(trainX, trainY)
            return model


        # In[15]:


        league_training_dataset = model_dataset_league.dropna().copy()
        all_league_training_dataset = model_dataset_all.dropna().copy()

        for stat in stat_list:
            if stat == 'Goals':
                continue
            predictors = ['Team ' + stat + ' History',
                          'Opponent ' + stat + ' History Against']
            target = 'Team ' + stat
            X = league_training_dataset[predictors]
            y = league_training_dataset[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            if stat in ['Passes', 'Successful Passes']:
                model = fit_model(X_train, y_train)
            else:
                model = grid_search(X_train, y_train)

            league_dir = model_file_path / league
            league_dir.mkdir(parents=True, exist_ok=True)

            model_path = league_dir / f"{league}_{stat}_model.sav"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            X_all = all_league_training_dataset[predictors]
            y_all = all_league_training_dataset[target]
            X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all)
            model_all = grid_search(X_train_all, y_train_all)

            all_leagues_dir = model_file_path / "All Leagues"
            all_leagues_dir.mkdir(parents=True, exist_ok=True)

            model_path = all_leagues_dir / f"All_Leagues_{stat}_model.sav"

            with open(model_path, "wb") as f:
                pickle.dump(model_all, f)

            # pickle.dump(model_all, open(rf"{model_file_path}\All Leagues\All_Leagues_{stat}_model.sav", 'wb'))

        # ## **Re-Calculate Accuracy**

        # ## Team Stat Accuracy

        # In[16]:


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
            d['Average Team Abs Error'] = (d['Home Team Abs Error'] + d['Away Team Abs Error']) / 2
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
                'Team Abs Error': d['Average Team Abs Error'].mean()
            }


        accuracy_df_league = pd.DataFrame([summarize(projection_accuracy_dataset_league.dropna(), stat) for stat in stat_list])
        accuracy_df_all = pd.DataFrame([summarize(projection_accuracy_dataset_all.dropna(), stat) for stat in stat_list])
        accuracy_df_league = accuracy_df_league.round(2)
        accuracy_df_all = accuracy_df_all.round(2)
        file_path = os.path.join(data_folder_path, f"{league} Projection Accuracy.csv")
        accuracy_df_league.to_csv(file_path, index=False)
        file_path = os.path.join(data_folder_path, "All Leagues Projection Accuracy.csv")
        accuracy_df_all.to_csv(file_path, index=False)

        # ## Fixture Result Accuracy

        # ## **Team Ratings**
        #
        # Team Ratings are calculated by combining a weighted average of Actual Goals (30%) and Expected Goals (70%) over the last 50 games.

        # In[ ]:


        ratings = get_ratings(league_id=league_id, previous_team_ratings=all_team_ratings, current_season_id=current_season_id,
                              all_season_ids=[current_season_id, previous_season_id, previous_season_id_above,
                                              previous_season_id_below],
                              comp_teams=comp_teams, teams_df=teams, fixtures_df=fixtures_df, team_stats=team_stats,
                              stats_types=stats_types, weight=0.96, games=30, weightings=weightings)
        ratings.to_csv(f"{save_file_path}/{league} Get Ratings.csv", index=False)
        # In[ ]:


        team_mapping = {
            'Bournemouth': 'AFC Bournemouth'
        }


        # In[19]:


        def rescale_to_range(series, new_min=0.5, new_max=2.0):
            old_min = series.min()
            old_max = series.max()
            return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)


        # In[ ]:


        market_values = get_market_value(league_dashed, div, country_code)
        market_values['MV Index'] = market_values['Market Value'].astype(float) / market_values['Market Value'].astype(
            float).median()
        market_values['MV Index'] = np.log1p(market_values['MV Index'])
        market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()
        max = market_values['MV Index'].max() if market_values['MV Index'].max() < 2.0 else 2.0
        min = market_values['MV Index'].min() if market_values['MV Index'].min() > 0.5 else 0.5
        market_values['MV Index'] = rescale_to_range(market_values['MV Index'], min, max)
        market_values['MV Index'] = market_values['MV Index'] / market_values['MV Index'].mean()
        market_values['Team'] = market_values['Team'].replace(team_mapping)
        market_values['Team'] = market_values['Team'].str.strip()

        ratings['Team'] = ratings['Team'].str.strip()
        ratings['MV Index'] = ratings['Team'].map(market_values.set_index('Team')['MV Index'])
        ratings['MV Index Reverse'] = (ratings['MV Index'].mean() / ratings['MV Index'])
        ratings['MV Index Reverse'] = ratings['MV Index Reverse'] / ratings['MV Index Reverse'].mean()

        teams_to_map = ratings.loc[ratings['MV Index'].isna(), 'Team']

        if len(teams_to_map) > 0:
            logger.warning('Statz Team Names to Map:')
            logger.warning(teams_to_map.to_string(index=False))
            market_values_not_mapped = market_values[~market_values['Team'].isin(ratings['Team'])]
            # debug print removed
            logger.warning(market_values_not_mapped['Team'].to_string(index=False))
            raise ValueError('Mapping Error for the teams above. Please Update the team_mapping dictionary in the code.')

        mv_beta = (mv_beta * (0.95 ** (matches_played)))
        ratings['MV Attack Underperformance'] = (ratings['MV Index'] - ratings['Attack'] / ratings['Attack'].mean()) * mv_beta
        ratings['MV Attack Underperformance %'] = ratings['MV Attack Underperformance'] / ratings['Attack']
        ratings['MV Defense Underperformance'] = (ratings['MV Index Reverse'] - ratings['Defense'] / ratings[
            'Defense'].mean()) * mv_beta
        ratings['MV Defense Underperformance %'] = ratings['MV Defense Underperformance'] / ratings['Defense']
        ratings['Attack'] = ratings['Attack'] * (1 + ratings['MV Attack Underperformance %'])
        ratings['Defense'] = ratings['Defense'] * (1 + ratings['MV Defense Underperformance %'])
        ratings.drop(
            columns=['MV Defense Underperformance', 'MV Attack Underperformance', 'MV Index', 'MV Defense Underperformance %',
                     'MV Attack Underperformance %', 'MV Index Reverse'], inplace=True)

        # In[21]:


        # Readjust so that 100 is the mean for Attack, Defense, and Overall
        for col in ['Attack', 'Defense']:
            ratings[col] = ratings[col] / ratings[col].mean() * 100
        ratings['Overall'] = ratings['Attack'] - ratings['Defense']
        ratings.sort_values('Overall', ascending=False, inplace=True)
        ratings.reset_index(drop=True, inplace=True)
        ratings = ratings.round(1)
        if league == 'Premier League':
            season_teams = comp_teams[comp_teams['season_id'] == current_season_id]
            teams_in_league = teams[teams['id'].isin(season_teams[season_teams['competition_id'] == league_id]['team_id'])]
            fixture_ticker_ratings = ratings.copy()
            fixture_ticker_ratings['Team ID'] = fixture_ticker_ratings['Team'].map(teams_in_league.set_index('name')['id'])
            fixture_ticker_ratings['Attack (Home)'] = fixture_ticker_ratings['Attack'] * 0.9
            fixture_ticker_ratings['Defense (Home)'] = fixture_ticker_ratings['Defense'] * 1.1
            fixture_ticker_ratings['Overall (Home)'] = fixture_ticker_ratings['Attack (Home)'] - fixture_ticker_ratings[
                'Defense (Home)']
            fixture_ticker_ratings['Attack (Away)'] = fixture_ticker_ratings['Attack'] * 1.1
            fixture_ticker_ratings['Defense (Away)'] = fixture_ticker_ratings['Defense'] * 0.9
            fixture_ticker_ratings['Overall (Away)'] = fixture_ticker_ratings['Attack (Away)'] - fixture_ticker_ratings[
                'Defense (Away)']
            fixture_ticker_ratings = fixture_ticker_ratings[['Team', 'Team ID', 'Overall (Home)', 'Overall (Away)']]
            fixture_ticker_ratings = fixture_ticker_ratings.round(0).astype({'Overall (Home)': 'int', 'Overall (Away)': 'int'})
            file_path = os.path.join(data_folder_path, f"{league} Fixture Ticker Ratings.csv")
            fixture_ticker_ratings.to_csv(file_path, index=False)
        ratings['Rank'] = ratings.index + 1
        old_ratings = all_team_ratings[all_team_ratings['League'] == league]
        old_ratings = old_ratings[old_ratings['Date'] == old_ratings['Date'].max()]
        old_ratings.reset_index(drop=True, inplace=True)
        old_ratings['Rank'] = old_ratings.index + 1
        for i in range(len(ratings)):
            team = ratings.loc[i, 'Team']
            old_rank = old_ratings.loc[old_ratings['Team'] == team, 'Rank'].values[0]
            new_rank = ratings.loc[i, 'Rank']
            ratings.loc[i, 'Movement'] = old_rank - new_rank
        ratings = ratings[['Team', 'Attack', 'Defense', 'Overall', 'Movement']]

        # In[22]:


        ratings['Date'] = pd.to_datetime('today').date()
        ratings['League'] = league
        all_team_ratings = pd.concat([all_team_ratings, ratings], ignore_index=True)
        all_team_ratings.drop_duplicates(subset=['Team', 'League', 'Date'], keep='last', inplace=True)
        all_team_ratings.reset_index(drop=True, inplace=True)
        file_path = os.path.join(data_folder_path, "Team Ratings.xlsx")
        all_team_ratings.to_excel(file_path, index=False)


        # ## **Make Predictions for Next Fixture Round**
        #
        # Result, Score, Clean Sheets, Over 1.5, Over 2.5 and BTTS all calculated here using Poisson Distribution.

        # In[23]:


        def get_fixtures(fixtures, teams, previous=False, odds=True, cup=False, leg=None, round_id=None, number_of_rounds=1):
            if cup == True:
                stage_id = get_stage_id(fixtures, previous)
                fixtures = fixtures[fixtures['stage_id'] == stage_id]
                if leg != None:
                    fixtures = fixtures[fixtures['leg'] == f'{leg}/2']
            else:
                if round_id == None:
                    round_id = get_round_id(fixtures, previous, number_of_rounds)
                    try:
                        fixtures = fixtures[fixtures['round_id'].isin(round_id)]
                    except:
                        fixtures = fixtures[fixtures['round_id'] == round_id]
                else:
                    fixtures = fixtures[fixtures['round_id'] == round_id]
            fixtures = fixtures[['id', 'kickoff_datetime', 'name', 'home_team_id', 'away_team_id', 'bet365_home_odds_decimal',
                                 'bet365_draw_odds_decimal', 'bet365_away_odds_decimal']]
            fixtures['home_team'] = fixtures['home_team_id'].apply(lambda x: get_team(x, teams))
            fixtures['away_team'] = fixtures['away_team_id'].apply(lambda x: get_team(x, teams))
            fixtures = fixtures[
                ['id', 'kickoff_datetime', 'home_team', 'away_team', 'bet365_home_odds_decimal', 'bet365_draw_odds_decimal',
                 'bet365_away_odds_decimal']]
            fixtures.sort_values(by=['kickoff_datetime', 'home_team'], inplace=True)
            return fixtures.reset_index(drop=True)


        def get_round_id(fixtures, previous=False, number_of_rounds=1):
            import pandas as pd
            date = pd.to_datetime('today')
            fixtures.loc[:, 'kickoff_datetime'] = pd.to_datetime(fixtures['kickoff_datetime'])
            if previous == True:
                fixtures = fixtures[fixtures['kickoff_datetime'] < date].reset_index(drop=True)
                fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=False)
            else:
                fixtures = fixtures[fixtures['kickoff_datetime'] > date].reset_index(drop=True)
                fixtures = fixtures.sort_values(by='kickoff_datetime', ascending=True)

            if number_of_rounds > 1:
                round_id = fixtures['round_id'].unique()[:number_of_rounds]
            else:
                round_id = fixtures['round_id'].iloc[0]
            return round_id


        # In[24]:


        next_fix = get_fixtures(fixtures, teams, False, number_of_rounds=1)
        next_6_fix = get_fixtures(fixtures, teams, False, number_of_rounds=6)

        # In[ ]:


        avg_home_goals = get_home_goal_avg(league_id, team_stats, fixtures, stats_types)
        avg_away_goals = get_away_goal_avg(league_id, team_stats, fixtures, stats_types)
        score_preds = make_round_goal_prediction(next_6_fix, ratings, avg_home_goals, avg_away_goals)
        boost = 1.1
        score_preds['Home Odds %'] = ((1 / next_6_fix['bet365_home_odds_decimal']) * 100)
        score_preds['Draw Odds %'] = ((1 / next_6_fix['bet365_draw_odds_decimal']) * 100)
        score_preds['Away Odds %'] = ((1 / next_6_fix['bet365_away_odds_decimal']) * 100)

        home_win = []
        draw = []
        away_win = []
        home_clean = []
        away_clean = []
        over_1 = []
        over_2 = []
        btts = []
        for i in range(len(score_preds)):
            bookie_margin = 1 + (score_preds.loc[i, 'Home Odds %'] + score_preds.loc[i, 'Draw Odds %'] + score_preds.loc[
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
                                                                       adjusted_draw_prob, adjusted_away_win_prob, boost)
                score_preds.loc[i, 'Home Goals'] = round(new_home_goals, 2)
                score_preds.loc[i, 'Away Goals'] = round(new_away_goals, 2)
                home_clean_sheet = poisson.pmf(0, new_away_goals)
                away_clean_sheet = poisson.pmf(0, new_home_goals)
            else:
                new_home_goals = home_goals
                new_away_goals = away_goals
                adjusted_home_win_prob, adjusted_draw_prob, adjusted_away_win_prob = get_result_probs(home_goals, away_goals,
                                                                                                      boost)
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
        score_preds_with_odds = score_preds.copy()
        score_preds.drop(columns=['Home Odds %', 'Draw Odds %', 'Away Odds %'], inplace=True)

        score_preds_next_fix = score_preds[score_preds['id'].isin(next_fix['id'])]
        score_preds_next_fix_with_odds = score_preds_with_odds[score_preds_with_odds['id'].isin(next_fix['id'])]
        file_path = os.path.join(data_folder_path, f"{league} Fixtures.csv")
        score_preds_next_fix.to_csv(file_path, index=False)

        # In[26]:


        score_preds_next_fix_with_odds.rename(
            columns={'id': 'fixture_id', 'Home Goals': 'Home Projected Goals', 'Away Goals': 'Away Projected Goals'},
            inplace=True)
        score_preds_next_fix_with_odds['Total Projected Goals'] = score_preds_next_fix_with_odds['Home Projected Goals'] + \
                                                                  score_preds_next_fix_with_odds['Away Projected Goals']
        score_preds_next_fix_with_odds['comp_id'] = league_id
        projection_accuracy_dataset_league = pd.concat([projection_accuracy_dataset_league, score_preds_next_fix_with_odds],
                                                       ignore_index=True)
        score_preds_next_fix_with_odds.rename(
            columns={'fixture_id': 'id', 'Home Projected Goals': 'Home Goals', 'Away Projected Goals': 'Away Goals'},
            inplace=True)
        score_preds_next_fix_with_odds.drop(columns=['comp_id', 'Total Projected Goals'], inplace=True)

        # ## **4+ Star Bets**

        # In[27]:

        file_path = os.path.join(data_folder_path, "Best Bets.xlsx")
        best_bets = pd.read_excel(file_path)

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
            over_1_5_goals_total_rating = (over_1_5_goals_edge_rating * 0.7 if over_1_5_goals_edge_rating > 0 else 0) + (
                over_1_5_goals_prob_rating * 0.3 if over_1_5_goals_prob_rating < 5 else 5 * 0.3)
            over_2_5_goals_total_rating = (over_2_5_goals_edge_rating * 0.7 if over_2_5_goals_edge_rating > 0 else 0) + (
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
                        'Price': [round(1 / locals()[bet_type.lower().replace(' ', '_').replace('.', '_') + '_odds'], 2)]
                    })], ignore_index=True)

        best_bets = pd.concat([best_bets, new_best_bets], ignore_index=True)
        best_bets.drop_duplicates(subset=['Date', 'Competition', 'Home Team', 'Away Team', 'Bet Type'], keep='last',
                                  inplace=True)
        file_path = os.path.join(data_folder_path, "Best Bets.xlsx")
        best_bets.to_excel(file_path, index=False)

        # # **League Projections**

        # In[ ]:


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

        current_standings = standings.copy()
        current_standings['Team'] = current_standings['team_id'].map(teams.set_index('id')['name'])
        current_standings.rename(columns={'goals_for': 'Goals For', 'goals_against': 'Goals Against', 'points': 'Points'},
                                 inplace=True)
        current_standings['Goal Difference'] = current_standings['Goals For'] - current_standings['Goals Against']
        current_standings = current_standings[['Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference']]
        current_standings.reset_index(drop=True, inplace=True)
        current_standings = current_standings.astype(
            {'Points': 'int', 'Goals For': 'int', 'Goals Against': 'int', 'Goal Difference': 'int'})
        current_league_table = {team: {'Points': points, 'Goals For': gf, 'Goals Against': ga, 'Goal Difference': gd} for
                                team, points, gf, ga, gd in current_standings.values}

        avg_table, all_tables = sim_multiple_seasons(season_score_preds, current_league_table, num_sims=10000)

        avg_table_with_probs = get_avg_table_with_probs(league, avg_table, all_tables)
        avg_table_with_probs_and_point_limits = get_avg_table_with_probs_and_point_limits(avg_table_with_probs, all_tables)
        file_path = os.path.join(data_folder_path, f"{league} Predicted Table.csv")
        avg_table_with_probs_and_point_limits.to_csv(file_path, index=False)

        # # **Team Projections**
        #
        # Getting each Teams stat projections using the models

        # In[29]:


        stat_list = get_stat_list()


        # In[30]:


        # def load_all_models(stat_list, file_path, league):
        #     models = {}
        #     for stat in stat_list:
        #         model = load_model(stat, file_path, league)
        #         models[stat] = model
        #     return models


        # def load_model(stat, file_path, league):
        #     import pickle
        #     if stat == 'Goals':
        #         return None
        #     filename = file_path + '\\' + league + '\\' + league + '_' + stat + '_model.sav'
        #     try:
        #         model = pickle.load(open(filename, 'rb'))
        #     except:
        #         model = pickle.load(open(rf"{file_path}\All Leagues\All_Leagues_{stat}_model.sav", 'rb'))
        #     return model

        def load_all_models(stat_list, file_path, league):  # UPDATED - New Parameter: league
            models = {}
            for stat in stat_list:
                model = load_model(stat, file_path, league)  # UPDATED - Pass league parameter
                models[stat] = model
            return models

        def load_model(stat, file_path, league):
            if stat == 'Goals':
                return None

            filename = os.path.join(file_path, league, f"{league}_{stat}_model.sav")

            try:
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
            except:
                fallback_path = os.path.join(file_path, "All Leagues", f"All_Leagues_{stat}_model.sav")
                with open(fallback_path, 'rb') as f:
                    model = pickle.load(f)

            return model

        # In[31]:


        models = load_all_models(stat_list, model_file_path, league)

        # In[32]:


        todays_date = pd.to_datetime(next_fix['kickoff_datetime'].iloc[0]).date()

        # In[33]:


        team_projections = get_team_round_predictions(next_6_fix, stat_list, fixtures_df, team_stats, teams, stats_types,
                                                      models, league_weightings=[league_above_attack_weight,
                                                                                 league_above_defense_weight,
                                                                                 league_below_attack_weight,
                                                                                 league_below_defense_weight],
                                                      season_id=[current_season_id, previous_season_id,
                                                                 previous_season_id_above, previous_season_id_below], games=50)

        # In[34]:


        new_rows = []

        team_projections_next_fix = team_projections[team_projections['fixture_id'].isin(next_fix['id'])]

        for i in range(len(team_projections_next_fix)):
            team_df = team_projections_next_fix.iloc[[i]]
            new_row = {}
            new_row['id'] = team_df['fixture_id'].values[0]
            new_row['kickoff_datetime'] = team_df['kickoff_datetime'].values[0]
            new_row['comp_id'] = league_id
            new_row['Team'] = team_df['Team'].values[0]
            new_row['Opponent'] = team_df['Opponent'].values[0]
            new_row['Venue'] = team_df['Venue'].values[0]
            for stat in stat_list:
                new_row['Team ' + stat + ' History'] = team_df['Team ' + stat + ' History'].values[0]
                new_row['Opponent ' + stat + ' History Against'] = team_df['Opponent ' + stat + ' History Against'].values[0]
            new_rows.append(new_row)

        model_dataset_league = pd.concat([model_dataset_league, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_all = pd.concat([model_dataset_all, pd.DataFrame(new_rows)], ignore_index=True)
        model_dataset_league.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        model_dataset_all.drop_duplicates(subset=['id', 'Team', 'Opponent', 'Venue'], keep='last', inplace=True)
        file_path = os.path.join(data_folder_path, f"{league}_model_dataset_with_history.xlsx")
        # model_dataset_league.to_excel(rf"{data_folder_path}\{league}_model_dataset_with_history.xlsx", index=False)
        model_dataset_league.to_excel(file_path, index=False)
        file_path = os.path.join(data_folder_path, "all_leagues_model_dataset_with_history.xlsx")
        model_dataset_all.to_excel(file_path, index=False)

        # In[35]:


        # Bake Goals into shots projections
        avg_goals = (avg_home_goals + avg_away_goals) / 2

        league_team_stats = team_stats[
            team_stats['fixture_id'].isin(fixtures_df[fixtures_df['competition_id'] == league_id]['id'])]

        league_shots = league_team_stats[league_team_stats['stats_type_id'] == get_stat_id('Shots Total', stats_types)].copy()
        league_shots['Date'] = league_shots['fixture_id'].map(fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(league_shots['Date'])).dt.days // 7
        league_shots['Weight'] = 0.9 ** (league_shots['Weeks Since Kickoff'] - 5)
        league_shots.loc[league_shots['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots['Weighted Shots'] = league_shots['Weight'] * league_shots['value']
        avg_shots = league_shots['Weighted Shots'].sum() / league_shots['Weight'].sum()

        league_shots_on_target = league_team_stats[
            league_team_stats['stats_type_id'] == get_stat_id('Shots On Target', stats_types)].copy()
        league_shots_on_target['Date'] = league_shots_on_target['fixture_id'].map(
            fixtures_df.set_index('id')['kickoff_datetime'])
        league_shots_on_target['Weeks Since Kickoff'] = (pd.to_datetime('now') - pd.to_datetime(
            league_shots_on_target['Date'])).dt.days // 7
        league_shots_on_target['Weight'] = 0.9 ** (league_shots_on_target['Weeks Since Kickoff'] - 5)
        league_shots_on_target.loc[league_shots_on_target['Weeks Since Kickoff'] < 6, 'Weight'] = 1
        league_shots_on_target['Weighted Shots On Target'] = league_shots_on_target['Weight'] * league_shots_on_target['value']
        avg_shots_on_target = league_shots_on_target['Weighted Shots On Target'].sum() / league_shots_on_target['Weight'].sum()

        avg_shots_per_goal = avg_shots / avg_goals
        avg_shots_on_target_per_goal = avg_shots_on_target / avg_goals

        # In[36]:


        # if 'team_projections' in globals():
        goals = []
        assists = []
        for i in range(len(team_projections)):
            team = team_projections['Team'].iloc[i]
            opp = team_projections['Opponent'].iloc[i]
            fixture = score_preds[score_preds['id'] == team_projections['fixture_id'].iloc[i]]
            team_pred = fixture['Home Goals'].values[0] if fixture['Home Team'].values[0] == team else \
            fixture['Away Goals'].values[0]
            opp_pred = fixture['Away Goals'].values[0] if fixture['Home Team'].values[0] == opp else \
            fixture['Home Goals'].values[0]
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
            fixture_id = team_projections['fixture_id'].iloc[i]
            fixture_team_projections = team_projections[team_projections['fixture_id'] == fixture_id]
            fixture_team_projections = fixture_team_projections.drop(i)
            saves.append(
                fixture_team_projections['Shots On Target'].values[0] - fixture_team_projections['Goals'].values[0])

        team_projections['Saves'] = saves
        team_projections['Saves'] = team_projections['Saves'].round(2)
        team_projections['Key Passes'] = (team_projections['Shots Total'] * 0.75).round(2)
        team_projections = team_projections[
            ['fixture_id', 'kickoff_datetime', 'Team', 'Opponent', 'Venue', 'Goals', 'Assists',
             'Key Passes'] + stat_list + ['Fouls Drawn', 'Saves']]
        team_projections.rename(columns={'Successful Passes': 'Accurate Passes'}, inplace=True)

        # In[37]:


        team_projections_save = team_projections.copy()
        team_projections_save.drop(['Assists', 'Fouls Drawn', 'Saves', 'Key Passes'], axis=1, inplace=True)
        team_projections_save = team_projections_save.round(2)
        team_projections_save_next_fix = team_projections_save[team_projections_save['fixture_id'].isin(next_fix['id'])]
        file_path = os.path.join(data_folder_path, f"{league} Team.csv")
        team_projections_save_next_fix.to_csv(file_path, index=False)

        # In[38]:


        team_projections_save_next_fix.rename(columns={'Accurate Passes': 'Successful Passes'}, inplace=True)

        # In[39]:


        for fixture_id in team_projections_save_next_fix['fixture_id'].unique():
            fixture_projections = team_projections_save_next_fix[team_projections_save_next_fix['fixture_id'] == fixture_id]
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
        file_path = os.path.join(data_folder_path, f"{league}_accuracy_dataset.xlsx")
        projection_accuracy_dataset_league.to_excel(file_path, index=False)
        projection_accuracy_dataset_all = pd.concat([projection_accuracy_dataset_all, projection_accuracy_dataset_league],
                                                    ignore_index=True)
        projection_accuracy_dataset_all.drop_duplicates(subset=['fixture_id'], keep='last', inplace=True)
        projection_accuracy_dataset_all.reset_index(drop=True, inplace=True)
        file_path = os.path.join(data_folder_path, "all_leagues_accuracy_dataset.xlsx")
        projection_accuracy_dataset_all.to_excel(file_path, index=False)

        # # **Player Projections**
        #
        # Distributing the above dataframe's values to each player based on the % of teams total

        # In[114]:


        pl_projections = distribute_team_predictions_to_players(player_stats, team_stats, team_projections, stats_types,
                                                                fixtures_df, players, teams, comps, 0.97,
                                                                season_id=[current_season_id, previous_season_id,
                                                                           previous_season_id_above, previous_season_id_below])

        player_pos = []
        saves = []
        for i in range(len(pl_projections[['fixture_id', 'Player', 'Team']].values)):
            player = pl_projections['Player'].iloc[i]
            team = pl_projections['Team'].iloc[i]
            pos = get_player_position(player, team, players, teams)
            if pos == 'GK':
                team_projections_fix = team_projections[team_projections['fixture_id'] == pl_projections['fixture_id'].iloc[i]]
                saves.append(team_projections_fix[team_projections_fix['Team'] == team]['Saves'].values[0])
            else:
                saves.append(0)
            player_pos.append(pos)
        pl_projections['Position'] = player_pos
        pl_projections['Saves'] = saves

        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Goals',
             'Assists', 'Shots Total',
             'Shots On Target', 'Key Passes', 'Passes', 'Accurate Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellowcards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]

        pl_projections.rename(columns={'Yellowcards': 'Yellow Cards'}, inplace=True)

        # ## **Predicted Lineups**
        #
        # Which players are predicted to play?

        # In[115]:


        pred_starters = player_stats[player_stats['fixture_id'].isin(next_fix['id'])].copy()
        pred_starters = pred_starters[pred_starters['stats_type_id'] == 11]

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
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?',
             'Goals', 'Assists', 'Shots Total',
             'Shots On Target', 'Key Passes', 'Passes', 'Accurate Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]
        pl_projections = pl_projections.round(2)

        # In[116]:


        team_projections_next6 = team_projections_save

        # In[117]:


        pl_projections.sort_values(by=['Goals'], ascending=False, inplace=True)
        pl_projections.reset_index(drop=True, inplace=True)
        pl_projections = pl_projections.round(2)
        pl_projections_next_fix = pl_projections[pl_projections['fixture_id'].isin(next_fix['id'])]
        pl_projections_next_fix.reset_index(drop=True, inplace=True)
        file_path = os.path.join(data_folder_path, f"{league} Player.csv")
        pl_projections_next_fix.to_csv(file_path, index=False)

        # In[118]:


        pl_projections = pl_projections[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue', 'Start?',
             'Goals', 'Assists', 'Shots Total',
             'Shots On Target', 'Key Passes', 'Passes', 'Accurate Passes', 'Interceptions', 'Tackles', 'Total Crosses',
             'Yellow Cards', 'Offsides', 'Fouls', 'Fouls Drawn', 'Saves']]

        # In[119]:


        pl_projections_next_fix = pl_projections[pl_projections['fixture_id'].isin(next_fix['id'])]
        pl_projections_next_fix.sort_values(by=['Goals'], ascending=False, inplace=True)
        pl_projections_next_fix.reset_index(drop=True, inplace=True)

        # ## **FPL Points**

        # In[120]:

        file_path = os.path.join(data_folder_path, "PL Fantasy Players.xlsx")
        pl_players = pd.read_excel(file_path)
        pl_projections['Player'] = pl_projections['Player'].str.strip()
        pl_projections['FPL Position'] = pl_projections['Player'].map(pl_players.set_index('Player')['FPL Position'])
        pl_projections.loc[:, 'CBIT Hit Rate'] = 0
        pl_projections.loc[:, 'CBIT Average'] = 0
        pl_projections.loc[:, 'Clearances Average'] = 0
        pl_projections.loc[:, 'Blocked Shots Average'] = 0
        pl_projections.loc[:, 'Ball Recovery Average'] = 0
        pl_projections.loc[:, 'Tackles Won Average'] = 0
        pl_projections.loc[:, 'Full Match Hit Rate'] = 0
        for player in pl_projections['Player'].unique():
            team = pl_projections[pl_projections['Player'] == player]['Team'].values[0]
            position = pl_projections[pl_projections['Player'] == player]['FPL Position'].values[0]
            try:
                player_cbit_perc, player_cbit_avg, player_clearances_avg, player_blocked_shots_avg, player_recovery_avg, tackles_won_avg, full_match_hit_rate = get_extra_stats(
                    player, position, team, teams, players, player_stats, fixtures_df, stats_types, weight=0.96, mins=50,
                    games=50)
                pl_projections.loc[
                    (pl_projections['Player'] == player) & (pl_projections['Team'] == team), 'CBIT Hit Rate'] = player_cbit_perc
                pl_projections.loc[
                    (pl_projections['Player'] == player) & (pl_projections['Team'] == team), 'CBIT Average'] = player_cbit_avg
                pl_projections.loc[(pl_projections['Player'] == player) & (
                            pl_projections['Team'] == team), 'Clearances Average'] = player_clearances_avg
                pl_projections.loc[(pl_projections['Player'] == player) & (
                            pl_projections['Team'] == team), 'Blocked Shots Average'] = player_blocked_shots_avg
                pl_projections.loc[(pl_projections['Player'] == player) & (
                            pl_projections['Team'] == team), 'Ball Recovery Average'] = player_recovery_avg
                pl_projections.loc[(pl_projections['Player'] == player) & (
                            pl_projections['Team'] == team), 'Tackles Won Average'] = tackles_won_avg
                pl_projections.loc[(pl_projections['Player'] == player) & (
                            pl_projections['Team'] == team), 'Full Match Hit Rate'] = full_match_hit_rate
            except:
                continue

        # In[121]:


        fpl_points_dict_gk = {
            'Goals': 10,
            'Assists': 3,
            'Clean Sheet': 4,
            'Saves': 1,
            'Penalties Saved': 5,
            'Goals Conceded': -1,
            'Yellow Card': -1,
        }

        fpl_points_dict_def = {
            'Goals': 6,
            'Assists': 3,
            'Clean Sheet': 4,
            'Goals Conceded': -1,
            'Yellow Card': -1,
        }

        fpl_points_dict_mid = {
            'Goals': 5,
            'Assists': 3,
            'Clean Sheet': 1,
            'Yellow Card': -1,
        }

        fpl_points_dict_fwd = {
            'Goals': 4,
            'Assists': 3,
            'Yellow Card': -1,
        }

        # In[122]:


        fpl_bonus_dict_gk = {
            'Goals': 12,
            'Winning Goal': 3,
            'Assists': 9,
            'Clean Sheet': 12,
            'Saves': 2.66,
            'Penalties Saved': 8,
            'Key Passes': 1,
            'Big Chances Created': 3,
            'Successful Dribbles': 1,
            'Clearance Offline': 9,
            'Big Chances Missed': -3,
            'Clearances, Blocks & Interceptions': 0.5,
            'Recoveries': 0.33,
            'Tackles Won': 2,
            'Fouls Drawn': 1,
            'Shots On Target': 2,
            'Shots Off Target': -1,
            'Offsides': -1,
            'Fouls': -1,
            '70-79% Passes Completed': 2,
            '80-89% Passes Completed': 4,
            '90%+ Passes Completed': 6,
            'Goals Conceded': -4,
            'Yellow Card': -3,
        }

        fpl_bonus_dict_def = {
            'Goals': 12,
            'Winning Goal': 3,
            'Assists': 9,
            'Clean Sheet': 12,
            'Clearances, Blocks & Interceptions': 0.5,
            'Recoveries': 0.33,
            'Tackles Won': 2,
            'Fouls Drawn': 1,
            'Shots On Target': 2,
            'Shots Off Target': -1,
            'Offsides': -1,
            'Fouls': -1,
            '70-79% Passes Completed': 2,
            '80-89% Passes Completed': 4,
            '90%+ Passes Completed': 6,
            'Key Passes': 1,
            'Big Chances Created': 3,
            'Successful Dribbles': 1,
            'Clearance Offline': 9,
            'Big Chances Missed': -3,
            'Goals Conceded': -4,
            'Yellow Card': -3,
        }

        fpl_bonus_dict_mid = {
            'Goals': 18,
            'Winning Goal': 3,
            'Assists': 9,
            'Clearances, Blocks & Interceptions': 0.5,
            'Recoveries': 0.33,
            'Tackles Won': 2,
            'Fouls Drawn': 1,
            'Shots On Target': 2,
            'Shots Off Target': -1,
            'Offsides': -1,
            'Fouls': -1,
            '70-79% Passes Completed': 2,
            '80-89% Passes Completed': 4,
            '90%+ Passes Completed': 6,
            'Key Passes': 1,
            'Big Chances Created': 3,
            'Successful Dribbles': 1,
            'Clearance Offline': 9,
            'Big Chances Missed': -3,
            'Yellow Card': -3,
        }

        fpl_bonus_dict_fwd = {
            'Goals': 24,
            'Winning Goal': 3,
            'Assists': 9,
            'Key Passes': 1,
            'Big Chances Created': 3,
            'Successful Dribbles': 1,
            'Clearance Offline': 9,
            'Big Chances Missed': -3,
            'Clearances, Blocks & Interceptions': 0.5,
            'Recoveries': 0.33,
            'Tackles Won': 2,
            'Fouls Drawn': 1,
            'Shots On Target': 2,
            'Shots Off Target': -1,
            'Offsides': -1,
            'Fouls': -1,
            '70-79% Passes Completed': 2,
            '80-89% Passes Completed': 4,
            '90%+ Passes Completed': 6,
            'Yellow Card': -3,
        }

        # In[123]:


        try:
            if len(pl_projections[pl_projections['FPL Position'].isnull()]['Player'].unique()) > 0:
                missing_positions = pl_projections[pl_projections['FPL Position'].isnull()]['Player'].unique().tolist()
                raise ValueError(f"Missing FPL Positions for players: {', '.join(missing_positions)}")
        except KeyError:
            pass

        # In[124]:


        fpl_point_df_next6 = get_fpl_points(pl_projections, score_preds, fpl_points_dict_gk, fpl_points_dict_def,
                                            fpl_points_dict_mid, fpl_points_dict_fwd)
        bps_df_next6 = bonus_points_score(pl_projections, score_preds, fpl_bonus_dict_gk, fpl_bonus_dict_def,
                                          fpl_bonus_dict_mid, fpl_bonus_dict_fwd)
        bonus_next6 = get_bonus_points(bps_df_next6, score_preds, expo_factor=0.1)

        fpl_df_next6 = fpl_point_df_next6.merge(bonus_next6, on=['Player', 'Team', 'Opponent'], how='left',
                                                suffixes=('', '_Bonus'))
        fpl_df_next6['FPL Points'] = fpl_df_next6['PTS'] + fpl_df_next6['Bonus Points'].fillna(0)
        fpl_df_next6 = fpl_df_next6[
            ['fixture_id', 'kickoff_datetime', 'player_id', 'Player', 'Position', 'Team', 'Opponent', 'Venue',
             'FPL Points']].copy()
        fpl_df_next_fix = fpl_df_next6[fpl_df_next6['fixture_id'].isin(next_fix['id'])]
        fpl_df_next6.sort_values(by=['kickoff_datetime'], inplace=True)
        fpl_df_next6['Gameweek'] = fpl_df_next6.groupby(['Player', 'Team']).cumcount() + 1 + matches_played
        fpl_df_next6['Gameweek'] = 'GW' + fpl_df_next6['Gameweek'].astype(str)
        fpl_df_next6.drop(columns=['kickoff_datetime', 'fixture_id', 'Opponent'], inplace=True)

        fpl_df_next_fix.sort_values(by=['FPL Points'], ascending=False, inplace=True)
        fpl_df_next_fix.reset_index(drop=True, inplace=True)
        fpl_df_next_fix = fpl_df_next_fix.round(2)
        file_path = os.path.join(save_file_path, f"{league} FPL.csv")
        fpl_df_next_fix.to_csv(file_path, index=False)
        await insert_fpl_projections_async(fpl_df_next_fix);

        # In[125]:

        fpl_df_next6 = fpl_df_next6.pivot_table(
            index=['player_id', 'Player', 'Position', 'Team'],
            columns='Gameweek',
            values='FPL Points',
            fill_value=0
        )

        fpl_df_next6.columns.name = None  # Remove the name of the columns index
        fpl_df_next6.reset_index(inplace=True)

        fpl_df_next6 = fpl_df_next6[
            ['player_id', 'Player', 'Position', 'Team'] + sorted([col for col in fpl_df_next6.columns if col.startswith('GW')],
                                                                 key=lambda x: int(x[2:]))]
        fpl_df_next6['Total'] = fpl_df_next6.iloc[:, 4:].sum(axis=1)
        fpl_df_next6.sort_values(by='Total', ascending=False, inplace=True)
        fpl_df_next6.reset_index(drop=True, inplace=True)
        fpl_df_next6 = fpl_df_next6.round(2)
        file_path = os.path.join(save_file_path, f"{league} FPL 6 Week.csv")
        fpl_df_next6.to_csv(file_path, index=False)

        # In[126]:


        ## Add floor points to draftkings


        # ## **Opta Points**

        # In[127]:


        opta_points_dict = {
            'Goals': 10,
            'Assists': 6,
            'Shots Off': 2,
            'Shots On Target': 4,
            'Passes': 0.2,
            'Interceptions': 2,
            'Tackles': 2,
            'Blocked Shots': 2,
            'Total Crosses': 0.2,
            'Yellow Cards': -2,
            'Fouls': -1,
            'Fouls Drawn': 1,
            'Saves': 5,
            'Offsides': -1,
            'Goals Conceded': -1,
            'Penalties Saved': 5
        }

        # In[128]:


        pl_projections_next_fix = pl_projections[pl_projections['fixture_id'].isin(next_fix['id'])]
        pl_projections_next_fix.reset_index(drop=True, inplace=True)
        opta = get_opta_points(pl_projections_next_fix, score_preds, opta_points_dict)
        file_path = os.path.join(save_file_path, f"{league} Opta.csv")
        opta.to_csv(file_path, index=False)
        await insert_opta_projections_async(opta)

        # ## **Fan Team Points**

        # In[129]:


        fanteam_points_dict_gk = {
            'Goals': 8,
            'Assists': 3,
            'Shots On Target': 1,
            'Saves': 0.5,
            'Penalties Saved': 5,
            'Clean Sheet': 4,
            'Win': 0.3,
            'Lose': -0.3,
            'Goals Conceded': -1,
            'Yellow Card': -1
        }

        fanteam_points_dict_def = {
            'Goals': 6,
            'Assists': 3,
            'Shots On Target': 0.6,
            'Clean Sheet': 4,
            'Win': 0.3,
            'Lose': -0.3,
            'Goals Conceded': -1,
            'Yellow Card': -1
        }

        fanteam_points_dict_mid = {
            'Goals': 5,
            'Assists': 3,
            'Shots On Target': 0.4,
            'Clean Sheet': 1,
            'Win': 0.3,
            'Lose': -0.3,
            'Yellow Card': -1,
            'Full Match': 1
        }

        fanteam_points_dict_fwd = {
            'Goals': 4,
            'Assists': 3,
            'Shots On Target': 0.4,
            'Win': 0.3,
            'Lose': -0.3,
            'Yellow Card': -1,
            'Full Match': 1
        }

        # In[130]:


        if fanteam_csv_imported == 'Yes':
            file_path = os.path.join(data_folder_path, "Fanteam Data.csv")
            fanteam_csv = pd.read_csv(file_path)
        file_path = os.path.join(data_folder_path, "Fanteam Mapping.xlsx")
        fanteam_mapping = pd.read_excel(file_path)

        # In[131]:


        pl_projections_next_fix.loc[:, 'FanTeam Position'] = pl_projections_next_fix['Player'].map(
            pl_players.set_index('Player')['FanTeam Position'])
        pl_projections_next_fix['FanTeam ID'] = pl_projections_next_fix['player_id'].map(
            fanteam_mapping.set_index('SM Player ID')['FanTeam PlayerID'])
        if fanteam_csv_imported == 'Yes':
            pl_projections_next_fix['Lineup'] = pl_projections_next_fix['FanTeam ID'].map(
                fanteam_csv.set_index('PlayerID')['Lineup'])
            pl_projections_next_fix['Price'] = pl_projections_next_fix['FanTeam ID'].map(
                fanteam_csv.set_index('PlayerID')['Price'])
            pl_projections_next_fix_temp = pl_projections_next_fix[
                pl_projections_next_fix['Lineup'].isin(['expected', 'possible'])]
            pl_projections_next_fix_temp = pl_projections_next_fix_temp[
                pl_projections_next_fix_temp['FanTeam Position'].notna()]
        else:
            pl_projections_next_fix['Price'] = 0
            pl_projections_next_fix_temp = pl_projections_next_fix[pl_projections_next_fix['FanTeam Position'].notna()]
        pl_projections_next_fix_temp.reset_index(drop=True, inplace=True)
        fanteam_point_df = get_fanteam_points(pl_projections_next_fix_temp, score_preds, fanteam_points_dict_gk,
                                              fanteam_points_dict_def, fanteam_points_dict_mid, fanteam_points_dict_fwd)
        fanteam_point_df.dropna(inplace=True)
        fanteam_point_df.to_csv(f'{save_file_path}/{league} Fanteam.csv')

        await insert_fanteam_projections_async(fanteam_point_df)
        # In[132]:


        draftkings_points_dict_gk = {
            'Goals': 10,
            'Assists': 6,
            'Shots Total': 1,
            'Shots On Target': 1,
            'Total Crosses': 0.7,
            'Key Passes': 1,
            'Successful Passes': 0.02,
            'Fouls Drawn': 1,
            'Fouls Committed': -0.5,
            'Tackles Won': 1,
            'Saves': 2,
            'Penalties Saved': 5,
            'Clean Sheet': 5,
            'Win': 5,
            'Goals Conceded': -2,
            'Yellow Card': -1.5,
        }

        draftkings_points_dict_def = {
            'Goals': 10,
            'Assists': 6,
            'Shots Total': 1,
            'Shots On Target': 1,
            'Total Crosses': 0.7,
            'Key Passes': 1,
            'Successful Passes': 0.02,
            'Fouls Drawn': 1,
            'Fouls Committed': -0.5,
            'Tackles Won': 1,
            'Interceptions': 0.5,
            'Clean Sheet': 3,
            'Yellow Card': -1.5,
        }

        draftkings_points_dict_mid = {
            'Goals': 10,
            'Assists': 6,
            'Shots Total': 1,
            'Shots On Target': 1,
            'Total Crosses': 0.7,
            'Key Passes': 1,
            'Successful Passes': 0.02,
            'Fouls Drawn': 1,
            'Fouls Committed': -0.5,
            'Tackles Won': 1,
            'Interceptions': 0.5,
            'Yellow Card': -1.5,
        }

        draftkings_points_dict_fwd = {
            'Goals': 10,
            'Assists': 6,
            'Shots Total': 1,
            'Shots On Target': 1,
            'Total Crosses': 0.7,
            'Key Passes': 1,
            'Successful Passes': 0.02,
            'Fouls Drawn': 1,
            'Fouls Committed': -0.5,
            'Tackles Won': 1,
            'Interceptions': 0.5,
            'Yellow Card': -1.5,
        }

        # In[133]:


        pl_projections_next_fix.loc[:, 'Draftkings Position'] = pl_projections_next_fix['Player'].map(
            pl_players.set_index('Player')['Draftkings Position'])
        draftkings_point_df = get_draftkings_points(pl_projections_next_fix, score_preds, draftkings_points_dict_gk,
                                                    draftkings_points_dict_def, draftkings_points_dict_mid,
                                                    draftkings_points_dict_fwd)

        # ## **Dream 11 Points**

        # In[134]:


        dream11_points_dict_gk = {
            'Goals': 60,
            'Assists': 20,
            'Key Passes': 3,
            'Shots On Target': 6,
            'Successful Passes': 0.2,
            'Tackles Won': 4,
            'Interceptions': 4,
            'Clean Sheet': 20,
            'Saves': 6,
            'Penalties Saved': 50,
            'Goals Conceded': -2,
            'Yellow Card': -4,
        }

        dream11_points_dict_def = {
            'Goals': 60,
            'Assists': 20,
            'Key Passes': 3,
            'Shots On Target': 6,
            'Successful Passes': 0.2,
            'Tackles Won': 4,
            'Interceptions': 4,
            'Clean Sheet': 20,
            'Goals Conceded': -2,
            'Yellow Card': -4,
        }

        dream11_points_dict_mid = {
            'Goals': 50,
            'Assists': 20,
            'Key Passes': 3,
            'Shots On Target': 6,
            'Successful Passes': 0.2,
            'Tackles Won': 4,
            'Interceptions': 4,
            'Yellow Card': -4,
        }

        dream11_points_dict_fwd = {
            'Goals': 40,
            'Assists': 20,
            'Key Passes': 3,
            'Shots On Target': 6,
            'Successful Passes': 0.2,
            'Tackles Won': 4,
            'Interceptions': 4,
            'Yellow Card': -4,
        }

        # In[135]:


        pl_projections_next_fix.loc[:, 'Dream11 Position'] = pl_projections_next_fix['Player'].map(
            pl_players.set_index('Player')['Dream11 Position'])
        dream11_point_df = get_dream11_points(pl_projections_next_fix, score_preds, dream11_points_dict_gk,
                                              dream11_points_dict_def, dream11_points_dict_mid, dream11_points_dict_fwd)

        # ## **Player Stat Probabilities**
        #
        # Using Poisson Distribution to get the likelihood of players acheiving certain statistics.

        # In[136]:


        pl_projections_next_fix.rename(columns={'Fouls': 'Fouls Committed'}, inplace=True)

        # In[137]:


        perc_stats = ['Shots On Target', 'Fouls Committed', 'Fouls Drawn']
        lines = [1, 2, 3]

        # In[138]:


        player_stat_probs = get_poisson_probs(pl_projections_next_fix, perc_stats, lines)
        player_stat_probs = player_stat_probs.round(2)
        file_path = os.path.join(save_file_path, f"{league} Player Stat Props.csv")
        player_stat_probs.to_csv(file_path, index=False)
        await insert_players_stats_async(player_stat_probs, teams=teams, competition_id=league_id, comp_teams=comp_teams)
