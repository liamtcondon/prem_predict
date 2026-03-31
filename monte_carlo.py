import pandas as pd
import numpy as np
import copy
import joblib
import warnings

warnings.simplefilter(action='ignore')

model = joblib.load('Data/xgboost_pl_model.pkl')
league_avg_xg = joblib.load('Data/league_avg_xg.pkl')
current_elos = joblib.load('Data/elo_dict.pkl') 

def get_current_standings(history_df):
    teams = pd.concat([history_df['home_team'], history_df['away_team']]).unique()
    standings = {team: {'points': 0, 'gd': 0} for team in teams}
    for index, row in history_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_goals = row['home_score'] 
        away_goals = row['away_score']
        
        # Safety check: skip unplayed games that slip through
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue
            
        standings[home]['gd'] += (home_goals - away_goals)
        standings[away]['gd'] += (away_goals - home_goals)
        if home_goals > away_goals: standings[home]['points'] += 3
        elif home_goals < away_goals: standings[away]['points'] += 3
        else:
            standings[home]['points'] += 1
            standings[away]['points'] += 1
    return standings

def get_team_snapshot(team_name, history_df):
    team_games = history_df[(history_df['home_team'] == team_name) | (history_df['away_team'] == team_name)]
    last_5 = team_games.sort_values('match_date', ascending=False).head(5)
    xg_f, xg_a = [], []
    for _, g in last_5.iterrows():
        if g['home_team'] == team_name:
            xg_f.append(g['Total_xG_home'])
            xg_a.append(g['Total_xG_away'])
        else:
            xg_f.append(g['Total_xG_away'])
            xg_a.append(g['Total_xG_home'])
    if len(xg_f) == 0: xg_f, xg_a = [1.3], [1.3]
    
    # Safe .get() prevents KeyErrors if a name slips through
    return np.mean(xg_f) / league_avg_xg, np.mean(xg_a) / league_avg_xg, current_elos.get(team_name, 1500)

def pre_calculate_probabilities(unplayed_games, history_df):
    teams = pd.concat([unplayed_games['home_team'], unplayed_games['away_team']]).unique()
    team_snapshots = {team: get_team_snapshot(team, history_df) for team in teams}
    future_features = []
    for index, game in unplayed_games.iterrows():
        home = game['home_team']
        away = game['away_team']
        home_att, home_def, home_elo = team_snapshots.get(home, (1.0, 1.0, 1500))
        away_att, away_def, away_elo = team_snapshots.get(away, (1.0, 1.0, 1500))
        future_features.append({
            'Home_Att_xG': home_att, 'Home_Def_xG': home_def, 'Home_Rest': 7, 'Home_Elo': home_elo,
            'Away_Att_xG': away_att, 'Away_Def_xG': away_def, 'Away_Rest': 7, 'Away_Elo': away_elo
        })
    all_probs = model.predict_proba(pd.DataFrame(future_features))
    unplayed_games = unplayed_games.copy()
    unplayed_games['away_prob'] = all_probs[:, 0]
    unplayed_games['draw_prob'] = all_probs[:, 1]
    unplayed_games['home_prob'] = all_probs[:, 2]
    return unplayed_games

def run_monte_carlo(current_standings, unplayed_games, n_simulations=10000):
    teams = list(current_standings.keys())
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(unplayed_games)
    
    baseline_pts = np.zeros(n_teams)
    baseline_gd = np.zeros(n_teams)
    for team, stats in current_standings.items():
        idx = team_to_idx[team]
        baseline_pts[idx] = stats['points']
        baseline_gd[idx] = stats['gd']
        
    home_indices = unplayed_games['home_team'].map(team_to_idx).values
    away_indices = unplayed_games['away_team'].map(team_to_idx).values
    
    p_away = unplayed_games['away_prob'].values
    p_draw = unplayed_games['draw_prob'].values
    p_home = unplayed_games['home_prob'].values
    totals = p_away + p_draw + p_home
    p_away, p_draw, p_home = p_away/totals, p_draw/totals, p_home/totals
    
    thresh_away = p_away
    thresh_draw = p_away + p_draw
    
    rolls = np.random.rand(n_simulations, n_games)
    away_wins = rolls < thresh_away
    draws = (rolls >= thresh_away) & (rolls < thresh_draw)
    home_wins = rolls >= thresh_draw
    
    home_pts_gained = (home_wins * 3) + (draws * 1)
    away_pts_gained = (away_wins * 3) + (draws * 1)
    
    home_routing = np.zeros((n_games, n_teams))
    home_routing[np.arange(n_games), home_indices] = 1
    away_routing = np.zeros((n_games, n_teams))
    away_routing[np.arange(n_games), away_indices] = 1
    
    final_sim_pts = (home_pts_gained @ home_routing) + (away_pts_gained @ away_routing) + baseline_pts
    sim_scores = final_sim_pts + (baseline_gd / 1000.0)
    
    # Calculate the exact rank (1-20) for every team
    team_ranks = np.argsort(np.argsort(-sim_scores, axis=1), axis=1)
    
    # Build the Spreadsheet Matrix
    rank_probs = {}
    for idx, team in enumerate(teams):
        counts = np.bincount(team_ranks[:, idx], minlength=n_teams)
        rank_probs[team] = (counts / n_simulations) * 100
        
    rank_df = pd.DataFrame.from_dict(rank_probs, orient='index')
    rank_df.columns = [str(i+1) for i in range(n_teams)] 
    
    # Sort the rows by expected points
    expected_points = np.mean(final_sim_pts, axis=0)
    team_pts = {team: expected_points[idx] for idx, team in enumerate(teams)}
    sorted_teams = sorted(teams, key=lambda t: team_pts[t], reverse=True)
    rank_df = rank_df.loc[sorted_teams]
    
    return rank_df

def generate_season_projections(n_sims=5000):
    sched_df = pd.read_csv('Data/global_schedule.csv')
    
    # --- NAME STANDARDIZATION FIX ---
    mapping = {
        'Leeds United': 'Leeds',
        'Brighton & Hove Albion': 'Brighton',
        'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves', 
        'West Ham United': 'West Ham',
        'Newcastle United': 'Newcastle',
        'AFC Bournemouth': 'Bournemouth',
        'Sheffield United': 'Sheffield Utd',
        'Nottingham Forest': "Nott'm Forest", 
        'Luton Town': 'Luton',
        'Ipswich Town': 'Ipswich',
        'Leicester City': 'Leicester'
    }
    sched_df['home_team'] = sched_df['home_team'].replace(mapping)
    sched_df['away_team'] = sched_df['away_team'].replace(mapping)
    # --------------------------------
    
    sched_df['date'] = pd.to_datetime(sched_df['date'], utc=True)
    current_season = sched_df[(sched_df['league'] == 'ENG-Premier League') & (sched_df['date'] > '2025-08-01')]
    
    played_games = current_season[current_season['date'] <= '2026-03-31']
    unplayed_games = current_season[current_season['date'] > '2026-03-31']
    
    current_table = get_current_standings(played_games)
    history_df = pd.read_csv('Data/live_pl_history.csv')
    history_df['match_date'] = pd.to_datetime(history_df['match_date'], utc=True)
    unplayed_with_odds = pre_calculate_probabilities(unplayed_games, history_df)
    
    return run_monte_carlo(current_table, unplayed_with_odds, n_simulations=n_sims)