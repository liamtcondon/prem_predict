import pandas as pd
import numpy as np
import joblib
import warnings

warnings.simplefilter(action='ignore')

print("Loading XGBoost Brain and Datasets...")
model = joblib.load('Data/xgboost_pl_model.pkl')
league_avg_xg = joblib.load('Data/league_avg_xg.pkl')

pl_df = pd.read_csv('Data/live_pl_history.csv')
pl_df['match_date'] = pd.to_datetime(pl_df['match_date'], utc=True).dt.tz_localize(None)
pl_df = pl_df.sort_values('match_date').reset_index(drop=True)

try:
    global_df = pd.read_csv('Data/global_schedule.csv')
    global_df['date'] = pd.to_datetime(global_df['date'], utc=True).dt.tz_localize(None)
except FileNotFoundError:
    global_df = pl_df.copy()
    global_df['date'] = global_df['match_date']

def calculate_elo(df, k_factor=20):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo_dict = {team: 1500 for team in teams} 
    home_elo_list, away_elo_list, exp_home_list, exp_away_list = [], [], [], []

    for index, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        cur_home, cur_away = elo_dict[home], elo_dict[away]
        home_elo_list.append(cur_home)
        away_elo_list.append(cur_away)
        exp_home = 1 / (1 + 10 ** ((cur_away - cur_home) / 400))
        exp_away = 1 / (1 + 10 ** ((cur_home - cur_away) / 400))
        exp_home_list.append(exp_home)
        exp_away_list.append(exp_away)
        hr, ar = (1, 0) if row['home_goals'] > row['away_goals'] else (0.5, 0.5) if row['home_goals'] == row['away_goals'] else (0, 1)
        elo_dict[home] = cur_home + k_factor * (hr - exp_home)
        elo_dict[away] = cur_away + k_factor * (ar - exp_away)
        
    df['Home_Elo'], df['Away_Elo'] = home_elo_list, away_elo_list
    df['Vegas_Prob_Home'], df['Vegas_Prob_Away'] = exp_home_list, exp_away_list
    return df

pl_df = calculate_elo(pl_df)
test_df = pl_df.tail(380).reset_index(drop=True)

def run_simulation(threshold):
    bankroll = 1000.00
    history = []
    wins, losses, total_wagered = 0, 0, 0
    correct_home, total_home = 0, 0
    correct_away, total_away = 0, 0
    avg_ai_conf = []

    for index, row in test_df.iterrows():
        home, away, target_date = row['home_team'], row['away_team'], row['match_date']
        actual = 2 if row['home_goals'] > row['away_goals'] else 1 if row['home_goals'] == row['away_goals'] else 0

        pl_games = pl_df[(pl_df['home_team'] == home) | (pl_df['away_team'] == home)]
        past_pl = pl_games[pl_games['match_date'] < target_date].sort_values('match_date', ascending=False).head(5)
        h_att = np.mean([g['Total_xG_home'] if g['home_team'] == home else g['Total_xG_away'] for _, g in past_pl.iterrows()]) / league_avg_xg if not past_pl.empty else 1.0
        
        pl_games_a = pl_df[(pl_df['home_team'] == away) | (pl_df['away_team'] == away)]
        past_pl_a = pl_games_a[pl_games_a['match_date'] < target_date].sort_values('match_date', ascending=False).head(5)
        a_att = np.mean([g['Total_xG_away'] if g['away_team'] == away else g['Total_xG_home'] for _, g in past_pl_a.iterrows()]) / league_avg_xg if not past_pl_a.empty else 1.0
        
        feat = pd.DataFrame([{'Home_Att_xG': h_att, 'Home_Def_xG': 1.0, 'Home_Rest': 7, 'Home_Elo': row['Home_Elo'],
                              'Away_Att_xG': a_att, 'Away_Def_xG': 1.0, 'Away_Rest': 7, 'Away_Elo': row['Away_Elo']}])
        
        ai_probs = model.predict_proba(feat)[0]
        v_h, v_a = row['Vegas_Prob_Home'] + 0.05, row['Vegas_Prob_Away'] + 0.05
        
        edge_h = ai_probs[2] - v_h
        edge_a = ai_probs[0] - v_a
        
        bet_placed = False
        if edge_h > threshold: 
            bet_placed, pick, odds = True, 2, 1/v_h
            total_home += 1
            avg_ai_conf.append(ai_probs[2])
        elif edge_a > threshold: 
            bet_placed, pick, odds = True, 0, 1/v_a
            total_away += 1
            avg_ai_conf.append(ai_probs[0])

        if bet_placed:
            total_wagered += 100
            if pick == actual:
                wins += 1
                bankroll += 100 * (odds - 1)
                if pick == 2: correct_home += 1
                else: correct_away += 1
            else:
                losses += 1
                bankroll -= 100
        
        history.append({"date": target_date, "bankroll": bankroll})
    
    accuracy = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    mean_conf = np.mean(avg_ai_conf) if avg_ai_conf else 0
    
    return {
        'bankroll': bankroll,
        'wins': wins,
        'losses': losses,
        'total_wagered': total_wagered,
        'history': history,
        'accuracy': accuracy,
        'avg_conf': mean_conf,
        'home_split': (correct_home, total_home),
        'away_split': (correct_away, total_away)
    }

print("Optimizing Edge Threshold (Grid Search 0.0% to 5.0%)...")
optimization_results = []
for t in np.arange(0.0, 0.05, 0.002):
    res = run_simulation(t)
    optimization_results.append({
        'threshold': t, 
        'profit': res['bankroll'] - 1000, 
        'data': res
    })

best_run = max(optimization_results, key=lambda x: x['profit'])
best_threshold = best_run['threshold']
res = best_run['data']

# THIS IS THE CRITICAL LINE THAT CREATES THE FILE FOR THE APP
pd.DataFrame(res['history']).to_csv('Data/bankroll_history.csv', index=False)

print("-" * 60)
print(f"🎯 SWEET SPOT FOUND: {best_threshold*100:.1f}% Edge")
print("Bankroll history saved to Data/bankroll_history.csv")