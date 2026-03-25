import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.simplefilter(action='ignore')

print("Loading Datasets...")
# 1. The Math Data (Understat PL games)
pl_df = pd.read_csv('Data/live_pl_history.csv')
# FORCE UTC, THEN STRIP IT TO MAKE IT NAIVE
pl_df['match_date'] = pd.to_datetime(pl_df['match_date'], utc=True).dt.tz_localize(None) 
pl_df = pl_df.sort_values('match_date').reset_index(drop=True)

# Calculate League Average xG for Scaling
league_avg_xg = (pl_df['Total_xG_home'].mean() + pl_df['Total_xG_away'].mean()) / 2

# 2. The Fatigue Data
try:
    global_df = pd.read_csv('Data/global_schedule.csv')
    # FORCE UTC, THEN STRIP IT TO MAKE IT NAIVE
    global_df['date'] = pd.to_datetime(global_df['date'], utc=True).dt.tz_localize(None)
except FileNotFoundError:
    print("WARNING: global_schedule.csv not found. Falling back to PL-only rest days.")
    global_df = pl_df.copy()
    global_df['date'] = global_df['match_date']

    
# --- ELO MATH FUNCTION ---
def calculate_elo(df, k_factor=20):
    print("Calculating Dynamic Elo Ratings...")
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo_dict = {team: 1500 for team in teams} 
    home_elo_list, away_elo_list = [], []

    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        current_home_elo = elo_dict[home]
        current_away_elo = elo_dict[away]
        home_elo_list.append(current_home_elo)
        away_elo_list.append(current_away_elo)
        
        if row['home_goals'] > row['away_goals']:
            home_result, away_result = 1, 0
        elif row['home_goals'] == row['away_goals']:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0, 1
            
        expected_home = 1 / (1 + 10 ** ((current_away_elo - current_home_elo) / 400))
        expected_away = 1 / (1 + 10 ** ((current_home_elo - current_away_elo) / 400))
        
        elo_dict[home] = current_home_elo + k_factor * (home_result - expected_home)
        elo_dict[away] = current_away_elo + k_factor * (away_result - expected_away)
        
    df['Home_Elo'] = home_elo_list
    df['Away_Elo'] = away_elo_list
    return df, elo_dict

pl_df, final_elo_dict = calculate_elo(pl_df)

print("Engineering Features (Rolling xG, True Fatigue, & Elo)...")
features = []

for index, row in pl_df.iterrows():
    if index < 50: continue

    home = row['home_team']
    away = row['away_team']
    target_date = row['match_date']
    
    if row['home_goals'] > row['away_goals']: target = 2
    elif row['home_goals'] == row['away_goals']: target = 1
    else: target = 0

    def get_team_context(team_name):
        pl_games = pl_df[(pl_df['home_team'] == team_name) | (pl_df['away_team'] == team_name)]
        past_pl = pl_games[pl_games['match_date'] < target_date].sort_values('match_date', ascending=False)
        
        last_5 = past_pl.head(5)
        if len(last_5) < 1: 
            return 1.3, 1.3, 7 
        
        xg_f, xg_a = [], []
        for _, g in last_5.iterrows():
            if g['home_team'] == team_name:
                xg_f.append(g['Total_xG_home'])
                xg_a.append(g['Total_xG_away'])
            else:
                xg_f.append(g['Total_xG_away'])
                xg_a.append(g['Total_xG_home'])
                
        all_games = global_df[(global_df['home_team'] == team_name) | (global_df['away_team'] == team_name)]
        past_global = all_games[all_games['date'] < target_date].sort_values('date', ascending=False)
        
        if len(past_global) > 0:
            last_game_date = past_global.iloc[0]['date']
            true_rest_days = min((target_date - last_game_date).days, 14)
        else:
            true_rest_days = 14

        scaled_att = np.mean(xg_f) / league_avg_xg
        scaled_def = np.mean(xg_a) / league_avg_xg

        return scaled_att, scaled_def, true_rest_days
    
    home_att, home_def, home_rest = get_team_context(home)
    away_att, away_def, away_rest = get_team_context(away)
    
    features.append({
        'Home_Att_xG': home_att, 'Home_Def_xG': home_def, 'Home_Rest': home_rest, 'Home_Elo': row['Home_Elo'],
        'Away_Att_xG': away_att, 'Away_Def_xG': away_def, 'Away_Rest': away_rest, 'Away_Elo': row['Away_Elo'],
        'Result': target
    })

ml_df = pd.DataFrame(features)

split_index = int(len(ml_df) * 0.8)
train = ml_df.iloc[:split_index]
test = ml_df.iloc[split_index:]

X_train = train.drop('Result', axis=1)
y_train = train['Result']
X_test = test.drop('Result', axis=1)
y_test = test['Result']

# ==========================================
# --- STEP 4: GRID SEARCH / MODEL TRAINING ---
# ==========================================
print("\nTraining XGBoost Neural Brain...")

# --- THE AUTOMATED HYPERPARAMETER TUNER ---
print("Starting Grid Search... Go grab a coffee, this might take 2-3 minutes!")

# 1. Define the parameters you want the computer to test
param_grid = {
    'max_depth': [2, 3, 4],                 # How deep should the tree think?
    'min_child_weight': [10, 15, 20],       # How much proof does it need to make a rule?
    'learning_rate': [0.01, 0.05, 0.1],     # How fast should it learn?
    'n_estimators': [100, 150, 200]         # How many trees should it build?
}

# 2. Set up the baseline model
base_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', subsample=0.8)

# 3. Unleash the Grid Search
# cv=3 means it will cross-validate every combination 3 times to ensure it isn't getting lucky
grid = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

print("\n" + "="*40)
print(f"🏆 BEST PARAMETERS FOUND:")
print(grid.best_params_)
print("="*40 + "\n")

# 4. Automatically use the best model it found!
model = grid.best_estimator_
# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("-" * 40)
print(f"✅ XGBoost Training Complete")
print(f"Holdout Test Accuracy: {accuracy * 100:.2f}%")
print("-" * 40)

# 6. Feature Importance
importance = model.feature_importances_
print("\nFeature Importance:")
for col, imp in zip(X_train.columns, importance):
    print(f"{col}: {imp*100:.1f}%")

joblib.dump(model, 'Data/xgboost_pl_model.pkl')
joblib.dump(league_avg_xg, 'Data/league_avg_xg.pkl')
joblib.dump(final_elo_dict, 'Data/elo_dict.pkl') 
print(f"\n💾 Model saved. League Avg used for scaling: {league_avg_xg:.2f}")