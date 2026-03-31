import requests
import pandas as pd
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Map the ESPN API slugs to your original naming convention
LEAGUES = {
    'eng.1': 'ENG-Premier League', 
    'eng.fa': 'ENG-FA Cup', 
    'eng.league_cup': 'ENG-League Cup', 
    'uefa.champions': 'UEFA-Champions League'
}

# We pull 4 years of history to match our XGBoost training data
SEASONS = [2021, 2022, 2023, 2024, 2025]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

print("Scraping Global Schedules (Direct via ESPN API)...")
all_games = []

for slug, name in LEAGUES.items():
    for year in SEASONS:
        print(f" -> Downloading {name} ({year})...")
        
        # Force ESPN to look at the full European calendar year (August 1st to June 30th)
        start_date = f"{year}0801"
        end_date = f"{year+1}0630"
        
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={start_date}-{end_date}&limit=1000"
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            events = response.json().get('events', [])
            
            for event in events:
                try:
                    game_date = event['date']
                    competitors = event['competitions'][0]['competitors']
                    
                    # Hunt for home and away teams
                    home_comp = next(c for c in competitors if c['homeAway'] == 'home')
                    away_comp = next(c for c in competitors if c['homeAway'] == 'away')
                    
                    home_team = home_comp['team']['displayName']
                    away_team = away_comp['team']['displayName']
                    
                    # Safely extract the nested ESPN score dictionary
                    home_score_data = home_comp.get('score', {})
                    away_score_data = away_comp.get('score', {})
                    
                    home_score = home_score_data.get('value', 0) if isinstance(home_score_data, dict) else home_score_data
                    away_score = away_score_data.get('value', 0) if isinstance(away_score_data, dict) else away_score_data

                    all_games.append({
                        'date': game_date,
                        'league': name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': float(home_score) if home_score else 0.0,
                        'away_score': float(away_score) if away_score else 0.0
                    })
                except Exception as e:
                    # If a specific game is corrupted, skip it and keep mining
                    continue 
        else:
            print(f"    [!] ESPN rejected request. Status: {response.status_code}")
            
        time.sleep(1) # Be polite to the servers

clean_schedule = pd.DataFrame(all_games)

if not clean_schedule.empty:
    clean_schedule['date'] = pd.to_datetime(clean_schedule['date'])
    clean_schedule = clean_schedule.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    clean_schedule.to_csv('Data/global_schedule.csv', index=False)
    print(f"\n✅ Saved Global Schedule! Successfully pulled {len(clean_schedule)} total matches.")
else:
    print("\n❌ Failed to pull any data.")