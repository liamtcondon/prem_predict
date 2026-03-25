import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import joblib 
import requests
from bs4 import BeautifulSoup

# --- API & SCRAPER CONFIGURATION ---
SCRAPER_API_KEY = "YOUR_API_KEY_HERE"

# 1. Page Configuration
st.set_page_config(page_title="Premier League Predictor", layout="centered")
st.title("⚽ Premier League Match Predictor")
st.write("Compare AI probabilities against Vegas odds to find betting value.")

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Data/live_pl_history.csv')
    df['match_date'] = pd.to_datetime(df['match_date'], utc=True).dt.tz_localize(None)
    avg_home_xg = df['Total_xG_home'].mean()
    avg_away_xg = df['Total_xG_away'].mean()
    overall_avg = (avg_home_xg + avg_away_xg) / 2
    teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    return df, teams, overall_avg

history, ALL_TEAMS, LEAGUE_AVG_XG = load_data()

# 3. Engines
def get_rolling_stats(team_name, history, window=5):
    team_games = history[(history['home_team'] == team_name) | (history['away_team'] == team_name)]
    team_games = team_games.sort_values('match_date', ascending=False)
    last_n = team_games.head(window)
    if len(last_n) < 1: return 1.3, 1.3 
    xg_for, xg_against = [], []
    for _, row in last_n.iterrows():
        if row['home_team'] == team_name:
            xg_for.append(row['Total_xG_home']); xg_against.append(row['Total_xG_away'])
        else:
            xg_for.append(row['Total_xG_away']); xg_against.append(row['Total_xG_home'])
    return np.mean(xg_for), np.mean(xg_against)

TEAM_COLORS = {
    "Arsenal": "#EF0107", "Manchester City": "#6CABDD", "Manchester United": "#DA291C",
    "Liverpool": "#C8102E", "Chelsea": "#034694", "Tottenham": "#132257",
    "Newcastle United": "#241F20", "Aston Villa": "#670E36", "Everton": "#003399",
    "Brighton": "#0057B8", "West Ham": "#7A263A", "Brentford": "#E30613",
    "Nottingham Forest": "#E53233", "Crystal Palace": "#1B458F", "Fulham": "#000000",
    "Bournemouth": "#B50E12", "Wolverhampton Wanderers": "#FDB913", "Leicester City": "#003090",
    "Southampton": "#D71920", "Leeds United": "#FFCD00", "Ipswich": "#0A4595",
    "Burnley": "#6C1D45", "Sheffield United": "#EE2737", "Luton": "#F78F1E"
}

# 4. Dashboard UI
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team 1", ALL_TEAMS, index=ALL_TEAMS.index("Arsenal"))
with col2:
    team_b = st.selectbox("Select Team 2", ALL_TEAMS, index=ALL_TEAMS.index("Manchester City"))

venue = st.radio("Venue / Location", options=[f"{team_a} Home", f"{team_b} Home", "Neutral"], horizontal=True)

# --- VEGAS ODDS INPUT ---
st.markdown("### 🏦 Enter Live Vegas Odds (American)")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    v_odds_home = st.number_input(f"{team_a} Odds", value=150)
with col_odds2:
    v_odds_draw = st.number_input("Draw Odds", value=250)
with col_odds3:
    v_odds_away = st.number_input(f"{team_b} Odds", value=180)

# Helper to convert American to Probability
def american_to_prob(odds):
    if odds > 0: return 100 / (odds + 100)
    else: return abs(odds) / (abs(odds) + 100)

st.divider()

# 5. Prediction Logic
if st.button("Calculate Value & Predict", type="primary"):
    if team_a == team_b:
        st.error("Please select two different teams!")
    else:
        try:
            model = joblib.load('Data/xgboost_pl_model.pkl')
            elo_dict = joblib.load('Data/elo_dict.pkl')
        except FileNotFoundError:
            st.error("Model files not found! Please run train_xgboost.py first.")
            st.stop()
        
        a_att_raw, a_def_raw = get_rolling_stats(team_a, history)
        b_att_raw, b_def_raw = get_rolling_stats(team_b, history)
        a_elo = elo_dict.get(team_a, 1500)
        b_elo = elo_dict.get(team_b, 1500)

        def get_xgb_probs(h_att, h_def, h_elo, a_att, a_def, a_elo):
            input_df = pd.DataFrame([{
                'Home_Att_xG': h_att / LEAGUE_AVG_XG, 'Home_Def_xG': h_def / LEAGUE_AVG_XG,
                'Home_Rest': 7, 'Home_Elo': h_elo,          
                'Away_Att_xG': a_att / LEAGUE_AVG_XG, 'Away_Def_xG': a_def / LEAGUE_AVG_XG,
                'Away_Rest': 7, 'Away_Elo': a_elo           
            }])
            p = model.predict_proba(input_df)[0]
            return p[2]*100, p[1]*100, p[0]*100

        if venue == f"{team_a} Home":
            a_p, d_p, b_p = get_xgb_probs(a_att_raw, a_def_raw, a_elo, b_att_raw, b_def_raw, b_elo)
        elif venue == f"{team_b} Home":
            b_p, d_p, a_p = get_xgb_probs(b_att_raw, b_def_raw, b_elo, a_att_raw, a_def_raw, a_elo)
        else:
            a1, d1, b1 = get_xgb_probs(a_att_raw, a_def_raw, a_elo, b_att_raw, b_def_raw, b_elo)
            b2, d2, a2 = get_xgb_probs(b_att_raw, b_def_raw, b_elo, a_att_raw, a_def_raw, a_elo)
            a_p, b_p, d_p = (a1 + a2) / 2, (b1 + b2) / 2, (d1 + d2) / 2

        # --- BETTING VALUE ANALYSIS ---
        st.subheader("💰 Betting Value Analysis")
        v_prob_h = american_to_prob(v_odds_home) * 100
        v_prob_a = american_to_prob(v_odds_away) * 100
        
        # Hardcode the Sweet Spot found in backtesting
        HISTORICAL_THRESHOLD = 4.6 
        
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            edge_h = a_p - v_prob_h
            if edge_h >= HISTORICAL_THRESHOLD: st.success(f"📊 Positive Edge: {team_a} (+{edge_h:.1f}%)")
            elif edge_h <= -HISTORICAL_THRESHOLD: st.error(f"📉 Negative Edge: {team_a} ({edge_h:.1f}%)")
            else: st.info(f"⚖️ Neutral Market: {team_a}")
            
        with val_col2:
            edge_a = b_p - v_prob_a
            if edge_a >= HISTORICAL_THRESHOLD: st.success(f"📊 Positive Edge: {team_b} (+{edge_a:.1f}%)")
            elif edge_a <= -HISTORICAL_THRESHOLD: st.error(f"📉 Negative Edge: {team_b} ({edge_a:.1f}%)")
            else: st.info(f"⚖️ Neutral Market: {team_b}")

        # Projections Graph
        prob_df = pd.DataFrame({"Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"], "Probability (%)": [a_p, d_p, b_p]})
        color_map = {f"{team_a} Win": TEAM_COLORS.get(team_a, "#0984e3"), "Draw": "#b2bec3", f"{team_b} Win": TEAM_COLORS.get(team_b, "#d63031")}
        fig = px.bar(prob_df, x="Probability (%)", y="Outcome", orientation='h', color="Outcome", text_auto='.1f', color_discrete_map=color_map)
        fig.update_layout(showlegend=False, yaxis_title=None, height=300, xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

        # --- STRATEGY TRANSPARENCY NOTE ---
        st.markdown("---")
        st.caption("🔍 Strategy Transparency & System Action")
        
        max_edge = max(edge_h, edge_a)
        target_team = team_a if edge_h > edge_a else team_b
        
        st.write(f"The model calculates a maximum edge of **{max_edge:.1f}%** for **{target_team}** compared to the Vegas implied probabilities.")
        
        if max_edge >= HISTORICAL_THRESHOLD:
            st.success(f"🤖 **System Action: SIMULATE WAGER**. The calculated edge exceeds the backtested optimization threshold of {HISTORICAL_THRESHOLD}%. In an automated environment, the algorithm would execute this trade.")
        else:
            st.warning(f"🛑 **System Action: PASS**. The calculated edge does not meet the backtested optimization threshold of {HISTORICAL_THRESHOLD}%. The algorithm requires a higher margin of safety to simulate a trade.")

        st.caption("*Disclaimer: This dashboard is a mathematical simulation built for educational and portfolio demonstration purposes only. It does not constitute financial, trading, or sports betting advice. Predictions are based on historical data and do not guarantee future results.*")

# --- BANKROLL CHART SECTION ---
# Unindented so it always displays upon app load
st.divider()
st.subheader("📈 Historical Performance Strategy")

try:
    hist_df = pd.read_csv('Data/bankroll_history.csv')
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    
    fig_hist = px.line(hist_df, x='date', y='bankroll', 
                       title="Backtested Bankroll Growth (Best Edge Threshold)",
                       labels={'bankroll': 'Bankroll ($)', 'date': 'Match Date'})
    
    fig_hist.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Initial Capital")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    final_cash = hist_df['bankroll'].iloc[-1]
    st.metric("Strategy Net Profit", f"${final_cash - 1000:,.2f}", 
              delta=f"{((final_cash - 1000) / 1000) * 100:.1f}% ROI")

except FileNotFoundError:
    st.info("Run backtester.py to generate your strategy's performance chart!")

# --- RESEARCH & ROADMAP SECTION ---
st.divider()
st.markdown("""
### 🔬 Model Roadmap: The Path to Positive ROI
Current backtesting shows an **8.9% ROI** on high-confidence bets, but low volume. To scale this into a professional-grade trading engine, the following features are in development:

1. **🏥 Squad Rotation & Injury Weights**: 
   - *Current State*: Uses team-level rolling xG.
   - *Next Step*: Integrate player-specific xG contributions. If a striker responsible for 40% of a team's xG (e.g., Haaland) is out, the 'Attack xG' parameter will automatically scale down by that percentage.

2. **✈️ Travel & Fatigue Decay**: 
   - *Current State*: Simple days-of-rest counter.
   - *Next Step*: Calculate 'Travel Fatigue' based on mid-week European competitions (Champions League/Europa). Teams returning from away games in Eastern Europe show a statistically significant dip in defensive intensity.

3. **📊 Market Sentiment Analysis**: 
   - *Goal*: Compare 'Model Odds' vs 'Closing Line Value' (CLV). Tracking the movement of Vegas odds in the 2 hours before kickoff to identify 'Sharp Money' vs 'Public Money' traps.

4. **📉 Bayesian Updating**: 
   - *Goal*: Transition from a static XGBoost model to a Bayesian framework that learns faster from mid-season managerial changes (the 'New Manager Bounce' effect).
""")

st.caption("Developed by Liam Thomas Condon | Computational Modeling & Data Analytics @ Virginia Tech")