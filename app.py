import streamlit as st
import pandas as pd
import numpy as np  
import joblib
import plotly.express as px
from datetime import datetime, timedelta
from monte_carlo import generate_season_projections

# --- PAGE CONFIG ---
st.set_page_config(page_title="Premier League ML Engine", page_icon="⚽", layout="wide")

# --- LOAD AI MODELS & DATA ---
@st.cache_resource
def load_brains():
    model = joblib.load('Data/xgboost_pl_model.pkl')
    league_avg_xg = joblib.load('Data/league_avg_xg.pkl')
    elos = joblib.load('Data/elo_dict.pkl')
    return model, league_avg_xg, elos

model, league_avg_xg, current_elos = load_brains()

# --- DATA PREP (Global) ---
sched_df = pd.read_csv('Data/global_schedule.csv')
sched_df['date'] = pd.to_datetime(sched_df['date'], utc=True)

history_df = pd.read_csv('Data/live_pl_history.csv')
history_df['match_date'] = pd.to_datetime(history_df['match_date'], utc=True)

# --- HELPER: GET TEAM FORM ---
def get_team_form(team_name):
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
            
    if not xg_f: return 1.3, 1.3
    return np.mean(xg_f) / league_avg_xg, np.mean(xg_a) / league_avg_xg

# --- TEAM COLOR MAPPING ---
team_colors = {
    'Arsenal': '#EF0107', 'Aston Villa': '#95BFE5', 'AFC Bournemouth': '#DA291C',
    'Brentford': '#E30613', 'Brighton & Hove Albion': '#0057B8', 'Burnley': '#6C1D45',
    'Chelsea': '#034694', 'Crystal Palace': '#1B458F', 'Everton': '#003399',
    'Fulham': '#FFFFFF', 'Ipswich Town': '#0033FF', 'Leeds United': '#FFCD00',
    'Leicester City': '#003090', 'Liverpool': '#C8102E', 'Luton Town': '#F78F1E',
    'Manchester City': '#6CABDD', 'Manchester United': '#DA291C', 'Newcastle United': '#241F20',
    'Nottingham Forest': '#DD0000', 'Sheffield United': '#EE2737', 'Southampton': '#D71920',
    'Sunderland': '#FF0000', 'Tottenham Hotspur': '#132257', 'Watford': '#FBEE23',
    'West Ham United': '#7A263A', 'Wolverhampton Wanderers': '#FDB913', 'Norwich City': '#FFF200'
}

# --- APP HEADER ---
st.title("⚽ Premier League Market Analysis Engine")
st.markdown("**Powered by XGBoost & Expected Goals (xG) Analytics**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Today's Edge (Live Matches)", "Historical Backtester", "Season Projections"])

# ==========================================
# TAB 1: TODAY'S EDGE (Interactive Calendar)
# ==========================================
with tab1:
    if 'view_date' not in st.session_state:
        st.session_state.view_date = datetime.now().date()

    # --- CALENDAR CONTROLS ---
    col_left, col_mid, col_right = st.columns([1, 3, 1])

    with col_left:
        if st.button("⬅️ Previous Day"):
            st.session_state.view_date -= timedelta(days=1)
            st.rerun()

    with col_mid:
        # FIX: Removed key='calendar_picker' so it doesn't fight the buttons
        new_date = st.date_input("Select Date", value=st.session_state.view_date, label_visibility="collapsed")
        if new_date != st.session_state.view_date:
            st.session_state.view_date = new_date
            st.rerun()
        st.markdown(f"<h3 style='text-align: center;'>{st.session_state.view_date.strftime('%A, %b %d, %Y')}</h3>", unsafe_allow_html=True)

    with col_right:
        if st.button("Next Day ➡️"):
            st.session_state.view_date += timedelta(days=1)
            st.rerun()
    st.divider()

    # --- DATA NORMALIZATION ---
    def standardize(name):
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
        return mapping.get(name, name)

    target_dt = pd.Timestamp(st.session_state.view_date).normalize()
    now_dt = pd.Timestamp.now().normalize()
    
    sched_df['date_only'] = pd.to_datetime(sched_df['date']).dt.tz_localize(None).dt.normalize()
    history_df['match_date_only'] = pd.to_datetime(history_df['match_date']).dt.tz_localize(None).dt.normalize()
    
    day_matches = sched_df[
        (sched_df['league'] == 'ENG-Premier League') & 
        (sched_df['date_only'] == target_dt)
    ]

    if day_matches.empty:
        st.info(f"No Premier League matches scheduled for this date.")
    else:
        for index, row in day_matches.iterrows():
            home_raw, away_raw = row['home_team'], row['away_team']
            home_std, away_std = standardize(home_raw), standardize(away_raw)
            
            match_check = history_df[
                (history_df['match_date_only'] == target_dt) & 
                (history_df['home_team'] == home_std)
            ]
            is_verified = not match_check.empty
            
            # --- ALWAYS CALCULATE AI PREDICTION ---
            h_att, h_def = get_team_form(home_std)
            a_att, a_def = get_team_form(away_std)
            
            X_live = pd.DataFrame([{
                'Home_Att_xG': h_att, 'Home_Def_xG': h_def, 'Home_Rest': 7, 
                'Home_Elo': current_elos.get(home_std, 1500),
                'Away_Att_xG': a_att, 'Away_Def_xG': a_def, 'Away_Rest': 7, 
                'Away_Elo': current_elos.get(away_std, 1500)
            }])
            
            probs = model.predict_proba(X_live)[0] # [Away, Draw, Home]
            
            # Style Configurations
            h_color = team_colors.get(home_raw, '#555555')
            a_color = team_colors.get(away_raw, '#555555')
            
            dark_text_teams = ['Fulham', 'Wolverhampton Wanderers', 'Leeds United', 'Watford', 'Luton Town', 'Manchester City', 'Aston Villa', 'Norwich City']
            h_text_color = '#000000' if home_raw in dark_text_teams else '#FFFFFF'
            a_text_color = '#000000' if away_raw in dark_text_teams else '#FFFFFF'

            # --- RENDER LOGIC (FLATTENED HTML FIX) ---
            if target_dt < now_dt:
                if is_verified:
                    h_score = int(match_check.iloc[0]['home_goals'])
                    a_score = int(match_check.iloc[0]['away_goals'])
                else:
                    h_score = int(row['home_score']) if pd.notna(row['home_score']) else 0
                    a_score = int(row['away_score']) if pd.notna(row['away_score']) else 0
                
                h_border = "border: 4px solid #FFD700; box-shadow: 0px 0px 15px #FFD700;" if h_score > a_score else "border: 2px solid #333;"
                a_border = "border: 4px solid #FFD700; box-shadow: 0px 0px 15px #FFD700;" if a_score > h_score else "border: 2px solid #333;"

                # Flat string to avoid Streamlit Markdown parsing bugs
                middle_section = (
                    f"<div style='font-size: 28px; font-weight: 900; color: white;'>{h_score} - {a_score}</div>"
                    f"<div style='font-size: 11px; color: #00ff87; margin-bottom: 8px; font-weight: bold;'>✅ FINAL</div>"
                    f"<div style='font-size: 13px; color: #aaa; background: #222; padding: 4px 8px; border-radius: 5px; border: 1px solid #444; white-space: nowrap;'>AI Draw: {probs[1]*100:.1f}%</div>"
                )
            else:
                highest_prob = np.max(probs)
                h_border = "border: 4px solid #FFD700; box-shadow: 0px 0px 15px #FFD700;" if probs[2] == highest_prob else "border: 2px solid #333;"
                a_border = "border: 4px solid #FFD700; box-shadow: 0px 0px 15px #FFD700;" if probs[0] == highest_prob else "border: 2px solid #333;"

                # Flat string
                middle_section = (
                    f"<div style='font-weight: 900; font-size: 24px; color: white; margin-bottom: 15px;'>VS</div>"
                    f"<div style='font-size: 13px; color: #aaa; background: #222; padding: 4px 8px; border-radius: 5px; border: 1px solid #444; white-space: nowrap;'>Draw: {probs[1]*100:.1f}%</div>"
                )

            # Flat master card
            html_card = (
                f"<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px;'>"
                f"<div style='box-sizing: border-box; flex: 3; background-color: {h_color}; {h_border} padding: 15px; border-radius: 12px; text-align: center; color: {h_text_color}; font-weight: bold;'>"
                f"<div style='font-size: 18px;'>{home_raw}</div>"
                f"<div style='font-size: 24px; margin-top: 5px;'>{probs[2]*100:.1f}%</div>"
                f"</div>"
                f"<div style='box-sizing: border-box; flex: 2; text-align: center; display: flex; flex-direction: column; align-items: center;'>"
                f"{middle_section}"
                f"</div>"
                f"<div style='box-sizing: border-box; flex: 3; background-color: {a_color}; {a_border} padding: 15px; border-radius: 12px; text-align: center; color: {a_text_color}; font-weight: bold;'>"
                f"<div style='font-size: 18px;'>{away_raw}</div>"
                f"<div style='font-size: 24px; margin-top: 5px;'>{probs[0]*100:.1f}%</div>"
                f"</div>"
                f"</div>"
            )
            
            st.markdown(html_card, unsafe_allow_html=True)
            st.divider()

# ==========================================
# TAB 2: HISTORICAL BACKTESTER
# ==========================================
with tab2:
    st.header("Model Evaluation vs. Vegas")
    try:
        bankroll_df = pd.read_csv('Data/bankroll_history.csv')
        bankroll_df['date'] = pd.to_datetime(bankroll_df['date'])
        
        st.write("Tracking a starting bankroll of $1,000 using the optimized mathematical edge threshold.")
        
        st.subheader("📈 AI Betting Bankroll Growth")
        fig = px.line(bankroll_df, x='date', y='bankroll')
        fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Starting Bankroll")
        fig.update_layout(xaxis_title="Timeline", yaxis_title="Bankroll ($)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💰 Backtesting Performance Metrics")
        
        bankroll_changes = bankroll_df['bankroll'].diff().fillna(0)
        bets_placed = (bankroll_changes.abs() > 0.01).sum()
        winning_bets = (bankroll_changes > 0.01).sum()
        total_profit = bankroll_df['bankroll'].iloc[-1] - 1000.0
        
        strike_rate = (winning_bets / bets_placed) * 100 if bets_placed > 0 else 0
        roi = (total_profit / (bets_placed * 100)) * 100 if bets_placed > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Profit", f"${total_profit:,.2f}")
        col2.metric("ROI", f"{roi:.1f}%")
        col3.metric("Bets Placed", int(bets_placed))
        col4.metric("Strike Rate", f"{strike_rate:.1f}%")
        
    except FileNotFoundError:
        st.warning("⚠️ Warning: Run `python backtester.py` locally to generate the betting history data.")

# ==========================================
# TAB 3: SEASON PROJECTIONS (MONTE CARLO)
# ==========================================
with tab3:
    st.header("🔮 End of Season Probability Matrix")
    st.write("Simulating the remainder of the 2025/2026 season 10,000 times using live XGBoost probabilities.")

    @st.cache_data(ttl=86400) 
    def get_cached_projections():
        return generate_season_projections(n_sims=10000)

    with st.spinner("Calculating 10,000 Vectorized Universes..."):
        rank_df = get_cached_projections()

    def format_prob(val):
        if val == 0: return ""
        elif val < 0.1: return "<0.1"
        else: return f"{val:.1f}"

    st.markdown("<h3 style='text-align: center; margin-bottom: -20px;'>Table Position</h3>", unsafe_allow_html=True)
    rank_df.index.name = "Team"
    styled_df = rank_df.style.format(format_prob)\
                     .background_gradient(cmap='Greens', axis=None, vmin=0, vmax=100)

    st.dataframe(styled_df, use_container_width=True, height=750)