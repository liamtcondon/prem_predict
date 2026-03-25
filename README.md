Markdown
# ⚽ Premier League Match Predictor & Value Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-green.svg)
![Data Science](https://img.shields.io/badge/CMDA-Virginia%20Tech-maroon.svg)

An end-to-end Machine Learning pipeline and interactive dashboard designed to identify Expected Value (EV) in Premier League betting markets. 

Developed by **Liam Condon** | Computational Modeling & Data Analytics (CMDA) @ Virginia Tech.

---

## 📖 Project Overview
Beating the Vegas sports betting market requires more than just predicting winners; it requires finding mathematical inefficiencies in the odds. This project utilizes an **XGBoost classification model** trained on historical Premier League data (Expected Goals, Elo ratings, and fatigue metrics) to generate independent win probabilities. 

Instead of acting as a standard "tipster," the application acts as a **Value Engine**. It compares its internal AI probabilities against live Vegas implied odds to identify statistically significant market edges.

## ⚙️ Core Architecture & Features

### 1. The Machine Learning Engine (`xgboost_pl_model.pkl`)
* **Algorithm:** XGBoost Classifier.
* **Features:** Rolling 5-game Expected Goals (xG) for and against, historical team Elo ratings, and days of rest.
* **Output:** A three-way probability matrix (Home Win, Draw, Away Win).

### 2. The Backtesting Simulator (`backtester.py`)
* **Grid Search Optimization:** Iterates through various confidence thresholds (0.0% to 5.0%) to identify the historical "Sweet Spot" for maximum profit.
* **Transparency Metrics:** Tracks model bias, identifying strong performance in Home Win predictions vs. Away Win predictions.
* **Financial Modeling:** Simulates a $1,000 starting bankroll over the last 380 Premier League matches, factoring in a 5% Vegas Vig (house edge).

### 3. The Live Dashboard (`app.py`)
* **Framework:** Built with Streamlit for a fast, responsive user interface.
* **Real-Time EV Calculation:** Users input live American odds, and the app instantly calculates if the market is offering a "Positive Edge" based on the model's optimized thresholds.
* **Risk Management:** Replaces prescriptive advice with an objective "System Action" simulator, acknowledging mathematical liability and variance.

---

## 🔬 Key Learnings & Data Realities
During the backtesting phase, grid search optimization revealed that a **4.6% Edge Threshold** was required to find profitable variance. 

**The CMDA Takeaway:** The betting market is highly efficient. A naive model that bets on every game yields a heavy negative ROI due to the bookmaker's "Vig." By implementing strict value-based filters, the model transitions from a high-volume/negative-yield strategy to a low-volume/positive-variance strategy. The model currently exhibits a strong predictive bias toward Home Field Advantage.

---

## 🚀 Model Roadmap (Future Work)
To scale this into a professional-grade quantitative trading engine, the following features are in development:

1. **Squad Rotation & Injury Weights**: Integrating player-specific xG contributions to adjust team strength dynamically when star players are absent.
2. **Travel & Fatigue Decay**: Calculating 'Travel Fatigue' based on mid-week European competitions (Champions League/Europa) rather than simple days-of-rest.
3. **Market Sentiment Analysis**: Comparing 'Model Odds' vs 'Closing Line Value' (CLV) to track sharp money movement.
4. **Bayesian Updating**: Transitioning to a Bayesian framework to learn faster from mid-season managerial changes.

---

## 💻 How to Run Locally

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy xgboost streamlit plotly bs4 requests scipy
3. Run the backtester to generate the latest optimal thresholds and bankroll history:
   ```bash
   python backtester.py
4. Launch the Streamlit Dashboard:
   ```bash
   streamlit run app.py
Disclaimer: This project is a mathematical simulation built for educational and portfolio demonstration purposes only. It does not constitute financial, trading, or sports betting advice. Predictions are based on historical data and do not guarantee future results.
