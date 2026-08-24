# Quantitative Machine Learning Stock Predictor

An 6-week journey building a Quantitative Machine Learning pipeline to predict short-term (next day) stock price movements on the Bombay Stock Exchange (BSE) and National Stock Exchange (NSE), evaluate strategies via vectorised backtesting, and deploy an interactive dashboard.

---

## Key Features

* **Automated Data Ingestion:** Fetches historical market data dynamically using `yfinance`.
* **Feature Engineering:** Technical indicators including RSI with Exponential Moving Average (EMA) smoothing, Simple Moving Average (SMA) spreads, and rolling volatility.
* **Leakage-Free Pipeline:** Enforces strict chronological train-test splits and `StandardScaler` transformations to prevent lookahead bias.
* **Machine Learning Model:** Regularized `XGBClassifier` tuned with Walk-Forward Time Series Cross-Validation (`TimeSeriesSplit`).
* **Vectorised Backtesting Engine:** Probabilistic signal execution with shift-lagging (`shift(1)`), risk-adjusted performance metrics (Sharpe Ratio, Max Drawdown), and strategy benchmarking against Buy & Hold.
* **Interactive Dashboard:** Built with `Streamlit` and `Plotly` for dynamic stock exploration and strategy visualization.

---

## Tech Stack

* **Language:** Python 3.10+
* **Data & Analytics:** `pandas`, `numpy`, `yfinance`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Web Application:** `streamlit`

---


##  Architecture & Data Flow

```text
  [yfinance] ──> Raw Market Data
                      │
                      ▼
             [Feature Engineering]
             • EMA-smoothed RSI
             • SMA Spreads & Volatility
                      │
                      ▼
         [Chronological Train/Test Split]
             • StandardScaler (Leakage-Free)
                      │
                      ▼
            [XGBoost Classifier]
             • Walk-Forward Time Series CV
                      │
                      ▼
             [Backtesting Engine]
             • Probabilistic Signal Generation
             • Shift-Lagged Execution [shift(1)]
                      │
                      ├──────────────────────────┐
                      ▼                          ▼
             [Console CLI Output]     [Interactive Streamlit UI]



##  **Progress Log**

* **Week 1:** Defined project scope, repository organization, and environment setup.
* **Week 2:** Built automated data pipeline with `yfinance`, handled MultiIndex data gaps, engineered price return features, and applied `StandardScaler` normalization.
* **Week 3:** Engineered technical indicators (RSI with EMA smoothing, SMA spreads, volatility), created binary classification target (`Target`), fixed scaler data leakage by enforcing chronological train-test splits, and trained baseline `RandomForestClassifier`.
* **Week 4:** Transitioned to regularized `XGBClassifier`, added Walk-Forward Time Series Cross-Validation (`TimeSeriesSplit`), and introduced `visualize.py` module to plot Feature Importance and Confusion Matrices.
* **Week 5:** Built vectorised backtesting module featuring probabilistic signal generation with adjustable confidence thresholds, shift-lagged signal execution (`shift(1)`) to eliminate lookahead bias, performance metrics (Sharpe Ratio, Max Drawdown), and Equity Curve strategy benchmarking against Buy & Hold.
* **Week 6:** Developed interactive Streamlit web UI (`app.py`), integrating real-time user inputs, custom risk parameters, metrics cards, data caching, and interactive Plotly charts.

---

## **Project Structure**

```text
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data fetching & cleaning
│   ├── features.py          # Feature engineering & indicators
│   ├── model.py             # Model training & CV logic
│   ├── backtest.py          # Signal evaluation & performance analytics
│   └── visualize.py        # Confusion matrix & equity curve plotting
├── app.py                   # Streamlit interactive UI application
├── main.py                  # CLI pipeline runner
├── requirements.txt         # Project dependencies
└── README.md



**## Installation & Setup **

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/quant-ml-stock-predictor.git](https://github.com/your-username/quant-ml-stock-predictor.git)
   cd quant-ml-stock-predictor



2.** Create a virtual environment:**

Bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate



3. **Install dependencies:**

 Bash
  pip install -r requirements.txt



4. **Usage**
   Ticker Symbol Format:
   When testing Indian equities via yfinance, append suffixes accordingly:
   
   NSE Stocks: Add .NS (e.g., RELIANCE.NS, TCS.NS, INFY.NS)
   
   BSE Stocks: Add .BO (e.g., 500325.BO, 532540.BO)
   
   
   >>Run CLI Pipeline
   To run the end-to-end data pipeline, model evaluation, and backtest via command line:
   
   Bash
   python main.py
   
   >>Run Interactive Streamlit App
   To launch the interactive dashboard:
   
   Bash
   streamlit run app.py


⚠️ **Disclaimer**
This application is strictly created for educational and quantitative research purposes. It does not constitute financial advice. Algorithmic trading involves substantial risk of capital loss.

