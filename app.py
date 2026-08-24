import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import scaled_bse_data
from src.train_model import train_pipeline
from src.backtest import run_backtest

# Page Configuration
st.set_page_config(
    page_title="Quant ML Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("Quant Machine Learning Stock Trend Predictor")
st.markdown("Predict stock direction using an XGBoost Machine Learning model.")

# Stock Selection Tuple
STOCKS = (
  STOCKS = (
    "ICICIBANK.NS", "SBIN.NS", "PARADEEP.NS", "RELIANCE.NS", "HINDUNILVR.NS", "INFY.NS", 
    "BAJFINANCE.NS", "LICI.NS", "ITC.NS", "LT.NS", "MARUTI.NS", "M&M.NS", "HCLTECH.NS", 
    "KOTAKBANK.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "TITAN.NS", 
    "NTPC.NS", "BAJAJFINSV.NS", "DMART.NS", "ONGC.NS", "HAL.NS", "ADANIPORTS.NS", 
    "BEL.NS", "POWERGRID.NS", "WIPRO.NS", "ADANIENT.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", 
    "ASIANPAINT.NS", "COALINDIA.NS", "HDFCBANK.NS", "ADANIPOWER.NS", "NESTLEIND.NS", 
    "INDIGO.NS", "TATASTEEL.NS", "HYUNDAI.NS", "JIOFIN.NS", "IOC.NS", "TRENT.NS", 
    "GRASIM.NS", "DLF.NS", "HINDZINC.NS", "SBILIFE.NS", "EICHERMOT.NS", "VEDL.NS", 
    "VBL.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "HINDALCO.NS", "TVSMOTOR.NS", 
    "IRFC.NS", "PIDILITIND.NS", "ADANIGREEN.NS", "LTIM.NS", "BAJAJHLDNG.NS", 
    "AMBUJACEM.NS", "BRITANNIA.NS", "BPCL.NS", "TECHM.NS", "GODREJCP.NS", 
    "PFC.NS", "SOLARINDS.NS", "CIPLA.NS", "TATAPOWER.NS", "BANKBARODA.NS", 
    "BOSCHLTD.NS", "TORNTPHARM.NS", "CHOLAFIN.NS", "LODHA.NS", "HDFCAMC.NS", 
    "PNB.NS", "GAIL.NS", "CGPOWER.NS", "SIEMENS.NS", "MAXHEALTH.NS", 
    "MUTHOOTFIN.NS", "APOLLOHOSP.NS", "INDHOTEL.NS", "ABB.NS", "MAZDOCK.NS", 
    "SHRIRAMFIN.NS", "SHREECEM.NS", "TATACONSUM.NS", "POLYCAB.NS", "DIXON.NS", 
    "HEROMOTOCO.NS", "CUMMINSIND.NS", "DRREDDY.NS", "MANKIND.NS", "JINDALSTEL.NS", 
    "ZYDUSLIFE.NS", "MOTHERSON.NS", "HAVELLS.NS", "SWIGGY.NS", "UNIONBANK.NS", "GMBREW.NS"
)

# Sidebar Options
st.sidebar.header("User Configurations")
ticker = st.sidebar.selectbox("Select a Stock Ticker", options=STOCKS, index=3)
threshold = st.sidebar.slider(
    label="Decision Threshold", 
    min_value=0.40, 
    max_value=0.60, 
    value=0.48, 
    step=0.01,
    help="Lower values make the strategy more aggressive (takes more trades). Higher values make it more selective."
)

# Sidebar Action Button
run_btn = st.sidebar.button("Run Pipeline", type="primary")

# Caching Data Loader
@st.cache_data(ttl=3600)
def load_stock_data(symbol):
    try:
        df = scaled_bse_data(symbol)
        if "Log_Return" in df.columns:
            df = df.drop(columns=["Log_Return"])
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Caching Model Training Pipeline
@st.cache_resource
def get_trained_model(symbol):
    return train_pipeline(symbol)

# Initial State Message
if not run_btn:
    st.info("Select your parameters in the sidebar and click **'Run Pipeline'** to generate predictions.")

# Execution Triggered on Button Click
else:
    df_data = load_stock_data(ticker)

    if df_data is not None:
        with st.spinner(f"Training XGBoost Model on {ticker}..."):
            model = get_trained_model(ticker)

        # Extract Newest Row and Calculate Dates
        X_all = df_data.drop(columns=["Target"])
        latest_features = X_all.iloc[[-1]]
        last_trading_date = latest_features.index[0]
        
        # Calculate Prediction Target Date with Short Weekday Name (e.g., 'Mon')
        target_date = last_trading_date + pd.Timedelta(days=1)
        while target_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            target_date += pd.Timedelta(days=1)
        
        formatted_target_date = target_date.strftime('%Y-%m-%d (%a)')

        # Get Predicted Probability
        up_probability = float(model.predict_proba(latest_features)[:, 1][0])
        is_bullish = up_probability >= threshold

        # Next-Day Prediction Display Section
        st.markdown("---")
        st.subheader(f"Directional Forecast for {formatted_target_date}")

        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

        if is_bullish:
            col_pred1.metric(
                label="Predicted Signal", 
                value=":green[▲ UP (BUY)]", 
                delta=f"+{(up_probability * 100):.2f}% Confidence"
            )
        else:
            col_pred1.metric(
                label="Predicted Signal", 
                value=":red[▼ DOWN / CASH]", 
                delta=f"-{( (1 - up_probability) * 100):.2f}% Confidence",
                delta_color="inverse"
            )

        col_pred2.metric("Target Prediction Date", formatted_target_date)
        col_pred3.metric("Bullish Probability", f"{up_probability:.2%}")
        col_pred4.metric("Based on Data Up To", last_trading_date.strftime('%Y-%m-%d'))

        # Strategy Backtest Engine
        df_bt = run_backtest(df_data, model, threshold=threshold)

        st.markdown("---")
        st.subheader(f"Strategy Performance Metrics ({ticker})")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        strat_ret = (df_bt['Strategy_Cum'].iloc[-1] - 1)
        bench_ret = (df_bt['Benchmark_Cum'].iloc[-1] - 1)
        std_dev = df_bt['Strategy_Return'].std()
        sharpe = (df_bt['Strategy_Return'].mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0.0
        max_dd = ((df_bt['Strategy_Cum'] - df_bt['Strategy_Cum'].cummax()) / df_bt['Strategy_Cum'].cummax()).min()

        col_m1.metric("Strategy Return", f"{strat_ret:.2%}")
        col_m2.metric("Benchmark Return", f"{bench_ret:.2%}")
        col_m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col_m4.metric("Max Drawdown", f"{max_dd:.2%}")

        # Plotly Charts
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1,
            subplot_titles=("Strategy Cumulative Return vs Benchmark", "RSI_14 Technical Indicator")
        )

        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Strategy_Cum'], mode='lines', name='XGBoost Strategy', line=dict(color='#00CC96', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Cum'], mode='lines', name=f'Buy & Hold ({ticker})', line=dict(color='gray', dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_data.index, y=df_data['RSI_14'], mode='lines', name='RSI (14)', line=dict(color='#FFA15A')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=550, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)