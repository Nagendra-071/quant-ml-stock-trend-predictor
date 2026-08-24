import sys
from src.preprocessing import scaled_bse_data
from src.train_model import train_pipeline
from src.backtest import run_backtest, plot_equity_curve


def main(ticker="RELIANCE.NS"):
    print(f"Starting Quant ML Pipeline for: {ticker}\n")

    # 1. Load Preprocessed Data
    df_feats = scaled_bse_data(ticker)
    if "Log_Return" in df_feats.columns:
        df_feats = df_feats.drop(columns=["Log_Return"])

    # 2. Train Model & Run Cross-Validation
    xgb_model = train_pipeline(ticker)

    # 3. Backtest Strategy
    df_bt = run_backtest(df_feats, xgb_model)

    # 4. Display Results Plot
    plot_equity_curve(df_bt,ticker)
    


if __name__ == "__main__":
    ticker_input = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    main(ticker_input)