import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(df_bt, ticker="RELIANCE.NS"):
    plt.figure(figsize=(10, 5))
    plt.plot(df_bt.index, df_bt['Strategy_Cum'], label='XGBoost Strategy', color='green', linewidth=2)
    plt.plot(df_bt.index, df_bt['Benchmark_Cum'], label=f'Buy & Hold ({ticker})', color='gray', linestyle='--')
    plt.title(f'Strategy Performance vs. Benchmark ({ticker})')
    plt.ylabel('Cumulative Growth (1.0 = Base)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_backtest(df_features, model, threshold=0.50):
    
    df_bt = df_features.dropna(subset=["Target"]).copy()
    X = df_bt.drop(columns=["Target"])
    
    #to handle misisng Daily_retrun col
    if "Daily_Return" not in df_bt.columns:
        if "Close" in df_bt.columns:
            df_bt["Daily_Return"] = df_bt["Close"].pct_change()
        elif "Log_Return" in df_bt.columns:
            df_bt["Daily_Return"] = np.exp(df_bt["Log_Return"]) - 1
        else:
            raise KeyError(
                "Neither 'Daily_Return', 'Close', nor 'Log_Return' found in df_features."
            )
            
    # Probabilistic Signal Generation
    probabilities = model.predict_proba(X)[:, 1]
    df_bt['Signal'] = (probabilities >= threshold).astype(int)
    
    # Shift signal by 1 day to prevent lookahead bias 
    df_bt['Strategy_Return'] = df_bt['Signal'].shift(1) * df_bt['Daily_Return']
    df_bt.dropna(subset=["Strategy_Return"], inplace=True)

    # Cumulative growth calculation
    df_bt['Benchmark_Cum'] = (1 + df_bt['Daily_Return']).cumprod()
    df_bt['Strategy_Cum'] = (1 + df_bt['Strategy_Return']).cumprod()

    # Metric calculations
    total_benchmark_return = df_bt['Benchmark_Cum'].iloc[-1] - 1
    total_strategy_return = df_bt['Strategy_Cum'].iloc[-1] - 1

    std_dev = df_bt['Strategy_Return'].std()
    sharpe_ratio = (df_bt['Strategy_Return'].mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0.0
    
    # Maximum Drawdown calculation
    rolling_max = df_bt['Strategy_Cum'].cummax()
    drawdown = (df_bt['Strategy_Cum'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    print("\n" + "=" * 50)
    print(f" STRATEGY BACKTEST PERFORMANCE (Threshold: {threshold})")
    print("=" * 50)
    print(f"Buy & Hold Return   : {total_benchmark_return:.2%}")
    print(f"Strategy Return     : {total_strategy_return:.2%}")
    print(f"Annualized Sharpe   : {sharpe_ratio:.2f}")
    print(f"Max Drawdown        : {max_drawdown:.2%}")
    print("=" * 50)

    return df_bt