# Quantitative Machine Learning Stock Predictor

An 8-week journey building a Quantitative Machine Learning pipeline to predict short-term stock movements on the Bombay Stock Exchange (BSE) and National Stock Exchange (NSE).

## Progress Log

* **Week 1:** Defined project scope, repository organization, and environment configuration.
* **Week 2:** Built automated data pipeline with `yfinance`, handled data gaps/MultiIndex structures, engineered price return features, and applied `StandardScaler` normalization.
* **Week 3:** Engineered technical indicators (RSI with Exponential Moving Average (EMA) smoothing for improved accuracy, SMA spreads, volatility), created binary classification target (`Target`), fixed scaler data leakage by enforcing chronological train-test splits, and trained baseline `RandomForestClassifier`.
* **Week 4:** Transitioned to regularized `XGBClassifier`, added Walk-Forward Time Series Cross-Validation (`TimeSeriesSplit`), and introduced `visualize.py` module to plot Feature Importance and Confusion Matrices.