import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import scaled_bse_data

def train_pipeline(ticker="RELIANCE.NS"):
    df_features = scaled_bse_data(ticker)

    # 1. Clean feature set (Drop duplicate/collinear columns if present)
    if 'Log_Return' in df_features.columns:
        df_features = df_features.drop(columns=['Log_Return'])

    X = df_features.drop(columns=["Target"])
    y = df_features["Target"].astype(int)

    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"\nData Shape -> Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Latest Date: {df_features.index[-1].strftime('%Y-%m-%d')}")

    # 3. XGBoost with Regularization (Fixes Overfitting on Financial Noise)
    xgb_model = XGBClassifier(
        n_estimators=50,      
        learning_rate=0.01,    
        max_depth=2,           # Shallow depth (2 levels) stops overfitting
        subsample=0.7,         # Train on 70% random rows per tree
        colsample_bytree=0.7,  # Train on 70% random features per tree
        gamma=1.0,             # Minimum loss reduction required to make a split
        reg_alpha=0.1,  
        scale_pos_weight=1.0,   # Balances weight between UP and DOWN days
        min_child_weight=3,     # Requires at least 3 samples per leaf (prevents noise fitting)
        reg_lambda=1.0,# L1 regularization on weights
        eval_metric="logloss",
        random_state=42
    )
    
    # Train directly on unscaled data (XGBoost is scale-invariant)
    xgb_model.fit(X_train, y_train)

    # 4. Predict & Evaluate
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print(" Tuned XGBoost Model Performance")
    print("=" * 50)
    print(f"Test Accuracy : {acc:.2%}")
    print(f"ROC-AUC Score : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return xgb_model

if __name__ == "__main__":
    model = train_pipeline("RELIANCE.NS")