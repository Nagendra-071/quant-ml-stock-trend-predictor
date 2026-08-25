import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBClassifier

# Safe import fallback for execution from root or src/ directory
try:
    from src.preprocessing import scaled_bse_data
    from src.visualize import plot_confusion_matrix, plot_feature_importance
except ModuleNotFoundError:
    from preprocessing import scaled_bse_data
    from visualize import plot_confusion_matrix, plot_feature_importance


def evaluate_walk_forward(df_features, n_splits=5):
    """Evaluates XGBoost model performance using Walk-Forward Time Series Cross-Validation."""
    X = df_features.drop(columns=["Target"])
    y = df_features["Target"].astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_accuracies, cv_precisions, cv_aucs = [], [], []

    print("\n" + "=" * 55)
    print(f" Running {n_splits}-Fold TimeSeriesSplit Cross-Validation")
    print("=" * 55)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            min_child_weight=2,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)[:, 1]

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_fold, y_proba)

        cv_accuracies.append(acc)
        cv_precisions.append(prec)
        cv_aucs.append(auc)

        start_date = df_features.index[test_idx[0]].strftime("%Y-%m-%d")
        end_date = df_features.index[test_idx[-1]].strftime("%Y-%m-%d")

        print(
            f"Fold {fold + 1} [{start_date} -> {end_date}] | Acc: {acc:.2%} | Prec: {prec:.2%} | AUC: {auc:.4f}"
        )

    print("-" * 55)
    print(f"Mean CV Accuracy  : {np.mean(cv_accuracies):.2%}")
    print(f"Mean CV Precision : {np.mean(cv_precisions):.2%}")
    print(f"Mean CV ROC-AUC   : {np.mean(cv_aucs):.4f}")
    print("=" * 55)


def train_pipeline(ticker="RELIANCE.NS"):
    df_features = scaled_bse_data(ticker)

    # 1. Clean feature set
    if "Log_Return" in df_features.columns:
        df_features = df_features.drop(columns=["Log_Return"])
        
    df_hist = df_features.dropna(subset=["Target"]).copy()
    df_live = df_features[df_features["Target"].isna()].copy()

    # 2. Walk-Forward Evaluation
    evaluate_walk_forward(df_hist, n_splits=5)

    X = df_hist.drop(columns=["Target"])
    y = df_hist["Target"].astype(int)

    # 3. Chronological Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"\nData Shape -> Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Latest Date: {df_features.index[-1].strftime('%Y-%m-%d')}")

    # 4. XGBoost with Regularization
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=42,
        
    )

    xgb_model.fit(X_train, y_train)

    # 5. Predictions & Evaluation
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

    # Optional plot visualizers
    try:
        plot_feature_importance(xgb_model, X.columns)
        plot_confusion_matrix(y_test, y_pred)
    except Exception:
        pass

    return xgb_model


if __name__ == "__main__":
    model = train_pipeline("RELIANCE.NS")