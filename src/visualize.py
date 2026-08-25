import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_feature_importance(model, feature_names, top_n=15):
    """Plots relative importance scores for trained model features."""
    #  Convert feature_names to list to ensure proper array alignment
    feature_names = list(feature_names)
    importances = model.feature_importances_

    # created a checkfor feature dimension length to prevent shape mismatch crashes
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature count mismatch: model has {len(importances)} features, "
            f"but {len(feature_names)} feature names were provided."
        )

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    
    # Sort and slice to top N features to keep plot clean and readable
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(
        df_imp["Feature"],
        df_imp["Importance"],
        color="#3498db",
        edgecolor="black",
    )
    plt.title(f"Top {top_n} XGBoost Feature Importance", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance Score")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plots confusion matrix heatmap for model predictions."""
    #  Ensure clean array alignment without index or NaN issues
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Down (0)", "Up (1)"],
        yticklabels=["Down (0)", "Up (1)"],
    )
    plt.xlabel("Predicted Direction")
    plt.ylabel("Actual Direction")
    plt.title("Confusion Matrix Breakdown", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()