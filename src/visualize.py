import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_feature_importance(model, feature_names):
    """Plots relative importance scores for trained model features."""
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='#3498db', edgecolor='black')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Relative Importance Score')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plots confusion matrix heatmap for model predictions."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down (0)', 'Up (1)'], 
                yticklabels=['Down (0)', 'Up (1)'])
    plt.xlabel('Predicted Direction')
    plt.ylabel('Actual Direction')
    plt.title('Confusion Matrix Breakdown')
    plt.tight_layout()
    plt.show()