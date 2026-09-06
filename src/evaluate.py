"""
Model Evaluation, Metrics, and Diagnostic Plotting Module
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate R2, RMSE, MAE, and MAPE for a given prediction array.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Safe MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0

    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE(%)": round(float(mape), 2),
    }


def evaluate_all(trained_models, X_test, y_test):
    """
    Evaluate all trained models on test set and return leaderboard dataframe.
    """
    records = []
    predictions = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics = calculate_metrics(y_test, y_pred)
        records.append({
            "Model": name,
            "R2 Score": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE(%)": metrics["MAPE(%)"],
        })

    leaderboard_df = pd.DataFrame(records).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return leaderboard_df, predictions


def save_diagnostic_plots(y_test, y_pred, model_name="Gradient Boosting", save_path="reports/residual_diagnostics.png"):
    """
    Generate and save a 2-panel diagnostic figure (Actual vs Predicted & Residual Normality).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_style("whitegrid")

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, color="#1f77b4", alpha=0.75, edgecolors="k", s=45)
    min_v = min(min(y_test), min(y_pred))
    max_v = max(max(y_test), max(y_pred))
    axes[0].plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="1:1 Perfect Fit")
    axes[0].set_title(f"Actual vs Predicted - {model_name}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Sales ($k)")
    axes[0].set_ylabel("Predicted Sales ($k)")
    axes[0].legend()

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, color="#2ca02c", ax=axes[1], bins=15)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Residual Error Distribution (Normality)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Error (Actual - Predicted)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Diagnostic plots saved to: {save_path}")
