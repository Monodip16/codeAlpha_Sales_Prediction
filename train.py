"""
Master Production Model Training and Benchmarking Pipeline
Run: python train.py
"""

import os
import sys
import json
import pandas as pd

from src.data_prep import get_train_test_data
from src.models import train_models, save_production_model
from src.evaluate import evaluate_all, save_diagnostic_plots
from src.optimizer import calculate_channel_elasticity, optimize_budget


def run_training_pipeline():
    print("=" * 70)
    print(">> STARTING PRODUCTION SALES PREDICTION TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/5] Ingesting and Validating Advertising Data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)} | Features: {list(X_train.columns)}")

    # 2. Train Models
    print("\n[Step 2/5] Training Candidate Regression Models...")
    trained_models = train_models(X_train, y_train)

    # 3. Evaluate on Unseen Test Data
    print("\n[Step 3/5] Benchmarking Models on Test Data...")
    leaderboard_df, predictions = evaluate_all(trained_models, X_test, y_test)
    print("\n" + "-" * 70)
    print("MODEL LEADERBOARD (Test Set Accuracy):")
    print("-" * 70)
    print(leaderboard_df.to_string(index=False))
    print("-" * 70)

    # Identify Winning Model
    best_name = leaderboard_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_r2 = leaderboard_df.iloc[0]["R2 Score"]
    best_rmse = leaderboard_df.iloc[0]["RMSE"]
    best_mae = leaderboard_df.iloc[0]["MAE"]
    print(f"\n[*] Selected Production Model: '{best_name}' (R2: {best_r2:.4f}, RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f})")

    # 4. Save Artifacts & Diagnostics
    print("\n[Step 4/5] Saving Production Model Artifact & Diagnostics...")
    metadata = {
        "model_name": best_name,
        "test_r2": best_r2,
        "test_rmse": best_rmse,
        "test_mae": best_mae,
        "features": list(X_train.columns),
        "target": "Sales"
    }
    save_production_model(best_model, metadata=metadata)
    save_diagnostic_plots(y_test, predictions[best_name], model_name=best_name)

    # 5. Business Impact & Optimization
    print("\n[Step 5/5] Computing Advertising Elasticity & Budget Optimizer...")
    elasticity_df = calculate_channel_elasticity(best_model)
    print("\nChannel Elasticity Ranking:")
    print(elasticity_df.to_string(index=False))

    avg_budget = float(X_test.sum(axis=1).mean())
    opt_summary = optimize_budget(avg_budget, best_model)
    print(f"\nOptimal Budget Allocation for Sample Budget ${avg_budget:.1f}k:")
    print(opt_summary["allocation_df"].to_string(index=False))
    print(f"Projected Max Sales: ${opt_summary['maximum_projected_sales']:.2f}k (ROAS: {opt_summary['estimated_roas']}x)")

    print("\n" + "=" * 70)
    print(">> PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
