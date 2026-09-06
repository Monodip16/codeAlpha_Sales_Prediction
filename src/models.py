"""
Model Training, Registry, and Persistence Module
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_model_registry(random_state=42):
    """
    Returns candidate model dictionary with tuned hyperparameters.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=0.1, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        ),
    }


def train_models(X_train, y_train, random_state=42):
    """
    Fit all registered candidate models on training data.
    """
    models = get_model_registry(random_state=random_state)
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def save_production_model(model, model_path="models/sales_model.joblib", metadata=None, meta_path="models/model_metrics.json"):
    """
    Save the winning production model and optional metadata.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[+] Model artifact saved to: {model_path}")

    if metadata is not None:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Model metadata saved to: {meta_path}")


def load_production_model(model_path="models/sales_model.joblib"):
    """
    Load saved production model from disk.
    """
    if not os.path.exists(model_path):
        # Check alternative base relative path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    return joblib.load(model_path)
