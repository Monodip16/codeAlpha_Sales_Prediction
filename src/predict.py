"""
Production Inference & Prediction Engine
"""

import os
import pandas as pd
import numpy as np
from src.models import load_production_model

# In-memory cached model
_CACHED_MODEL = None


def get_model(model_path="models/sales_model.joblib"):
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_production_model(model_path)
    return _CACHED_MODEL


def predict_sales(tv_spend: float, radio_spend: float, newspaper_spend: float, model_path="models/sales_model.joblib"):
    """
    Predict sales for a single advertising campaign budget scenario.
    All spends are in thousands of dollars ($k).
    Returns a dict with forecasted sales, total spend, and estimated ROAS.
    """
    # Validation
    tv = max(float(tv_spend), 0.0)
    radio = max(float(radio_spend), 0.0)
    newspaper = max(float(newspaper_spend), 0.0)

    model = get_model(model_path)

    # DataFrame with strict feature order
    input_df = pd.DataFrame([{"TV": tv, "Radio": radio, "Newspaper": newspaper}])
    prediction = float(model.predict(input_df)[0])

    total_spend = tv + radio + newspaper
    roas = (prediction / total_spend) if total_spend > 0 else 0.0

    return {
        "tv_spend": round(tv, 2),
        "radio_spend": round(radio, 2),
        "newspaper_spend": round(newspaper, 2),
        "total_ad_spend": round(total_spend, 2),
        "predicted_sales": round(prediction, 2),
        "estimated_roas": round(roas, 3),
        "unit": "thousands ($k)"
    }


def predict_batch(df: pd.DataFrame, model_path="models/sales_model.joblib"):
    """
    Batch prediction for multiple campaign scenarios.
    Expects DataFrame with columns: ['TV', 'Radio', 'Newspaper']
    """
    model = get_model(model_path)
    req_cols = ["TV", "Radio", "Newspaper"]

    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

    clean_df = df[req_cols].fillna(0).clip(lower=0)
    preds = model.predict(clean_df)

    res_df = clean_df.copy()
    res_df["Predicted_Sales"] = np.round(preds, 2)
    res_df["Total_Spend"] = np.round(res_df[req_cols].sum(axis=1), 2)
    res_df["ROAS"] = np.where(res_df["Total_Spend"] > 0, np.round(res_df["Predicted_Sales"] / res_df["Total_Spend"], 3), 0.0)

    return res_df
