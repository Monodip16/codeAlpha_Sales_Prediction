"""
Automated Unit and Integration Tests for Sales Prediction Pipeline
Run: pytest tests/
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import load_and_validate_data, get_train_test_data
from src.models import train_models, save_production_model, load_production_model
from src.predict import predict_sales, predict_batch
from src.optimizer import optimize_budget, calculate_channel_elasticity


def test_data_ingestion():
    df = load_and_validate_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(["TV", "Radio", "Newspaper", "Sales"]).issubset(df.columns)
    assert df.isnull().sum().sum() == 0
    assert (df >= 0).all().all()


def test_train_test_partition():
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.20, random_state=42)
    assert len(X_train) == 160
    assert len(X_test) == 40
    assert list(X_train.columns) == ["TV", "Radio", "Newspaper"]


def test_model_training_and_saving():
    X_train, X_test, y_train, y_test = get_train_test_data()
    models = train_models(X_train, y_train)
    assert "Gradient Boosting" in models
    assert "Linear Regression" in models

    best_model = models["Gradient Boosting"]
    model_path = "models/test_model.joblib"
    save_production_model(best_model, model_path=model_path)
    assert os.path.exists(model_path)

    loaded = load_production_model(model_path=model_path)
    test_pred = loaded.predict(X_test[:2])
    assert len(test_pred) == 2


def test_single_prediction_engine():
    res = predict_sales(tv_spend=200.0, radio_spend=40.0, newspaper_spend=20.0)
    assert "predicted_sales" in res
    assert "estimated_roas" in res
    assert res["predicted_sales"] > 0
    assert res["total_ad_spend"] == 260.0


def test_batch_prediction_engine():
    batch_df = pd.DataFrame([
        {"TV": 100.0, "Radio": 20.0, "Newspaper": 10.0},
        {"TV": 250.0, "Radio": 40.0, "Newspaper": 5.0},
    ])
    res_df = predict_batch(batch_df)
    assert "Predicted_Sales" in res_df.columns
    assert len(res_df) == 2
    assert (res_df["Predicted_Sales"] > 0).all()


def test_budget_optimizer():
    budget = 200.0
    res = optimize_budget(budget)
    assert "maximum_projected_sales" in res
    assert "allocation" in res
    total_allocated = sum(item["Optimal_Spend($k)"] for item in res["allocation"])
    assert np.isclose(total_allocated, budget, atol=0.5)


def test_channel_elasticity():
    elasticity_df = calculate_channel_elasticity()
    assert isinstance(elasticity_df, pd.DataFrame)
    assert len(elasticity_df) == 3
    assert set(elasticity_df["Channel"]) == set(["TV", "Radio", "Newspaper"])


if __name__ == "__main__":
    print("[*] Running Sales Prediction Test Suite...")
    test_data_ingestion()
    print("  [+] test_data_ingestion passed.")
    test_train_test_partition()
    print("  [+] test_train_test_partition passed.")
    test_model_training_and_saving()
    print("  [+] test_model_training_and_saving passed.")
    test_single_prediction_engine()
    print("  [+] test_single_prediction_engine passed.")
    test_batch_prediction_engine()
    print("  [+] test_batch_prediction_engine passed.")
    test_budget_optimizer()
    print("  [+] test_budget_optimizer passed.")
    test_channel_elasticity()
    print("  [+] test_channel_elasticity passed.")
    print("[OK] ALL UNIT TESTS PASSED SUCCESSFULLY!")
