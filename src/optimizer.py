"""
Advertising Impact, Elasticity, and Constrained Budget Optimization Module
Uses Scipy SLSQP to solve for the mathematically optimal ad spend allocation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.predict import get_model


def calculate_channel_elasticity(model=None, base_spend=None, delta_pct=0.10):
    """
    Computes Advertising Elasticity of Demand (AED):
    Elasticity = (% Change in Sales) / (% Change in Spend)
    """
    if model is None:
        model = get_model()

    if base_spend is None:
        base_spend = {"TV": 150.0, "Radio": 25.0, "Newspaper": 30.0}

    base_df = pd.DataFrame([base_spend])
    base_sales = float(model.predict(base_df)[0])

    elasticities = []

    for channel, val in base_spend.items():
        perturbed = base_spend.copy()
        perturbed[channel] = val * (1 + delta_pct)
        new_df = pd.DataFrame([perturbed])
        new_sales = float(model.predict(new_df)[0])

        pct_sales_change = (new_sales - base_sales) / base_sales
        elasticity = pct_sales_change / delta_pct

        elasticities.append({
            "Channel": channel,
            "Base_Spend($k)": val,
            "Elasticity": round(float(elasticity), 4),
            "Sales_Lift_on_+10%_Spend(%)": round(float(pct_sales_change * 100), 2)
        })

    return pd.DataFrame(elasticities).sort_values(by="Elasticity", ascending=False).reset_index(drop=True)


def optimize_budget(total_budget: float, model=None, bounds=None):
    """
    Solves for the optimal spend in [TV, Radio, Newspaper] to maximize Sales
    subject to: TV + Radio + Newspaper == total_budget.
    """
    if model is None:
        model = get_model()

    total_budget = float(total_budget)
    channels = ["TV", "Radio", "Newspaper"]
    n_channels = len(channels)

    # Initial guess: equal distribution
    x0 = np.full(n_channels, total_budget / n_channels)

    # Objective: Minimize negative sales (maximizes predicted sales)
    def objective(spends):
        input_df = pd.DataFrame([{"TV": spends[0], "Radio": spends[1], "Newspaper": spends[2]}])
        return -float(model.predict(input_df)[0])

    # Constraint: sum of channel spends == total_budget
    constraint = {"type": "eq", "fun": lambda s: np.sum(s) - total_budget}

    # Bounds per channel: [0, total_budget]
    if bounds is None:
        bounds = [(0, total_budget) for _ in range(n_channels)]

    opt_result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        options={"maxiter": 150, "ftol": 1e-4}
    )

    optimal_spends = opt_result.x
    max_sales = -opt_result.fun

    summary = []
    for i, ch in enumerate(channels):
        summary.append({
            "Channel": ch,
            "Optimal_Spend($k)": round(float(optimal_spends[i]), 2),
            "Allocation_Pct(%)": round(float((optimal_spends[i] / total_budget) * 100), 1)
        })

    alloc_df = pd.DataFrame(summary).sort_values(by="Optimal_Spend($k)", ascending=False).reset_index(drop=True)

    return {
        "total_budget": round(total_budget, 2),
        "maximum_projected_sales": round(float(max_sales), 2),
        "estimated_roas": round(float(max_sales / total_budget), 3),
        "allocation": alloc_df.to_dict(orient="records"),
        "allocation_df": alloc_df
    }
