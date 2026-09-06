"""
Production FastAPI REST Service for Sales Prediction & Marketing Optimization
Run: uvicorn api.main:app --reload --port 8000
"""

import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_sales, predict_batch, get_model
from src.optimizer import optimize_budget, calculate_channel_elasticity

app = FastAPI(
    title="Sales Prediction & Marketing ROI Engine API",
    description="Production REST API for forecasting sales from advertising spend and optimizing marketing budgets.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class CampaignInput(BaseModel):
    TV: float = Field(..., ge=0.0, description="TV Advertising spend in thousands ($k)", example=230.1)
    Radio: float = Field(..., ge=0.0, description="Radio Advertising spend in thousands ($k)", example=37.8)
    Newspaper: float = Field(..., ge=0.0, description="Newspaper Advertising spend in thousands ($k)", example=69.2)


class PredictionResponse(BaseModel):
    tv_spend: float
    radio_spend: float
    newspaper_spend: float
    total_ad_spend: float
    predicted_sales: float
    estimated_roas: float
    unit: str


class OptimizationInput(BaseModel):
    total_budget: float = Field(..., gt=0.0, description="Total marketing budget to allocate in thousands ($k)", example=200.0)


class ChannelAllocation(BaseModel):
    Channel: str
    Optimal_Spend_k: float = Field(..., alias="Optimal_Spend($k)")
    Allocation_Pct: float = Field(..., alias="Allocation_Pct(%)")


class OptimizationResponse(BaseModel):
    total_budget: float
    maximum_projected_sales: float
    estimated_roas: float
    allocation: List[dict]


# Endpoints
@app.get("/", tags=["General"])
def root():
    return {
        "service": "Sales Prediction & Budget Optimization API",
        "status": "online",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
def health_check():
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model unavailable: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def forecast_sales(campaign: CampaignInput):
    """
    Predict sales and estimated ROAS for a given advertising campaign budget.
    """
    try:
        result = predict_sales(
            tv_spend=campaign.TV,
            radio_spend=campaign.Radio,
            newspaper_spend=campaign.Newspaper
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", tags=["Inference"])
def forecast_sales_batch(campaigns: List[CampaignInput]):
    """
    Batch prediction endpoint for evaluating multiple marketing budget combinations.
    """
    import pandas as pd
    try:
        data = [c.dict() for c in campaigns]
        df = pd.DataFrame(data)
        res_df = predict_batch(df)
        return res_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/optimize", tags=["Optimization"])
def get_optimal_budget(req: OptimizationInput):
    """
    Calculate the mathematically optimal spend distribution across TV, Radio, and Newspaper
    to maximize revenue for a given total budget constraint.
    """
    try:
        opt = optimize_budget(req.total_budget)
        return {
            "total_budget": opt["total_budget"],
            "maximum_projected_sales": opt["maximum_projected_sales"],
            "estimated_roas": opt["estimated_roas"],
            "allocation": opt["allocation"]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/elasticity", tags=["Optimization"])
def get_elasticity_rankings():
    """
    Get the advertising elasticity of demand and sensitivity ranking for each medium.
    """
    try:
        df = calculate_channel_elasticity()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
