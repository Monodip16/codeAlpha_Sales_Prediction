# Sales Prediction & Marketing ROI Optimization Engine
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, enterprise-ready Machine Learning system that forecasts product sales revenue based on multi-channel advertising expenditures (TV, Radio, Newspaper) and mathematically calculates the optimal budget allocation across channels using constrained non-linear optimization (SLSQP).

---

## 🏗️ System Architecture

```
Sales-prediction/
├── api/
│   └── main.py                     # Production FastAPI microservice with Swagger docs
├── app/
│   └── app.py                      # Interactive Streamlit Web UI & "What-If" simulator
├── data/
│   └── Advertising.csv             # Benchmark marketing dataset (200 records)
├── models/
│   ├── sales_model.joblib          # Serialized production Gradient Boosting model
│   └── model_metrics.json          # Production evaluation metadata & schema
├── notebooks/
│   └── Sales_prediction_Analysis.ipynb # End-to-end exploratory data analysis & diagnostics
├── reports/
│   └── residual_diagnostics.png    # Validation plots (Actual vs Predicted, Residual Normality)
├── src/
│   ├── __init__.py
│   ├── data_prep.py                # Schema validation, data cleaning & partitioning
│   ├── models.py                   # Model registry, training, serialization
│   ├── evaluate.py                 # R², RMSE, MAE, MAPE scoring & diagnostics
│   ├── predict.py                  # Production inference engine (single & batch)
│   └── optimizer.py                # Elasticity calculation & SLSQP budget optimizer
├── tests/
│   └── test_pipeline.py            # Automated unit & integration tests
├── Dockerfile                      # Production container configuration
├── requirements.txt                # Pinned production dependencies
├── train.py                        # Master training & benchmarking script
└── README.md                       # Comprehensive documentation
```

---

## 🏆 Model Performance Benchmark

Models evaluated on held-out unseen test data (80/20 train/test split, `random_state=42`):

| Rank | Model | $R^2$ Score (Accuracy) | RMSE (Error) | MAE (Mean Error) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting Regressor** | **98.05%** | **0.7849** | **0.6306** | **Deployed in Production** |
| 🥈 | **Random Forest Regressor** | **98.01%** | **0.7934** | **0.6316** | Champion Ensemble |
| 🥉 | **Lasso Regression** | **89.96%** | **1.7806** | **1.4598** | Regularized L1 Baseline |
| 4 | **Linear Regression (OLS)** | **89.94%** | **1.7816** | **1.4608** | Interpretability Baseline |
| 5 | **Ridge Regression** | **89.94%** | **1.7816** | **1.4608** | Regularized L2 Baseline |

> **Key Machine Learning Finding:** Tree-based ensemble models outperform linear models by capturing the **cross-channel interaction synergy** between TV and Radio advertising, cutting prediction error by **>55%**.

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train and Serialize the Production Model
Execute the automated training pipeline to train all models, validate accuracy, and serialize artifacts:
```bash
python train.py
```
*Outputs generated:*
- `models/sales_model.joblib`: Serialized Gradient Boosting model.
- `models/model_metrics.json`: Accuracy scores and metadata.
- `reports/residual_diagnostics.png`: Diagnostic normality and fit plots.

### 3. Run Automated Tests
Run unit tests verifying data integrity, model inference, and budget optimization:
```bash
python tests/test_pipeline.py
```

---

## 🖥️ Interactive Web Dashboard (Streamlit)

Launch the interactive UI for marketing teams:
```bash
streamlit run app/app.py
```
Open **http://localhost:8501** in your browser to access:
- **Live Sales Predictor ("What-If" Simulator):** Real-time spend sliders with instant sales forecasts and estimated Return on Ad Spend (ROAS).
- **Budget Allocation Optimizer:** Input total marketing budget (e.g. `$200k`) to calculate the optimal channel split.
- **Model Leaderboard:** Compare $R^2$ and RMSE scores across all candidate models.
- **Data Exploration:** Interactive correlation heatmaps and sample records.

---

## 🌐 Production REST API (FastAPI)

Launch the high-performance API server:
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at: **http://localhost:8000/docs**

### Sample API Request (Prediction):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}'
```
**Response:**
```json
{
  "tv_spend": 230.1,
  "radio_spend": 37.8,
  "newspaper_spend": 69.2,
  "total_ad_spend": 337.1,
  "predicted_sales": 22.04,
  "estimated_roas": 0.065,
  "unit": "thousands ($k)"
}
```

### Sample API Request (Budget Optimization):
```bash
curl -X POST "http://localhost:8000/optimize" \
     -H "Content-Type: application/json" \
     -d '{"total_budget": 200.0}'
```

---

## 🐳 Docker Deployment

Build and run the containerized application:
```bash
# Build Docker image
docker build -t sales-prediction-engine .

# Run container (exposes API on port 8000)
docker run -p 8000:8000 sales-prediction-engine
```

---

## 💡 Strategic Marketing Insights

1. **Prioritize TV & Radio Synergy:** Broadcast awareness (TV) combined with high-frequency localized ads (Radio) delivers exponential sales conversion.
2. **Reallocate Newspaper Budget:** Newspaper advertising displays near-zero elasticity (+0.0028) and negligible feature importance.
3. **Zero-Cost Revenue Lift:** Reallocating existing funds according to the SLSQP optimization schedule delivers an estimated **8% to 15% revenue lift** with **zero increase in total marketing budget**.
