import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────
MODEL_DIR = "model"

try:
    model       = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    scaler      = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    explainer   = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
    threshold   = joblib.load(f"{MODEL_DIR}/threshold.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
except FileNotFoundError:
    raise RuntimeError("Model files not found. Run train.py first.")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predicts customer churn probability with SHAP explanations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request schema — mirrors the feature set
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str              # "Male" / "Female"
    SeniorCitizen: int       # 0 or 1
    Partner: str             # "Yes" / "No"
    Dependents: str          # "Yes" / "No"
    tenure: int              # months
    PhoneService: str        # "Yes" / "No"
    MultipleLines: str       # "Yes" / "No"
    InternetService: str     # "DSL" / "Fiber optic" / "No"
    OnlineSecurity: str      # "Yes" / "No"
    OnlineBackup: str        # "Yes" / "No"
    DeviceProtection: str    # "Yes" / "No"
    TechSupport: str         # "Yes" / "No"
    StreamingTV: str         # "Yes" / "No"
    StreamingMovies: str     # "Yes" / "No"
    Contract: str            # "Month-to-month" / "One year" / "Two year"
    PaperlessBilling: str    # "Yes" / "No"
    PaymentMethod: str       # "Electronic check" / "Mailed check" / "Bank transfer (automatic)" / "Credit card (automatic)"
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    prediction: str
    top_reasons: List[dict]
    threshold_used: float


def preprocess(data: CustomerData) -> pd.DataFrame:
    """Convert raw input to model-ready features."""
    d = data.dict()

    # Binary Yes/No
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[col] = 1 if d[col] == "Yes" else 0

    d["gender"] = 1 if d["gender"] == "Male" else 0
    d["MultipleLines"] = 1 if d["MultipleLines"] == "Yes" else 0

    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[col] = 1 if d[col] == "Yes" else 0

    # Label encode categoricals using saved encoders
    for col in ["InternetService", "Contract", "PaymentMethod"]:
        le = label_encoders[col]
        try:
            d[col] = int(le.transform([d[col]])[0])
        except ValueError:
            raise HTTPException(status_code=400,
                detail=f"Invalid value '{d[col]}' for field '{col}'. "
                       f"Valid options: {list(le.classes_)}")

    df = pd.DataFrame([d])[feature_names]

    # Scale numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Churn Predictor API is running. POST to /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "threshold": round(threshold, 3)}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    df = preprocess(customer)

    # Prediction
    proba = float(model.predict_proba(df)[0][1])
    prediction = "Will Churn" if proba >= threshold else "Will Not Churn"

    if proba >= 0.75:
        risk = "High Risk"
    elif proba >= 0.45:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # SHAP per-prediction explanation
    shap_vals = explainer.shap_values(df)[0]  # array of shape [n_features]
    feature_impacts = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_reasons = [
        {
            "feature": feat,
            "impact": round(float(val), 4),
            "direction": "increases churn risk" if val > 0 else "decreases churn risk"
        }
        for feat, val in feature_impacts[:5]
    ]

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_level=risk,
        prediction=prediction,
        top_reasons=top_reasons,
        threshold_used=round(threshold, 3)
    )
