# Customer Churn Predictor with Explainable AI

An end-to-end machine learning system that predicts customer churn probability and explains **why** each prediction was made using SHAP feature attribution.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://churn-ml-1-ui.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://churn-ml-0kud.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/rdeekshitha184/churn-ml)

---

## Live Links

| | URL |
|---|---|
| 🌐 Streamlit Dashboard | https://churn-ml-1-ui.onrender.com |
| 📡 FastAPI Backend | https://churn-ml-0kud.onrender.com |
| 📄 API Documentation | https://churn-ml-0kud.onrender.com/docs |

> **Note:** Hosted on Render free tier. If the app takes 30–60 seconds to load, the server is waking up from sleep. Please wait and refresh.

---

## Problem Statement

Telecom companies lose significant revenue when customers cancel their subscriptions (churn). Identifying at-risk customers **before** they leave allows the business to take proactive retention actions — such as offering discounts, upgrading plans, or assigning retention agents.

This project builds a production-style churn prediction system that:
- Predicts the probability that a customer will churn
- Explains the top reasons behind each prediction
- Recommends a business action based on the risk level

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (Gradient Boosting) |
| Explainability | SHAP (TreeExplainer) |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Backend API | FastAPI + Uvicorn |
| Frontend Dashboard | Streamlit |
| Deployment | Render (both services) |
| Language | Python 3.11 |

---

## ML Concepts Applied

### 1. Class Imbalance — SMOTE
The dataset has a 74/26 split (non-churn/churn). A naive model predicting "no churn" for everyone would be 74% accurate but completely useless. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic minority class samples to create a balanced training set.

```
Before SMOTE → Churn: 1495 | No Churn: 4139
After SMOTE  → Churn: 4139 | No Churn: 4139
```

### 2. Decision Threshold Tuning
Instead of using the default 0.5 classification threshold, the optimal threshold is found by maximising the F1 score on the Precision-Recall curve. This is important because in churn prediction, **false negatives** (missing a real churner) are more costly than false positives.

### 3. SHAP Explainability
SHAP (SHapley Additive Explanations) assigns each feature a contribution score for every individual prediction. This answers: *"why did the model predict this specific customer will churn?"*

- **Global SHAP** — which features matter most across all customers
- **Local SHAP** — why the model made a specific prediction for one customer

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.8303 |
| 5-Fold CV AUC | 0.9033 ± 0.0048 |
| F1 Score (Churn) | 0.63 |
| Recall (Churn) | 0.74 |
| Accuracy | 0.77 |

The high CV AUC (0.90) with low variance (±0.0048) confirms the model generalises well and is not overfitting.

---

## Dataset

**Telco Customer Churn** — IBM Sample Dataset via Kaggle

- 7,043 customer records
- 21 features (demographics, services, billing)
- Target: `Churn` (Yes/No)
- Churn rate: 26.5%

Download: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Structure

```
churn-predictor/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   ├── threshold.pkl
│   ├── feature_names.pkl
│   └── label_encoders.pkl
├── static/
│   ├── shap_global.png
│   └── shap_beeswarm.png
├── train.py          ← Data prep, model training, SHAP generation
├── api.py            ← FastAPI prediction endpoint
├── app.py            ← Streamlit dashboard
├── requirements.txt
├── render.yaml
└── README.md
```

---

## How to Run Locally

### Prerequisites
- Python 3.11
- Kaggle dataset downloaded and placed in `data/`

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train.py
```
Expected output:
```
AUC-ROC : 0.8303
5-Fold CV AUC: 0.9033 ± 0.0048
Training complete!
```

### Step 3 — Start the API (Terminal 1)
```bash
python -m uvicorn api:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`

### Step 4 — Start the dashboard (Terminal 2)
```bash
python -m streamlit run app.py
```
Dashboard available at: `http://localhost:8501`

---

## API Usage

**Endpoint:** `POST /predict`

**Sample Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0
}
```

**Sample Response:**
```json
{
  "churn_probability": 0.849,
  "risk_level": "High Risk",
  "prediction": "Will Churn",
  "top_reasons": [
    {
      "feature": "Contract",
      "impact": 0.730,
      "direction": "increases churn risk"
    },
    {
      "feature": "MonthlyCharges",
      "impact": 0.625,
      "direction": "increases churn risk"
    },
    {
      "feature": "tenure",
      "impact": 0.411,
      "direction": "increases churn risk"
    }
  ],
  "threshold_used": 0.487
}
```

---

## End-to-End Pipeline

```
Raw CSV Data
    ↓
Data Cleaning & Feature Engineering
(missing values, encoding, binary conversion)
    ↓
Train/Test Split (stratified 80/20)
    ↓
StandardScaler (numeric features)
    ↓
SMOTE (balance training classes)
    ↓
XGBoost Training
    ↓
Threshold Tuning (Precision-Recall curve)
    ↓
Model Evaluation (AUC-ROC, F1, CV)
    ↓
SHAP Explainability (global + local)
    ↓
FastAPI REST Endpoint
    ↓
Streamlit Dashboard
    ↓
Deployed on Render (live public URL)
```

---

## Business Impact

| Risk Level | Churn Probability | Recommended Action |
|---|---|---|
| 🔴 High Risk | ≥ 75% | Assign retention agent, offer annual plan discount |
| 🟡 Medium Risk | 45–75% | Send personalised loyalty offer via email |
| 🟢 Low Risk | < 45% | No immediate action needed |

---

## Author

**Deekshitha R**
B.E. Computer Science Engineering | R R Institute of Technology, Bengaluru (VTU)
CGPA: 9.21/10 | 2027 Passout

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/your-linkedin-id)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/rdeekshitha184)

---

## Acknowledgements

- Dataset: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- SHAP library: [shap.readthedocs.io](https://shap.readthedocs.io)
- Deployed on [Render](https://render.com)
