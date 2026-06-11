import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, RocCurveDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────────
# TotalCharges has spaces as missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID — not a feature
df.drop(columns=["customerID"], inplace=True)

# Target: Churn Yes/No → 1/0
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Binary Yes/No columns → 0/1
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

# gender: Male=1, Female=0
df["gender"] = (df["gender"] == "Male").astype(int)

# MultipleLines: No phone service → No
df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# Internet service-dependent columns: No internet service → No
internet_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
for col in internet_cols:
    df[col] = df[col].replace("No internet service", "No")
    df[col] = (df[col] == "Yes").astype(int)

df["MultipleLines"] = (df["MultipleLines"] == "Yes").astype(int)

# Label encode remaining categoricals
cat_cols = ["InternetService", "Contract", "PaymentMethod"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ─────────────────────────────────────────────
# 4. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

feature_names = list(X.columns)
joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# ─────────────────────────────────────────────
# 6. SMOTE — fix class imbalance
# ─────────────────────────────────────────────
print(f"\nBefore SMOTE — Class distribution:\n{y_train.value_counts()}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE  — Class distribution:\n{pd.Series(y_train_res).value_counts()}")

# ─────────────────────────────────────────────
# 7. MODEL TRAINING — XGBoost
# ─────────────────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ─────────────────────────────────────────────
# 8. THRESHOLD TUNING (business cost logic)
#    False Negative (missing a churner) costs more
#    than False Positive (retaining a loyal customer)
#    We optimise for F1 on the precision-recall curve
# ─────────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\nBest classification threshold (F1-optimised): {best_threshold:.3f}")
joblib.dump(best_threshold, f"{MODEL_DIR}/threshold.pkl")

y_pred = (y_proba >= best_threshold).astype(int)

# ─────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────
print("\n── Evaluation Results ──────────────────")
print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 10. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
print("\nComputing SHAP values (this may take ~30 seconds)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  feature_names=feature_names, show=False)
plt.title("Global Feature Importance (SHAP)", fontsize=14)
plt.tight_layout()
plt.savefig("static/shap_global.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static/shap_global.png")

# SHAP beeswarm plot (shows direction of impact too)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14)
plt.tight_layout()
plt.savefig("static/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static/shap_beeswarm.png")

# Save explainer for per-prediction explanations
joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")

# ─────────────────────────────────────────────
# 11. SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(model, f"{MODEL_DIR}/xgb_model.pkl")
print(f"\nModel saved to {MODEL_DIR}/xgb_model.pkl")
print("\nTraining complete!")
