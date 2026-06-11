import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)


API_URL = os.getenv("API_URL", "https://churn-ml-0kud.onrender.com")

st.title("📉 Customer Churn Predictor")
st.markdown(
    "Enter customer details below. The model will predict their churn probability "
    "and explain **why** using SHAP feature attribution."
)
st.divider()

# ─────────────────────────────────────────────
# Input form — two columns
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

with col3:
    st.subheader("Billing")
    contract = st.selectbox("Contract Type", [
        "Month-to-month", "One year", "Two year"
    ])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges (Rs)", 0.0, 200.0, 70.0, step=0.5)
    total_charges = st.number_input("Total Charges (Rs)", 0.0, 10000.0,
                                    float(monthly_charges * tenure), step=1.0)

st.divider()

# ─────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────
if st.button("Predict Churn Risk", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure api.py is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Results ──────────────────────────────
    prob = result["churn_probability"]
    risk = result["risk_level"]
    prediction = result["prediction"]

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")

    with res_col2:
        color_map = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
        st.metric("Risk Level", f"{color_map.get(risk, '')} {risk}")

    with res_col3:
        st.metric("Prediction", prediction)

    # ── SHAP waterfall (top reasons) ─────────
    st.subheader("Why did the model predict this?")
    st.caption("SHAP values show each feature's contribution to the churn prediction.")

    reasons = result["top_reasons"]
    feat_labels = [r["feature"] for r in reasons]
    impacts = [r["impact"] for r in reasons]
    colors = ["#e05c5c" if v > 0 else "#4a9e78" for v in impacts]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(feat_labels, impacts, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (contribution to churn prediction)", fontsize=10)
    ax.set_title(f"Top contributing features — Churn prob: {prob*100:.1f}%", fontsize=11)
    ax.invert_yaxis()

    for bar, val in zip(bars, impacts):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
            color="#333333"
        )

    red_patch = mpatches.Patch(color="#e05c5c", label="Increases churn risk")
    green_patch = mpatches.Patch(color="#4a9e78", label="Decreases churn risk")
    ax.legend(handles=[red_patch, green_patch], fontsize=9, loc="lower right")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plain English explanation ─────────────
    st.subheader("Plain English Summary")
    lines = []
    for r in reasons[:3]:
        direction = "pushing towards churn" if r["impact"] > 0 else "reducing churn risk"
        lines.append(f"- **{r['feature']}** is {direction} (SHAP: {r['impact']:+.3f})")
    st.markdown("\n".join(lines))

    # ── Recommended action ───────────────────
    if risk == "High Risk":
        st.warning(
            "**Recommended action:** Assign a retention agent. "
            "Offer a discounted annual contract or a free service upgrade."
        )
    elif risk == "Medium Risk":
        st.info(
            "**Recommended action:** Send a personalised loyalty offer via email. "
            "Monitor for the next 30 days."
        )
    else:
        st.success("**No immediate action needed.** Customer appears stable.")

# ─────────────────────────────────────────────
# Global SHAP plots section
# ─────────────────────────────────────────────
st.divider()
st.subheader("Global Model Insights")
tab1, tab2 = st.tabs(["Feature Importance", "SHAP Beeswarm"])

with tab1:
    if os.path.exists("static/shap_global.png"):
        st.image("static/shap_global.png", use_column_width=True)
    else:
        st.info("Run train.py first to generate SHAP plots.")

with tab2:
    if os.path.exists("static/shap_beeswarm.png"):
        st.image("static/shap_beeswarm.png", use_column_width=True)
    else:
        st.info("Run train.py first to generate SHAP plots.")
