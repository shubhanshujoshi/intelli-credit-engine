import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 IntelliCredit-X | AI Credit Decision Engine")

# ----------------------------------------------------------
# LOAD MODEL FILES
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("financial_model.pkl")
    return model

@st.cache_resource
def load_threshold():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config["best_threshold"]

@st.cache_resource
def load_feature_names():
    return joblib.load("feature_names.pkl")

@st.cache_resource
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)

model = load_model()
threshold = load_threshold()
feature_names = load_feature_names()
metrics = load_metrics()

# SHAP Explainer (cached)
@st.cache_resource
def load_explainer(model):
    return shap.Explainer(model)

explainer = load_explainer(model)

# ----------------------------------------------------------
# SHOW MODEL PERFORMANCE
# ----------------------------------------------------------

st.sidebar.header("📊 Model Performance")
st.sidebar.write(metrics)

# ----------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------

st.header("📥 Enter Company Financial Details")

col1, col2 = st.columns(2)

with col1:
    revenue = st.number_input("Revenue", value=5e8)
    ebitda = st.number_input("EBITDA", value=1e8)
    debt = st.number_input("Debt", value=2e8)
    interest_cov = st.number_input("Interest Coverage", value=2.5)
    gst = st.slider("GST Mismatch", 0.0, 0.5, 0.1)

with col2:
    litigation = st.number_input("Litigation Count", value=1)
    sentiment = st.slider("Negative Sentiment", 0.0, 1.0, 0.2)
    sector = st.slider("Sector Risk", 0.0, 1.0, 0.3)
    mgmt = st.slider("Management Quality", 1.0, 10.0, 7.0)
    capacity = st.slider("Capacity Utilization", 0.0, 1.0, 0.8)

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Analyze Credit Risk"):

    input_data = np.array([[
        revenue, ebitda, debt, interest_cov,
        gst, litigation, sentiment, sector,
        mgmt, capacity
    ]])

    input_df = pd.DataFrame(input_data, columns=feature_names)

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob > threshold)

    st.subheader("📊 Risk Assessment")

    st.write("Probability of Default (PD):", round(prob, 3))
    st.write("Decision Threshold:", round(threshold, 3))

    if prediction == 1:
        st.error("🚫 HIGH RISK — Loan Rejected")
    else:
        st.success("✅ LOW RISK — Loan Approved")

    # ------------------------------------------------------
    # LOAN PRICING LOGIC
    # ------------------------------------------------------

    max_loan = ebitda * 3.5
    adjusted_loan = max_loan * (1 - prob)

    base_rate = 9
    risk_premium = prob * 6
    interest_rate = base_rate + risk_premium

    st.subheader("💰 Recommended Terms")

    st.write("Recommended Loan Amount: ₹", round(adjusted_loan, 2))
    st.write("Recommended Interest Rate:", round(interest_rate, 2), "%")

    # ------------------------------------------------------
    # SHAP EXPLANATION
    # ------------------------------------------------------

    st.subheader("🔎 Explainability (Why this decision?)")

    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
