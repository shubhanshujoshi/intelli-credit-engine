import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
import os
from num2words import num2words
from google import genai
import re

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(layout="wide")
st.title("IntelliCredit-X | AI Credit Decision Engine")

# ----------------------------------------------------------
# LOAD API KEY (STREAMLIT CLOUD SAFE)
# ----------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    client = None

# ----------------------------------------------------------
# GEMINI SENTIMENT FUNCTION
# ----------------------------------------------------------

def get_news_sentiment_score(company_name, news_snippet):

    if not client or not news_snippet.strip():
        return 0.0

    prompt = f"""
    You are an expert corporate credit risk analyst.

    Analyze the following news about '{company_name}'.

    Return a number between -1.0 and +1.0.
    Negative = high credit risk
    Positive = financially strong
    Only return the number.

    News:
    "{news_snippet}"
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        score_text = response.text.strip()
        number = re.findall(r'-?\d+\.?\d*', score_text)[0]
        return float(number)

    except:
        return 0.0


# ----------------------------------------------------------
# SAFE BASE DIRECTORY
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))

@st.cache_resource
def load_threshold():
    with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
        return json.load(f)["best_threshold"]

@st.cache_resource
def load_feature_names():
    return joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

@st.cache_resource
def load_metrics():
    with open(os.path.join(BASE_DIR, "metrics.json"), "r") as f:
        return json.load(f)

@st.cache_resource
def load_explainer():
    model = joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))
    return shap.TreeExplainer(model)

model = load_model()
threshold = load_threshold()
feature_names = load_feature_names()
metrics = load_metrics()
explainer = load_explainer()

# ----------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------

st.header("Enter Company Financial Details")

col1, col2 = st.columns(2)

with col1:
    revenue = st.number_input("Revenue", value=5e8)
    ebitda = st.number_input("EBITDA", value=1e8)
    debt = st.number_input("Debt", value=2e8)
    interest_cov = st.number_input("Interest Coverage", value=2.5)
    gst = st.slider("GST Mismatch", 0.0, 0.5, 0.1)

with col2:
    litigation = st.number_input("Litigation Count", value=1)
    sector = st.slider("Sector Risk", 0.0, 1.0, 0.3)
    mgmt = st.slider("Management Quality", 1.0, 10.0, 7.0)
    capacity = st.slider("Capacity Utilization", 0.0, 1.0, 0.8)

st.subheader("External News Risk Analysis")

company_name = st.text_input("Company Name")
news_snippet = st.text_area("Paste Recent News About Company")

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Analyze Credit Risk"):

    sentiment_score = get_news_sentiment_score(company_name, news_snippet)

    input_data = np.array([[ 
        revenue, ebitda, debt, interest_cov,
        gst, litigation, sentiment_score, sector,
        mgmt, capacity
    ]])

    input_df = pd.DataFrame(input_data, columns=feature_names)

    prob = model.predict_proba(input_df)[0][1]

    # Optional: Blend sentiment impact slightly
    prob = prob + (0.05 * sentiment_score)
    prob = max(0, min(prob, 1))  # keep within 0-1

    prediction = int(prob > threshold)

    st.subheader("Risk Assessment")

    st.metric("Probability of Default (PD)", round(prob, 3))
    st.metric("News Sentiment Score", round(sentiment_score, 2))
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

    st.subheader("Recommended Terms")

    formatted_loan = f"{adjusted_loan:,.0f}"
    loan_in_words = num2words(int(adjusted_loan), lang='en_IN').title()

    st.success(f"Recommended Loan Amount: ₹ {formatted_loan}")
    st.info(f"Rupees {loan_in_words} Only")
    st.write("Recommended Interest Rate:", round(interest_rate, 2), "%")

    # ------------------------------------------------------
    # SHAP EXPLANATION
    # ------------------------------------------------------

    st.subheader("Explainability (Why this decision?)")

    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
