import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
import re
from datetime import datetime, timedelta

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="IntelliCredit-X",
    layout="wide"
)

st.title("🏦 IntelliCredit-X | AI Credit Risk Analyzer")

# ======================================================
# LOAD MODEL
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))

@st.cache_resource
def load_features():
    return joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

@st.cache_resource
def load_threshold():
    with open(os.path.join(BASE_DIR, "config.json")) as f:
        return json.load(f)["best_threshold"]

model = load_model()
feature_names = load_features()
threshold = load_threshold()

# ======================================================
# SECTOR CONFIG
# ======================================================

SECTOR_CONFIG = {
    "Manufacturing": {"risk": 0.5},
    "IT Services": {"risk": 0.3},
    "Real Estate": {"risk": 0.6},
    "Retail": {"risk": 0.5},
    "Energy": {"risk": 0.6},
    "NBFC": {"risk": 0.7},
}

# ======================================================
# FINANCIAL RATIOS
# ======================================================

def calculate_financial_ratios(revenue, ebitda, debt, equity, finance_cost):

    ratios = {}

    ratios["EBITDA Margin"] = ebitda / revenue if revenue else 0
    ratios["Debt/Equity"] = debt / equity if equity else 0
    ratios["Interest Coverage"] = ebitda / finance_cost if finance_cost else 0
    ratios["Debt/EBITDA"] = debt / ebitda if ebitda else 0

    return ratios

# ======================================================
# EARLY WARNING SYSTEM
# ======================================================

def detect_early_warnings(interest_coverage, debt_to_equity, gst_variance, sentiment):

    warnings = []

    if interest_coverage < 1:
        warnings.append("⚠ Interest coverage below safe level")

    if debt_to_equity > 3:
        warnings.append("⚠ High leverage detected")

    if gst_variance > 0.2:
        warnings.append("⚠ Possible GST / revenue mismatch")

    if sentiment < -0.4:
        warnings.append("⚠ Negative market sentiment")

    return warnings

# ======================================================
# CREDIT SCORE
# ======================================================

def generate_credit_score(pd_probability):

    score = int(900 - (pd_probability * 600))
    return max(300, min(score, 900))

# ======================================================
# INDUSTRY BENCHMARKS
# ======================================================

INDUSTRY_BENCHMARKS = {
    "IT Services": {"Debt/Equity": 0.5, "Interest Coverage": 5},
    "Manufacturing": {"Debt/Equity": 1.5, "Interest Coverage": 3},
    "Real Estate": {"Debt/Equity": 2.5, "Interest Coverage": 2},
}

def compare_to_industry(sector, debt_to_equity, interest_coverage):

    benchmark = INDUSTRY_BENCHMARKS.get(sector)

    if not benchmark:
        return []

    insights = []

    if debt_to_equity > benchmark["Debt/Equity"]:
        insights.append("⚠ Leverage above industry average")

    if interest_coverage < benchmark["Interest Coverage"]:
        insights.append("⚠ Weak interest coverage vs industry")

    return insights

# ======================================================
# SIMPLE NEWS SENTIMENT
# ======================================================

def get_news_sentiment(company):

    try:

        api = st.secrets["GNEWS_API_KEY"]

        url = f"https://gnews.io/api/v4/search?q={company}&max=5&apikey={api}"

        r = requests.get(url).json()

        articles = r.get("articles", [])

        text = " ".join([a["title"] for a in articles])

        negative_words = ["fraud", "scam", "default", "loss"]

        sentiment = 0

        for word in negative_words:
            if word in text.lower():
                sentiment -= 0.2

        return sentiment, articles

    except:
        return 0, []

# ======================================================
# INPUT SECTION
# ======================================================

st.header("Financial Inputs")

col1, col2 = st.columns(2)

with col1:

    revenue = st.number_input("Revenue", value=500000000.0)
    pbt = st.number_input("Profit Before Tax", value=50000000.0)
    finance_cost = st.number_input("Finance Cost", value=10000000.0)
    depreciation = st.number_input("Depreciation", value=10000000.0)

with col2:

    debt = st.number_input("Total Debt", value=200000000.0)
    equity = st.number_input("Equity", value=200000000.0)

sector = st.selectbox("Sector", list(SECTOR_CONFIG.keys()))

gst_variance = st.slider("GST Variance", 0.0, 0.5, 0.1)

company = st.text_input("Company Name")

# ======================================================
# CALCULATIONS
# ======================================================

ebitda = pbt + finance_cost + depreciation

interest_coverage = ebitda / finance_cost if finance_cost else 0
debt_to_equity = debt / equity if equity else 0

ratios = calculate_financial_ratios(
    revenue, ebitda, debt, equity, finance_cost
)

# ======================================================
# DISPLAY RATIOS
# ======================================================

st.subheader("Financial Ratios")

cols = st.columns(len(ratios))

for i, (k, v) in enumerate(ratios.items()):
    cols[i].metric(k, f"{v:.2f}")

# ======================================================
# ANALYSIS
# ======================================================

if st.button("Run Credit Analysis"):

    sentiment, articles = get_news_sentiment(company)

    input_data = np.array([[

        revenue,
        ebitda,
        debt,
        interest_coverage,
        gst_variance,
        sentiment,
        SECTOR_CONFIG[sector]["risk"]

    ]])

    df = pd.DataFrame(input_data, columns=feature_names)

    df = df.reindex(columns=feature_names, fill_value=0)

    pd_probability = model.predict_proba(df)[0][1]

    decision = "Approved" if pd_probability <= threshold else "Rejected"

    credit_score = generate_credit_score(pd_probability)

    st.header("Decision")

    col1, col2, col3 = st.columns(3)

    col1.metric("Decision", decision)
    col2.metric("Default Probability", f"{pd_probability:.2%}")
    col3.metric("Credit Score", credit_score)

    # ==================================================
    # EARLY WARNINGS
    # ==================================================

    warnings = detect_early_warnings(
        interest_coverage,
        debt_to_equity,
        gst_variance,
        sentiment
    )

    if warnings:

        st.subheader("Early Warning Signals")

        for w in warnings:
            st.error(w)

    # ==================================================
    # INDUSTRY COMPARISON
    # ==================================================

    insights = compare_to_industry(
        sector,
        debt_to_equity,
        interest_coverage
    )

    if insights:

        st.subheader("Industry Benchmark Comparison")

        for i in insights:
            st.warning(i)

    # ==================================================
    # NEWS
    # ==================================================

    if articles:

        st.subheader("Recent News")

        for a in articles:
            st.markdown(f"[{a['title']}]({a['url']})")

st.caption("IntelliCredit-X | AI Credit Decision Engine")
