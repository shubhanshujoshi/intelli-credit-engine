import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
import os
import re
import requests
from datetime import datetime, timedelta
from num2words import num2words
from google import genai

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(layout="wide")
st.title("IntelliCredit-X | AI Credit Decision Engine")

# ----------------------------------------------------------
# LOAD API KEYS
# ----------------------------------------------------------

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    client = None

# ----------------------------------------------------------
# NEWS SENTIMENT
# ----------------------------------------------------------

def get_news_sentiment_score(company_name, combined_news_text):

    if not client or not combined_news_text.strip():
        return 0.0

    prompt = f"""
    You are an expert corporate credit risk analyst.

    Based on the following recent news about '{company_name}',
    return a sentiment score between -1.0 and +1.0.

    Negative = high credit risk
    Positive = financially strong

    Return ONLY the number.

    News:
    {combined_news_text}
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
# DUE DILIGENCE ANALYSIS
# ----------------------------------------------------------

def analyze_due_diligence(notes):

    if not notes.strip() or client is None:
        return 0.0

    prompt = f"""
    You are a senior credit risk analyst.

    Analyze the following credit officer notes.

    Return a score between -0.3 and +0.3

    Positive = lowers risk
    Negative = increases risk

    Return ONLY the number.

    Notes:
    {notes}
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        number = re.findall(r'-?\d+\.?\d*', response.text)[0]
        return float(number)

    except:
        return 0.0


# ----------------------------------------------------------
# FETCH NEWS
# ----------------------------------------------------------

def fetch_last_60_days_news(company_name):

    if not GNEWS_API_KEY:
        return []

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=60)

    url = (
        f"https://gnews.io/api/v4/search?"
        f"q={company_name}&"
        f"from={start_date.strftime('%Y-%m-%d')}&"
        f"to={end_date.strftime('%Y-%m-%d')}&"
        f"lang=en&"
        f"max=10&"
        f"apikey={GNEWS_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get("articles", [])
    except:
        return []


# ----------------------------------------------------------
# CREDIT MEMO GENERATOR
# ----------------------------------------------------------

def generate_cam(company, revenue, ebitda, debt, interest_cov,
                 sector, mgmt, litigation, sentiment,
                 loan, rate, decision):

    if client is None:
        return "AI CAM generation unavailable."

    prompt = f"""
    Generate a professional Credit Appraisal Memo.

    Company: {company}

    Financials
    Revenue: {revenue}
    EBITDA: {ebitda}
    Debt: {debt}
    Interest Coverage: {interest_cov}

    Risk Indicators
    Sector Risk: {sector}
    Management Quality: {mgmt}
    Litigation Count: {litigation}
    News Sentiment Score: {sentiment}

    Recommendation
    Loan Amount: {loan}
    Interest Rate: {rate}
    Decision: {decision}

    Structure the memo using Five Cs of Credit:
    Character
    Capacity
    Capital
    Conditions
    Collateral
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except:
        return "Unable to generate CAM at the moment."


# ----------------------------------------------------------
# LOAD MODEL FILES
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
def load_explainer():
    model = joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))
    return shap.TreeExplainer(model)

model = load_model()
threshold = load_threshold()
feature_names = load_feature_names()
explainer = load_explainer()

# ----------------------------------------------------------
# DOCUMENT UPLOAD
# ----------------------------------------------------------

st.subheader("Upload Supporting Documents")

uploaded_files = st.file_uploader(
    "Upload Annual Report / GST / Bank Statement",
    type=["pdf", "xlsx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} documents uploaded successfully.")

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

# ----------------------------------------------------------
# RESEARCH AGENT INPUT
# ----------------------------------------------------------

st.subheader("External AI News Risk Analysis")
company_name = st.text_input("Company Name")

# ----------------------------------------------------------
# DUE DILIGENCE NOTES
# ----------------------------------------------------------

st.subheader("Primary Due Diligence Notes")

site_visit_notes = st.text_area(
    "Credit Officer Observations",
    placeholder="Example: Factory operating at 40% capacity due to demand slowdown."
)

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Analyze Credit Risk"):

    articles = []
    sentiment_score = 0.0

    with st.spinner("Fetching news and analyzing sentiment..."):

        if company_name.strip() != "":
            articles = fetch_last_60_days_news(company_name)

            combined_news = ""
            for article in articles:
                combined_news += article.get("title", "") + ". "
                combined_news += article.get("description", "") + ". "

            sentiment_score = get_news_sentiment_score(company_name, combined_news)

    due_diligence_score = analyze_due_diligence(site_visit_notes)

    # ------------------------------------------------------
    # NEWS DISPLAY
    # ------------------------------------------------------

    st.subheader("News Intelligence Layer")

    if len(articles) > 0:
        st.success(f"Articles Analyzed: {len(articles)}")

        for article in articles:
            title = article.get("title", "No Title")
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "")

            st.markdown(f"• **{source}** – [{title}]({url})")

    else:
        st.warning("No recent news articles found.")

    # ------------------------------------------------------
    # MODEL INPUT
    # ------------------------------------------------------

    input_data = np.array([[ 
        revenue, ebitda, debt, interest_cov,
        gst, litigation, sentiment_score, sector,
        mgmt, capacity
    ]])

    input_df = pd.DataFrame(input_data, columns=feature_names)

    prob = model.predict_proba(input_df)[0][1]

    prob = prob - (0.05 * sentiment_score)
    prob = prob - (0.03 * due_diligence_score)

    prob = max(0, min(prob, 1))

    prediction = int(prob > threshold)

    # ------------------------------------------------------
    # RISK OUTPUT
    # ------------------------------------------------------

    st.subheader("Risk Assessment")

    st.metric("Probability of Default", round(prob,3))
    st.metric("News Sentiment", round(sentiment_score,2))
    st.metric("Due Diligence Adjustment", round(due_diligence_score,2))

    if prediction == 1:
        st.error("🚫 HIGH RISK — Loan Rejected")
    else:
        st.success("✅ LOW RISK — Loan Approved")

    # ------------------------------------------------------
    # LOAN STRUCTURING
    # ------------------------------------------------------

    max_loan = ebitda * 3.5
    adjusted_loan = max_loan * (1 - prob)

    base_rate = 9
    risk_premium = prob * 6
    interest_rate = base_rate + risk_premium

    st.subheader("Recommended Loan Terms")

    formatted_loan = f"{adjusted_loan:,.0f}"
    st.success(f"Recommended Loan Amount: ₹ {formatted_loan}")

    try:
        if adjusted_loan < 1e12:
            loan_in_words = num2words(int(adjusted_loan), lang='en_IN').title()
            st.info(f"Rupees {loan_in_words} Only")
    except:
        pass

    st.write("Recommended Interest Rate:", round(interest_rate,2), "%")

    # ------------------------------------------------------
    # RISK DRIVERS
    # ------------------------------------------------------

    st.subheader("Top Risk Drivers")

    drivers = []

    if debt > revenue:
        drivers.append("High debt relative to revenue")

    if interest_cov < 2:
        drivers.append("Weak interest coverage")

    if litigation > 3:
        drivers.append("Multiple legal disputes")

    if sentiment_score < -0.3:
        drivers.append("Negative news sentiment")

    if gst > 0.3:
        drivers.append("GST mismatch risk")

    if len(drivers) == 0:
        drivers.append("Financial profile appears stable")

    for d in drivers:
        st.write("•", d)

    # ------------------------------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------------------------------

    st.subheader("Explainability")

    try:
        shap_values = explainer(input_df)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)

        st.pyplot(fig)

    except:
        st.warning("Explainability chart unavailable.")

    # ------------------------------------------------------
    # CREDIT MEMO
    # ------------------------------------------------------

    st.subheader("Credit Appraisal Memo")

    decision_text = "Approved" if prediction == 0 else "Rejected"

    cam_text = generate_cam(
        company_name,
        revenue,
        ebitda,
        debt,
        interest_cov,
        sector,
        mgmt,
        litigation,
        sentiment_score,
        adjusted_loan,
        interest_rate,
        decision_text
    )

    st.write(cam_text)
