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
import pdfplumber
from datetime import datetime, timedelta
from num2words import num2words
from google import genai

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(layout="wide")
st.title("IntelliCredit-X | AI Credit Decision Engine")

# -----------------------------------------------------
# LOAD API KEYS
# -----------------------------------------------------

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

client = None
if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except:
        client = None

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(BASE_DIR,"financial_model.pkl"))
    except:
        return None

@st.cache_resource
def load_threshold():
    try:
        with open(os.path.join(BASE_DIR,"config.json")) as f:
            return json.load(f)["best_threshold"]
    except:
        return 0.5

@st.cache_resource
def load_features():
    try:
        return joblib.load(os.path.join(BASE_DIR,"feature_names.pkl"))
    except:
        return None

@st.cache_resource
def load_explainer():
    try:
        model = joblib.load(os.path.join(BASE_DIR,"financial_model.pkl"))
        return shap.TreeExplainer(model)
    except:
        return None

model = load_model()
threshold = load_threshold()
feature_names = load_features()
explainer = load_explainer()

# -----------------------------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------------------------

def extract_pdf_text(uploaded_file):

    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except:
        return ""

    return text

# -----------------------------------------------------
# AI FINANCIAL EXTRACTION
# -----------------------------------------------------

def ai_extract_financials(text):

    if not client or text == "":
        return {}

    prompt = f"""
Extract financial values from this financial statement.

Return JSON with:

revenue
profit_before_tax
finance_cost
depreciation
total_debt

If value not found return null.

Text:
{text[:10000]}
"""

    try:

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        cleaned = response.text.replace("```json","").replace("```","")
        data = json.loads(cleaned)

        return data

    except:
        return {}

# -----------------------------------------------------
# NEWS SENTIMENT
# -----------------------------------------------------

def get_news_sentiment(company):

    if not company or not GNEWS_API_KEY:
        return 0, []

    end = datetime.utcnow()
    start = end - timedelta(days=60)

    url = (
        f"https://gnews.io/api/v4/search?"
        f"q=\"{company}\" AND (business OR company OR earnings OR finance)"
        f"&from={start.strftime('%Y-%m-%d')}"
        f"&to={end.strftime('%Y-%m-%d')}"
        f"&lang=en"
        f"&max=5"
        f"&apikey={GNEWS_API_KEY}"
    )

    try:
        response = requests.get(url,timeout=10)
        articles = response.json().get("articles",[])
    except:
        articles = []

    combined = ""
    for a in articles:
        combined += a.get("title","") + ". "

    if not client or combined == "":
        return 0, articles

    prompt = f"""
Return sentiment score between -1 and 1.

News:
{combined}
"""

    try:

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        number = re.findall(r'-?\d+\.?\d*',resp.text)[0]
        return float(number), articles

    except:
        return 0, articles

# -----------------------------------------------------
# CAM GENERATOR
# -----------------------------------------------------

def generate_cam(company,revenue,ebitda,debt,decision):

    if not client:
        return f"""
Credit Appraisal Memo

Company: {company}

Revenue: {revenue}
EBITDA: {ebitda}
Debt: {debt}

Decision: {decision}
"""

    prompt = f"""
Create a professional Credit Appraisal Memo.

Company: {company}

Revenue: {revenue}
EBITDA: {ebitda}
Debt: {debt}

Decision: {decision}
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text

    except:
        return "CAM generation unavailable."

# -----------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------

st.header("Upload Financial Statement")

uploaded_pdf = st.file_uploader(
    "Upload Balance Sheet / P&L",
    type=["pdf"]
)

ai_data = {}

if uploaded_pdf:

    st.success("PDF uploaded")

    text = extract_pdf_text(uploaded_pdf)

    ai_data = ai_extract_financials(text)

    if ai_data:
        st.info("AI extracted financial values. Please verify.")

# -----------------------------------------------------
# EDITABLE FINANCIAL FIELDS
# -----------------------------------------------------

st.header("Financial Inputs")

revenue = st.number_input("Revenue",
value=float(ai_data.get("revenue",500000000)))

pbt = st.number_input("Profit Before Tax",
value=float(ai_data.get("profit_before_tax",50000000)))

finance_cost = st.number_input("Finance Cost",
value=float(ai_data.get("finance_cost",10000000)))

depreciation = st.number_input("Depreciation",
value=float(ai_data.get("depreciation",10000000)))

debt = st.number_input("Total Debt",
value=float(ai_data.get("total_debt",200000000)))

# -----------------------------------------------------
# AUTOMATIC CALCULATIONS
# -----------------------------------------------------

ebitda = pbt + finance_cost + depreciation

interest_coverage = 0
if finance_cost > 0:
    interest_coverage = ebitda / finance_cost

st.subheader("Calculated Ratios")

st.write("EBITDA:",round(ebitda,2))
st.write("Interest Coverage:",round(interest_coverage,2))

# -----------------------------------------------------
# ADDITIONAL INPUTS
# -----------------------------------------------------

gst = st.slider("GST Mismatch",0.0,0.5,0.1)
litigation = st.number_input("Litigation Count",value=1)
sector = st.slider("Sector Risk",0.0,1.0,0.3)
mgmt = st.slider("Management Quality",1.0,10.0,7.0)
capacity = st.slider("Capacity Utilization",0.0,1.0,0.8)

company = st.text_input("Company Name")

articles = []

# -----------------------------------------------------
# RUN MODEL
# -----------------------------------------------------

if st.button("Analyze Credit Risk"):

    sentiment, articles = get_news_sentiment(company)

    st.subheader("News Intelligence")

    if articles:

        for article in articles:

            title = article.get("title","")
            url = article.get("url","")
            source = article.get("source",{}).get("name","Unknown")
            description = article.get("description","")

            st.markdown(f"**{source}** — [{title}]({url})")

            if description:
                st.caption(description)

            st.write("---")

    else:
        st.warning("No relevant business news found.")

    if model and feature_names:

        input_data = np.array([[

            revenue,
            ebitda,
            debt,
            interest_coverage,
            gst,
            litigation,
            sentiment,
            sector,
            mgmt,
            capacity

        ]])

        df = pd.DataFrame(input_data,columns=feature_names)

        prob = model.predict_proba(df)[0][1]

    else:
        prob = 0.3

    prob = max(0,min(prob,1))

    st.subheader("Risk Assessment")

    st.metric("Probability of Default",round(prob,3))

    decision = "Approved"
    if prob > threshold:
        decision = "Rejected"

    if decision == "Rejected":
        st.error("Loan Rejected")
    else:
        st.success("Loan Approved")

# -----------------------------------------------------
# LOAN RECOMMENDATION
# -----------------------------------------------------

    max_loan = ebitda * 3.5
    loan = max_loan * (1-prob)

    rate = 9 + prob*6

    st.subheader("Loan Recommendation")

    st.write("Loan Amount:",round(loan,2))
    st.write("Interest Rate:",round(rate,2),"%")

    try:
        words = num2words(int(loan),lang="en_IN")
        st.info(words)
    except:
        pass

# -----------------------------------------------------
# SHAP
# -----------------------------------------------------

    if explainer:

        try:
            shap_values = explainer(df)

            fig,ax = plt.subplots()
            shap.plots.waterfall(shap_values[0],show=False)

            st.pyplot(fig)

        except:
            st.warning("Explainability unavailable.")

# -----------------------------------------------------
# CAM
# -----------------------------------------------------

    st.subheader("Credit Appraisal Memo")

    cam = generate_cam(company,revenue,ebitda,debt,decision)

    st.write(cam)
