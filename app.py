import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import requests
import pdfplumber
from datetime import datetime, timedelta
from num2words import num2words
from google import genai
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="IntelliCredit-X")
st.title("🏦 IntelliCredit-X")
st.markdown("AI Credit Decision Engine")

# API KEYS
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

# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))
    except:
        return None

@st.cache_resource
def load_threshold():
    try:
        with open(os.path.join(BASE_DIR, "config.json")) as f:
            return json.load(f)["best_threshold"]
    except:
        return 0.5

@st.cache_resource
def load_features():
    try:
        return joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    except:
        return None

model = load_model()
threshold = load_threshold()
feature_names = load_features()

# PDF EXTRACTION
def extract_pdf_text(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    text += f"--- PAGE {page_num + 1} ---\n{t}\n"
    except:
        return ""
    return text

# GST-ITR RECONCILIATION
def gst_itr_reconciliation(gst_turnover, itr_revenue, gst_2a, gst_3b):
    reconciliation = {
        "status": "PASS",
        "risk_level": "LOW",
        "findings": [],
        "risk_score_impact": 0.0,
        "red_flags": []
    }
    
    if gst_turnover and itr_revenue:
        mismatch_pct = abs(gst_turnover - itr_revenue) / itr_revenue
        
        if mismatch_pct > 0.20:
            reconciliation["risk_level"] = "CRITICAL"
            reconciliation["status"] = "FAIL"
            reconciliation["red_flags"].append(
                f"🚨 CRITICAL: GST-ITR mismatch {mismatch_pct:.1%}"
            )
            reconciliation["risk_score_impact"] = 0.15
        elif mismatch_pct > 0.10:
            reconciliation["risk_level"] = "HIGH"
            reconciliation["red_flags"].append(
                f"⚠️ HIGH: GST vs ITR variance {mismatch_pct:.1%}"
            )
            reconciliation["risk_score_impact"] = 0.08
        elif mismatch_pct > 0.05:
            reconciliation["risk_level"] = "MEDIUM"
            reconciliation["findings"].append(
                f"Minor discrepancy {mismatch_pct:.1%}"
            )
            reconciliation["risk_score_impact"] = 0.03
        else:
            reconciliation["findings"].append("✓ GST-ITR aligned")
    
    if gst_2a and gst_3b:
        iTC_discrepancy = abs(gst_2a - gst_3b) / gst_2a if gst_2a > 0 else 0
        if iTC_discrepancy > 0.30:
            reconciliation["red_flags"].append(
                f"🚨 ITC fraud risk: {iTC_discrepancy:.1%}"
            )
            reconciliation["risk_score_impact"] += 0.10
    
    return reconciliation

# AI EXTRACTION
def ai_extract_financials(text):
    if not client or text == "":
        return {}
    
    prompt = f"""Extract: revenue, profit_before_tax, finance_cost, depreciation, total_debt
Return ONLY JSON like: {{"revenue": 500000000}}
Text: {text[:3000]}"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        if cleaned.startswith('{'):
            return json.loads(cleaned)
        return {}
    except:
        return {}

# NEWS SENTIMENT
def get_news_sentiment(company):
    if not company or not GNEWS_API_KEY:
        return 0, []
    
    try:
        url = (
            f"https://gnews.io/api/v4/search?"
            f"q=\"{company}\" AND business"
            f"&lang=en&max=3"
            f"&apikey={GNEWS_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
        return 0, articles
    except:
        return 0, []

# CAM GENERATOR
def generate_cam(company, revenue, ebitda, debt, decision):
    if not client:
        return f"CAM: {company} - {decision}"
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Credit memo for {company}: Revenue ₹{revenue:,.0f}, EBITDA ₹{ebitda:,.0f}, Debt ₹{debt:,.0f}. Decision: {decision}",
            generation_config={"max_output_tokens": 1000}
        )
        return response.text
    except:
        return f"CAM: {company} - {decision}"

# MAIN UI
st.header("📄 Upload Financial Statement")
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

ai_data = {}
if uploaded_pdf:
    st.success("✅ PDF uploaded")
    with st.spinner("Extracting..."):
        text = extract_pdf_text(uploaded_pdf)
        ai_data = ai_extract_financials(text)
    if ai_data:
        st.info("✓ Data extracted")

st.header("💰 Financial Inputs")

col1, col2 = st.columns(2)

with col1:
    revenue = st.number_input("Revenue (₹)", min_value=0.0, 
                             value=float(ai_data.get("revenue", 500000000)))
    pbt = st.number_input("Profit Before Tax (₹)", min_value=0.0,
                         value=float(ai_data.get("profit_before_tax", 50000000)))
    finance_cost = st.number_input("Finance Cost (₹)", min_value=0.0,
                                  value=float(ai_data.get("finance_cost", 10000000)))
    depreciation = st.number_input("Depreciation (₹)", min_value=0.0,
                                  value=float(ai_data.get("depreciation", 10000000)))

with col2:
    debt = st.number_input("Total Debt (₹)", min_value=0.0,
                          value=float(ai_data.get("total_debt", 200000000)))
    current_assets = st.number_input("Current Assets (₹)", min_value=0.0, value=200000000.0)
    current_liabilities = st.number_input("Current Liabilities (₹)", min_value=0.0, value=150000000.0)
    equity = st.number_input("Total Equity (₹)", min_value=0.0, value=300000000.0)

# RATIOS
ebitda = pbt + finance_cost + depreciation
interest_coverage = ebitda / finance_cost if finance_cost > 0 else 0
current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
debt_equity = debt / equity if equity > 0 else 0
profit_margin = pbt / revenue if revenue > 0 else 0
dscr = ebitda / (finance_cost * 1.1) if finance_cost > 0 else 0

st.subheader("📈 Calculated Ratios")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("EBITDA", f"₹{ebitda:,.0f}")
    st.metric("Interest Coverage", f"{interest_coverage:.2f}x")
with c2:
    st.metric("Current Ratio", f"{current_ratio:.2f}x")
    st.metric("D/E Ratio", f"{debt_equity:.2f}x")
with c3:
    st.metric("Profit Margin", f"{profit_margin:.1%}")
    st.metric("DSCR", f"{dscr:.2f}x")

# GST-ITR
st.header("📊 GST-ITR Reconciliation")
col1, col2 = st.columns(2)

with col1:
    gst_turnover = st.number_input("GST Turnover (₹)", min_value=0.0, value=revenue * 0.95)
    gst_2a = st.number_input("GSTR-2A ITC (₹)", min_value=0.0, value=5000000.0)

with col2:
    itr_revenue = st.number_input("ITR Revenue (₹)", min_value=0.0, value=revenue)
    gst_3b = st.number_input("GSTR-3B Claimed (₹)", min_value=0.0, value=4500000.0)

# RISK FACTORS
st.header("⚠️ Risk Factors")
col1, col2, col3 = st.columns(3)

with col1:
    company = st.text_input("Company Name")
    sector = st.selectbox("Sector", ["AUTO", "NBFC", "PHARMA", "RETAIL", "OTHER"])
    cibil_score = st.slider("CIBIL Score", 1, 100, 75)

with col2:
    mgmt = st.slider("Management Quality (1-10)", 1, 10, 7)
    capacity = st.slider("Capacity Utilization (%)", 0, 100, 80)
    litigation = st.number_input("Litigation Cases", min_value=0, value=0)

with col3:
    gst_risk = st.slider("GST Risk (0-0.5)", 0.0, 0.5, 0.1)
    sector_sentiment = st.slider("Sector Sentiment (-1 to +1)", -1.0, 1.0, 0.0)

# ANALYSIS
if st.button("🚀 ANALYZE CREDIT RISK", use_container_width=True):
    
    if not company:
        st.error("⚠️ Enter Company Name")
    else:
        st.subheader("📊 Analysis Results")
        
        # GST-ITR Check
        st.subheader("🔍 GST-ITR Reconciliation")
        gst_recon = gst_itr_reconciliation(gst_turnover, itr_revenue, gst_2a, gst_3b)
        
        if gst_recon["red_flags"]:
            for flag in gst_recon["red_flags"]:
                st.error(flag)
        else:
            for finding in gst_recon["findings"]:
                st.success(finding)
        
        # Litigation
        st.subheader("⚖️ Litigation Assessment")
        if litigation > 0:
            st.warning(f"⚠️ {litigation} case(s) - Risk: +{litigation * 0.05:.1%}")
        else:
            st.success("✓ No litigation")
        
        # CIBIL
        st.subheader("💳 CIBIL Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CIBIL Score", cibil_score)
        with col2:
            rating = "EXCELLENT" if cibil_score > 90 else "GOOD" if cibil_score > 70 else "FAIR"
            st.metric("Rating", rating)
        
        # News
        st.subheader("📰 News Analysis")
        sentiment, articles = get_news_sentiment(company)
        if articles:
            st.metric("Articles Found", len(articles))
            for article in articles[:2]:
                st.markdown(f"[{article.get('title', 'News')}]({article.get('url')})")
        else:
            st.info("No news found")
        
        # Prediction
        st.subheader("🤖 Risk Assessment")
        
        if model and feature_names:
            try:
                input_data = np.array([[
                    revenue, ebitda, debt, interest_coverage,
                    gst_risk, float(litigation), sentiment,
                    sector_sentiment, mgmt / 10.0, capacity / 100.0
                ]])
                df = pd.DataFrame(input_data, columns=feature_names)
                prob = model.predict_proba(df)[0][1]
            except:
                prob = (100 - cibil_score) / 100.0
        else:
            prob = (100 - cibil_score) / 100.0
        
        prob = min(1.0, max(0.0, prob + gst_recon["risk_score_impact"]))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probability of Default", f"{prob:.1%}")
        
        decision = "✅ APPROVED" if prob <= threshold else "❌ REJECTED"
        
        with col2:
            if "APPROVED" in decision:
                st.success(decision)
            else:
                st.error(decision)
        
        with col3:
            risk_level = "LOW" if prob < 0.2 else "MEDIUM" if prob < 0.4 else "HIGH" if prob < 0.6 else "CRITICAL"
            st.metric("Risk Level", risk_level)
        
        # Loan
        st.subheader("💵 Loan Recommendation")
        
        max_loan = ebitda * 3.5
        recommended_loan = max_loan * (1 - prob)
        interest_rate = 9 + prob * 6
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Loan (3.5x)", f"₹{max_loan:,.0f}")
        
        with col2:
            st.metric("Recommended", f"₹{recommended_loan:,.0f}")
        
        with col3:
            st.metric("Interest Rate", f"{interest_rate:.2f}%")
        
        try:
            loan_words = num2words(int(recommended_loan), lang="en_IN")
            st.info(f"💬 {loan_words}")
        except:
            pass
        
        # Explainability
        st.subheader("💡 Decision Explanation")
        
        explanation = f"""
📊 DECISION: {decision}
Probability of Default: {prob:.1%}

KEY FACTORS:
"""
        
        if gst_recon["red_flags"]:
            explanation += "\n🔴 GST-ITR Issues:\n"
            for flag in gst_recon["red_flags"][:2]:
                explanation += f"  • {flag}\n"
        
        if interest_coverage > 2.5:
            explanation += f"\n✅ Strong interest coverage: {interest_coverage:.2f}x\n"
        
        if cibil_score > 70:
            explanation += f"✅ Good CIBIL score: {cibil_score}\n"
        
        st.text(explanation)
        
        # CAM
        st.subheader("📋 Credit Appraisal Memo")
        cam = generate_cam(company, revenue, ebitda, debt, decision)
        st.text_area("CAM", cam, height=300)
        
        st.download_button(
            label="📥 Download CAM",
            data=cam,
            file_name=f"CAM_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown("**IntelliCredit-X** | AI Credit Decision Engine")
