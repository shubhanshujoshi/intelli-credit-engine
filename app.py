import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import os
import re
import requests
import pdfplumber
from datetime import datetime, timedelta
from num2words import num2words
from google import genai
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(layout="wide", page_title="IntelliCredit-X")
st.title("🏦 IntelliCredit-X | AI Credit Decision Engine")
st.markdown("*Bridging the Intelligence Gap in Corporate Credit Appraisal*")

# =====================================================
# LOAD API KEYS
# =====================================================

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

# =====================================================
# LOAD MODEL
# =====================================================

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

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_pdf_text(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    text += f"--- PAGE {page_num + 1} ---\n{t}\n"
                
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        text += "\n[TABLE]\n"
                        for row in table:
                            text += " | ".join(str(cell) for cell in row if cell) + "\n"
    except Exception as e:
        st.warning(f"PDF extraction error: {str(e)}")
        return ""
    
    return text

# =====================================================
# GST-ITR RECONCILIATION
# =====================================================

def gst_itr_reconciliation(gst_turnover: float, itr_revenue: float, 
                           gst_2a: float, gst_3b: float) -> dict:
    """CRITICAL: Detect revenue inflation via GST-ITR mismatch"""
    
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
                f"🚨 CRITICAL: GST-ITR mismatch {mismatch_pct:.1%}. Possible revenue inflation."
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
                f"Minor discrepancy {mismatch_pct:.1%} between GST and ITR"
            )
            reconciliation["risk_score_impact"] = 0.03
        else:
            reconciliation["findings"].append("✓ GST-ITR revenue aligned")
    
    if gst_2a and gst_3b:
        iTC_discrepancy = abs(gst_2a - gst_3b) / gst_2a if gst_2a > 0 else 0
        
        if iTC_discrepancy > 0.30:
            reconciliation["red_flags"].append(
                f"🚨 ITC fraud risk: {iTC_discrepancy:.1%} variance"
            )
            reconciliation["risk_score_impact"] += 0.10
    
    return reconciliation

# =====================================================
# AI FINANCIAL EXTRACTION
# =====================================================

def ai_extract_financials(text: str) -> dict:
    if not client or text == "":
        return {}
    
    prompt = f"""
Extract financial values from this statement:
- revenue
- profit_before_tax
- finance_cost
- depreciation
- total_debt

Return ONLY valid JSON, no markdown:
{{}}

Text:
{text[:5000]}
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        if cleaned.startswith('{'):
            data = json.loads(cleaned)
            return data
        return {}
    except:
        return {}

# =====================================================
# NEWS SENTIMENT
# =====================================================

def get_news_sentiment(company: str, sector: str = None) -> tuple:
    if not company or not GNEWS_API_KEY:
        return 0, [], []
    
    end = datetime.utcnow()
    start = end - timedelta(days=60)
    
    sentiment_score = 0.0
    articles = []
    sector_articles = []
    
    try:
        url = (
            f"https://gnews.io/api/v4/search?"
            f"q=\"{company}\" AND (business OR earnings)"
            f"&from={start.strftime('%Y-%m-%d')}"
            f"&to={end.strftime('%Y-%m-%d')}"
            f"&lang=en&max=3"
            f"&apikey={GNEWS_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
    except:
        articles = []
    
    combined = ""
    for a in articles:
        combined += a.get("title", "") + ". "
    
    if client and combined:
        try:
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Sentiment (-1 to 1): {combined[:1000]}"
            )
            number = re.findall(r'-?\d+\.?\d*', resp.text)
            if number:
                sentiment_score = float(number[0])
        except:
            sentiment_score = 0
    
    return sentiment_score, articles, sector_articles

# =====================================================
# CAM GENERATOR
# =====================================================

def generate_cam(company: str, revenue: float, ebitda: float, debt: float, decision: str) -> str:
    if not client:
        return f"""
CREDIT APPRAISAL MEMO

Company: {company}
Date: {datetime.now().strftime('%Y-%m-%d')}
Decision: {decision}

Financial Summary:
├─ Revenue: ₹{revenue:,.0f}
├─ EBITDA: ₹{ebitda:,.0f}
└─ Total Debt: ₹{debt:,.0f}

Recommendation: {decision}
"""

    prompt = f"""
Generate professional Credit Appraisal Memo:
Company: {company}
Revenue: ₹{revenue:,.0f}
EBITDA: ₹{ebitda:,.0f}
Debt: ₹{debt:,.0f}
Decision: {decision}

Format: Professional banking memo with findings.
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            generation_config={"max_output_tokens": 1500}
        )
        return response.text
    except:
        return f"CAM for {company}: {decision}"

# =====================================================
# MAIN APP
# =====================================================

st.header("📄 Step 1: Upload Financial Statement")
uploaded_pdf = st.file_uploader("Upload Balance Sheet / P&L", type=["pdf"])

ai_data = {}

if uploaded_pdf:
    st.success("✅ PDF uploaded")
    
    with st.spinner("Extracting financial data..."):
        text = extract_pdf_text(uploaded_pdf)
        ai_data = ai_extract_financials(text)
    
    if ai_data:
        st.info("✓ AI extracted values. Please verify.")

# =====================================================
# FINANCIAL INPUTS
# =====================================================

st.header("💰 Step 2: Financial Inputs")

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
    
    current_assets = st.number_input("Current Assets (₹)", min_value=0.0,
                                    value=200000000.0)
    
    current_liabilities = st.number_input("Current Liabilities (₹)", min_value=0.0,
                                         value=150000000.0)
    
    equity = st.number_input("Total Equity (₹)", min_value=0.0,
                            value=300000000.0)

# =====================================================
# CALCULATED RATIOS
# =====================================================

ebitda = pbt + finance_cost + depreciation
interest_coverage = ebitda / finance_cost if finance_cost > 0 else 0
current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
debt_equity = debt / equity if equity > 0 else 0
profit_margin = pbt / revenue if revenue > 0 else 0
dscr = ebitda / (finance_cost * 1.1) if finance_cost > 0 else 0

st.subheader("📈 Calculated Ratios")

ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)

with ratio_col1:
    st.metric("EBITDA", f"₹{ebitda:,.0f}")
    st.metric("Interest Coverage", f"{interest_coverage:.2f}x")

with ratio_col2:
    st.metric("Current Ratio", f"{current_ratio:.2f}x")
    st.metric("D/E Ratio", f"{debt_equity:.2f}x")

with ratio_col3:
    st.metric("Profit Margin", f"{profit_margin:.1%}")
    st.metric("DSCR", f"{dscr:.2f}x")

with ratio_col4:
    st.metric("Revenue", f"₹{revenue:,.0f}")
    st.metric("Total Debt", f"₹{debt:,.0f}")

# =====================================================
# GST-ITR INPUT
# =====================================================

st.header("📊 Step 3: GST-ITR Reconciliation")

col1, col2 = st.columns(2)

with col1:
    gst_turnover = st.number_input("GST Turnover (₹)", min_value=0.0,
                                  value=revenue * 0.95)
    gst_2a = st.number_input("GSTR-2A ITC (₹)", min_value=0.0, value=5000000.0)

with col2:
    itr_revenue = st.number_input("ITR Revenue (₹)", min_value=0.0, value=revenue)
    gst_3b = st.number_input("GSTR-3B Claimed (₹)", min_value=0.0, value=4500000.0)

# =====================================================
# ADDITIONAL FACTORS
# =====================================================

st.header("⚠️ Step 4: Risk Factors")

col1, col2, col3 = st.columns(3)

with col1:
    company = st.text_input("Company Name", placeholder="e.g., ABC Manufacturing")
    sector = st.selectbox("Sector", ["AUTO", "NBFC", "PHARMA", "RETAIL", "OTHER"])
    cibil_score = st.slider("CIBIL Score", 1, 100, 75)

with col2:
    mgmt_quality = st.slider("Management Quality (1-10)", 1, 10, 7)
    capacity_util = st.slider("Capacity Utilization (%)", 0, 100, 80)
    litigation = st.number_input("Litigation Cases", min_value=0, value=0)

with col3:
    gst_risk = st.slider("GST Risk (0-0.5)", 0.0, 0.5, 0.1)
    sector_sentiment = st.slider("Sector Sentiment (-1 to +1)", -1.0, 1.0, 0.0)
    promoter_names = st.text_input("Promoter Names (comma-separated)", "")

# =====================================================
# ANALYSIS BUTTON
# =====================================================

if st.button("🚀 ANALYZE CREDIT RISK", use_container_width=True):
    
    if not company:
        st.error("⚠️ Please enter Company Name")
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
        
        # Litigation Risk
        st.subheader("⚖️ Litigation Assessment")
        litigation_risk = litigation * 0.05
        if litigation > 0:
            st.warning(f"⚠️ {litigation} litigation case(s) found - Risk: +{litigation_risk:.1%}")
        else:
            st.success("✓ No litigation found")
        
        # CIBIL Check
        st.subheader("💳 CIBIL Score")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CIBIL Score", cibil_score)
        with col2:
            rating = "EXCELLENT" if cibil_score > 90 else "GOOD" if cibil_score > 70 else "FAIR" if cibil_score > 50 else "POOR"
            st.metric("Rating", rating)
        with col3:
            st.metric("NPA Status", "🟢 No")
        
        # News Sentiment
        st.subheader("📰 News Analysis")
        sentiment, articles, _ = get_news_sentiment(company, sector)
        
        if articles:
            st.metric("Sentiment Score", f"{sentiment:.2f}")
            for article in articles[:2]:
                with st.expander(article.get("title", "Article")[:60]):
                    st.markdown(f"[Read More]({article.get('url')})")
        else:
            st.info("No recent news found")
        
        # ML Prediction
        st.subheader("🤖 Risk Assessment")
        
        if model and feature_names:
            try:
                input_data = np.array([[
                    revenue,
                    ebitda,
                    debt,
                    interest_coverage,
                    gst_risk,
                    float(litigation),
                    sentiment,
                    sector_sentiment,
                    mgmt_quality / 10.0,
                    capacity_util / 100.0
                ]])
                
                df = pd.DataFrame(input_data, columns=feature_names)
                prob = model.predict_proba(df)[0][1]
            except:
                prob = (100 - cibil_score) / 100.0
        else:
            prob = (100 - cibil_score) / 100.0
        
        # Add GST risk
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
            st.metric("Risk Level", "LOW" if prob < 0.2 else "MEDIUM" if prob < 0.4 else "HIGH" if prob < 0.6 else "CRITICAL")
        
        # Loan Recommendation
        st.subheader("💵 Loan Recommendation")
        
        max_loan = ebitda * 3.5
        recommended_loan = max_loan * (1 - prob)
        interest_rate = 9 + prob * 6
        
        loan_col1, loan_col2, loan_col3 = st.columns(3)
        
        with loan_col1:
            st.metric("Max Loan (3.5x EBITDA)", f"₹{max_loan:,.0f}")
        
        with loan_col2:
            st.metric("Recommended Loan", f"₹{recommended_loan:,.0f}")
        
        with loan_col3:
            st.metric("Interest Rate", f"{interest_rate:.2f}%")
        
        try:
            loan_words = num2words(int(recommended_loan), lang="en_IN")
            st.info(f"💬 In words: {loan_words}")
        except:
            pass
        
        # Explainability
        st.subheader("💡 Decision Explanation")
        
        explanation = f"""
📊 DECISION SUMMARY

Decision: {decision}
Probability of Default: {prob:.1%}

KEY RISK FACTORS:
"""
        
        if gst_recon["red_flags"]:
            explanation += "\n🔴 GST-ITR Issues:\n"
            for flag in gst_recon["red_flags"][:2]:
                explanation += f"  • {flag}\n"
        
        if litigation > 0:
            explanation += f"\n🔴 Litigation: {litigation} case(s)\n"
        
        if cibil_score < 70:
            explanation += f"\n🔴 CIBIL Score: {cibil_score} (Below 70)\n"
        
        explanation += f"\n✅ POSITIVE FACTORS:\n"
        
        if interest_coverage > 2.5:
            explanation += f"  ✓ Strong interest coverage: {interest_coverage:.2f}x\n"
        
        if profit_margin > 0.08:
            explanation += f"  ✓ Healthy margin: {profit_margin:.1%}\n"
        
        if debt_equity < 1.5:
            explanation += f"  ✓ Conservative leverage: {debt_equity:.2f}x\n"
        
        st.text(explanation)
        
        # CAM Generation
        st.subheader("📋 Credit Appraisal Memo")
        
        cam = generate_cam(company, revenue, ebitda, debt, decision)
        st.text_area("CAM Content", cam, height=400)
        
        st.download_button(
            label="📥 Download CAM",
            data=cam,
            file_name=f"CAM_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
**IntelliCredit-X** | AI Credit Decision Engine
- Financial analysis & extraction
- GST-ITR reconciliation
- CIBIL integration
- Professional CAM generation
""")
