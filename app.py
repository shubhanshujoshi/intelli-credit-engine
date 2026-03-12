import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import re
import requests
import pdfplumber
from datetime import datetime, timedelta
from num2words import num2words
from google import genai

# =======================================================
# PAGE CONFIG
# =======================================================

st.set_page_config(
    layout="wide",
    page_title="IntelliCredit-X",
    initial_sidebar_state="expanded"
)

st.title("🏦 IntelliCredit-X | AI Powered Credit Risk Analyzer")

# =======================================================
# API KEYS
# =======================================================

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

# =======================================================
# LOAD MODEL & CONFIG
# =======================================================

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

# =======================================================
# SECTOR DATABASE (Lightweight, No External Calls)
# =======================================================

SECTOR_CONFIG = {
    "NBFC": {
        "regulatory_risk": 0.7,
        "headwinds": "RBI liquidity tightening, rising NPA pressure",
        "typical_interest_rate": 12.5
    },
    "Pharma": {
        "regulatory_risk": 0.4,
        "headwinds": "Price controls, generic competition, API costs",
        "typical_interest_rate": 9.5
    },
    "Manufacturing": {
        "regulatory_risk": 0.5,
        "headwinds": "Supply chain volatility, energy cost inflation",
        "typical_interest_rate": 10.0
    },
    "IT Services": {
        "regulatory_risk": 0.3,
        "headwinds": "Visa restrictions, client concentration",
        "typical_interest_rate": 8.5
    },
    "Real Estate": {
        "regulatory_risk": 0.6,
        "headwinds": "Land acquisition costs, regulatory delays",
        "typical_interest_rate": 11.0
    },
    "Retail": {
        "regulatory_risk": 0.5,
        "headwinds": "E-commerce disruption, margin compression",
        "typical_interest_rate": 10.5
    },
    "Energy": {
        "regulatory_risk": 0.6,
        "headwinds": "Volatile commodity prices, energy transition",
        "typical_interest_rate": 11.0
    },
    "Telecom": {
        "regulatory_risk": 0.7,
        "headwinds": "Spectrum costs, tariff wars, debt burden",
        "typical_interest_rate": 12.0
    },
    "Other": {
        "regulatory_risk": 0.5,
        "headwinds": "General business risk",
        "typical_interest_rate": 10.5
    }
}

# =======================================================
# FIVE Cs OF CREDIT FRAMEWORK
# =======================================================

def calculate_five_cs(revenue, pbt, debt, equity, litigation, sentiment,
                      management_quality, capacity_util, gst_variance, sector_risk):
    scores = {}

    character_score = 100
    character_score -= (litigation * 5)
    character_score -= (max(0, gst_variance - 0.1) * 100)
    character_score = max(0, character_score) + (sentiment * 10)
    scores["Character"] = min(100, max(0, character_score))

    if revenue > 0:
        profit_margin = (pbt / revenue) * 100
    else:
        profit_margin = 0

    capacity_score = 50 + (min(profit_margin, 15) * 2)
    capacity_score += (capacity_util * 20)
    scores["Capacity"] = min(100, max(0, capacity_score))

    if equity > 0:
        debt_to_equity = debt / equity
    else:
        debt_to_equity = 10

    capital_score = 100 - (debt_to_equity * 10)
    capital_score += (management_quality * 3)
    scores["Capital"] = min(100, max(0, capital_score))

    total_assets = debt + equity
    if total_assets > 0:
        collateral_ratio = total_assets / debt if debt > 0 else 2.0
    else:
        collateral_ratio = 1.0

    collateral_score = min(collateral_ratio, 3.0) * 33.33
    scores["Collateral"] = min(100, max(0, collateral_score))

    conditions_score = 100 - (sector_risk * 40)
    scores["Conditions"] = min(100, max(0, conditions_score))

    return scores

# =======================================================
# PDF TEXT EXTRACTION
# =======================================================

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

# =======================================================
# AI FINANCIAL EXTRACTION
# =======================================================

def ai_extract_financials(text):
    if not client or text == "":
        return {}

    prompt = f"""
Extract financial values from this financial statement.

Return ONLY valid JSON with these keys:
- revenue (numeric)
- profit_before_tax (numeric)
- finance_cost (numeric)
- depreciation (numeric)
- total_debt (numeric)
- total_equity (numeric)
- total_assets (numeric)

If value not found, use null.

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

# =======================================================
# ENHANCED NEWS SENTIMENT (Company + Promoter + Sector)
# =======================================================

def get_news_sentiment(company, promoter=None, sector=None):
    if not GNEWS_API_KEY:
        return 0, []

    end = datetime.utcnow()
    start = end - timedelta(days=60)

    all_articles = []
    combined_text = ""

    if company:
        try:
            url = (
                f"https://gnews.io/api/v4/search?"
                f"q=\"{company}\" AND (business OR earnings OR finance OR credit)"
                f"&from={start.strftime('%Y-%m-%d')}"
                f"&to={end.strftime('%Y-%m-%d')}"
                f"&lang=en"
                f"&max=10"
                f"&apikey={GNEWS_API_KEY}"
            )
            response = requests.get(url, timeout=10)
            articles = response.json().get("articles",[])
            all_articles.extend(articles)
            for a in articles:
                combined_text += a.get("title","") + ". "
        except:
            pass

    if promoter and len(promoter.strip()) > 0:
        try:
            url = (
                f"https://gnews.io/api/v4/search?"
                f"q=\"{promoter}\" AND (business OR fraud OR scandal OR arrest)"
                f"&from={start.strftime('%Y-%m-%d')}"
                f"&to={end.strftime('%Y-%m-%d')}"
                f"&lang=en"
                f"&max=2"
                f"&apikey={GNEWS_API_KEY}"
            )
            response = requests.get(url, timeout=10)
            articles = response.json().get("articles",[])
            all_articles.extend(articles)
            for a in articles:
                combined_text += a.get("title","") + ". "
        except:
            pass

    if sector:
        try:
            sector_keywords = SECTOR_CONFIG.get(sector, {}).get("headwinds", "")
            if sector_keywords:
                url = (
                    f"https://gnews.io/api/v4/search?"
                    f"q=\"{sector}\" AND ({sector_keywords.split(',')[0]})"
                    f"&from={start.strftime('%Y-%m-%d')}"
                    f"&to={end.strftime('%Y-%m-%d')}"
                    f"&lang=en"
                    f"&max=2"
                    f"&apikey={GNEWS_API_KEY}"
                )
                response = requests.get(url, timeout=10)
                articles = response.json().get("articles",[])
                all_articles.extend(articles)
                for a in articles:
                    combined_text += a.get("title","") + ". "
        except:
            pass

    sentiment = 0
    if client and combined_text != "":
        prompt = f"""
Return sentiment score between -1 and 1.
-1 = very negative, 0 = neutral, 1 = very positive

News:
{combined_text[:2000]}
"""
        try:
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            numbers = re.findall(r'-?\d+\.?\d*', resp.text)
            if numbers:
                sentiment = float(numbers[0])
        except:
            pass

    seen = set()
    unique_articles = []
    for a in all_articles:
        url = a.get("url")
        if url not in seen:
            seen.add(url)
            unique_articles.append(a)

    return max(-1, min(sentiment, 1)), unique_articles[:5]

# =======================================================
# LITIGATION RISK ASSESSMENT
# =======================================================

def assess_litigation_risk(litigation_count, litigation_notes=""):
    base_risk = min(litigation_count * 0.1, 0.5)

    severity_keywords = ["fraud", "criminal", "writ petition", "nclt"]
    severity_penalty = 0
    if litigation_notes:
        for keyword in severity_keywords:
            if keyword.lower() in litigation_notes.lower():
                severity_penalty += 0.15

    total_risk = min(base_risk + severity_penalty, 0.8)

    return {
        "risk_score": total_risk,
        "severity": "High" if total_risk > 0.5 else "Medium" if total_risk > 0.2 else "Low",
        "count": litigation_count
    }

# =======================================================
# GST COMPLIANCE CHECK
# =======================================================

def validate_gst_compliance(gst_turnover, bank_statement_turnover, gst_variance_input):
    if gst_turnover > 0 and bank_statement_turnover > 0:
        actual_variance = abs(gst_turnover - bank_statement_turnover) / gst_turnover
    else:
        actual_variance = gst_variance_input

    flags = []
    risk_level = "Low"

    if actual_variance > 0.2:
        flags.append("⚠️ High variance between GST and bank statements (>20%)")
        risk_level = "High"
    elif actual_variance > 0.1:
        flags.append("⚠️ Moderate variance between GST and bank statements (10-20%)")
        risk_level = "Medium"

    return {
        "variance": actual_variance,
        "flags": flags,
        "risk_level": risk_level
    }

# =======================================================
# RISK RATING (AAA to D)
# =======================================================

def calculate_risk_rating(pd_probability):
    if pd_probability < 0.02:
        return "AAA", "Minimal Credit Risk"
    elif pd_probability < 0.05:
        return "AA", "Very Low Credit Risk"
    elif pd_probability < 0.08:
        return "A", "Low Credit Risk"
    elif pd_probability < 0.12:
        return "BBB", "Moderate Credit Risk"
    elif pd_probability < 0.20:
        return "BB", "High Credit Risk"
    elif pd_probability < 0.35:
        return "B", "Very High Credit Risk"
    else:
        return "C", "Extremely High Credit Risk"

# =======================================================
# DECISION ROADMAP
# =======================================================

def generate_decision_roadmap(decision, pd_probability, five_cs_scores,
                               litigation_risk, gst_check, sentiment,
                               interest_coverage, debt_to_equity, sector_headwinds):
    roadmap = []

    if decision == "Rejected":
        roadmap.append("**❌ LOAN REJECTED**\n")
    else:
        roadmap.append("**✅ LOAN APPROVED**\n")

    roadmap.append(f"**Probability of Default**: {pd_probability:.2%}\n")

    risk_factors = []

    for c_name, c_score in five_cs_scores.items():
        if c_score < 40:
            risk_factors.append(f"• **{c_name}**: CRITICAL (Score: {c_score:.0f}/100)")
        elif c_score < 60:
            risk_factors.append(f"• **{c_name}**: WEAK (Score: {c_score:.0f}/100)")

    if litigation_risk["severity"] == "High":
        risk_factors.append(f"• **Litigation Risk**: HIGH ({litigation_risk['count']} active cases)")
    elif litigation_risk["severity"] == "Medium":
        risk_factors.append(f"• **Litigation Risk**: MEDIUM ({litigation_risk['count']} cases)")

    if gst_check["risk_level"] == "High":
        risk_factors.append(f"• **GST Compliance**: HIGH variance ({gst_check['variance']:.1%})")
    elif gst_check["risk_level"] == "Medium":
        risk_factors.append(f"• **GST Compliance**: MEDIUM variance ({gst_check['variance']:.1%})")

    if interest_coverage < 1.0:
        risk_factors.append(f"• **Interest Coverage**: CRITICAL ({interest_coverage:.2f}x < 1.0x minimum)")
    elif interest_coverage < 1.5:
        risk_factors.append(f"• **Interest Coverage**: WEAK ({interest_coverage:.2f}x)")

    if debt_to_equity > 3.0:
        risk_factors.append(f"• **Debt/Equity**: HIGH LEVERAGE ({debt_to_equity:.2f}x)")
    elif debt_to_equity > 2.0:
        risk_factors.append(f"• **Debt/Equity**: ELEVATED ({debt_to_equity:.2f}x)")

    if sentiment < -0.5:
        risk_factors.append(f"• **Market Sentiment**: VERY NEGATIVE (Score: {sentiment:.2f})")
    elif sentiment < -0.2:
        risk_factors.append(f"• **Market Sentiment**: NEGATIVE (Score: {sentiment:.2f})")

    if risk_factors:
        roadmap.append("\n**Key Risk Factors:**\n")
        for factor in risk_factors[:5]:
            roadmap.append(factor + "\n")

    positive_factors = []

    for c_name, c_score in five_cs_scores.items():
        if c_score > 80:
            positive_factors.append(f"• **{c_name}**: STRONG (Score: {c_score:.0f}/100)")

    if interest_coverage >= 2.0:
        positive_factors.append(f"• **Interest Coverage**: HEALTHY ({interest_coverage:.2f}x)")

    if gst_check["risk_level"] == "Low":
        positive_factors.append(f"• **GST Compliance**: GOOD (Variance: {gst_check['variance']:.1%})")

    if sentiment > 0.3:
        positive_factors.append(f"• **Market Sentiment**: POSITIVE (Score: {sentiment:.2f})")

    if positive_factors:
        roadmap.append("\n**Strengths:**\n")
        for factor in positive_factors:
            roadmap.append(factor + "\n")

    if sector_headwinds:
        roadmap.append(f"\n**Sector Context**: {sector_headwinds}\n")

    return "".join(roadmap)

# =======================================================
# ENHANCED CAM GENERATION
# =======================================================

def generate_cam(company, revenue, ebitda, debt, equity, decision,
                 five_cs_scores, risk_rating, interest_coverage, pd_probability):
    if not client:
        return f"""
# CREDIT APPRAISAL MEMO

**Company**: {company}

**Decision**: {decision}

**Risk Rating**: {risk_rating}

---

## Financial Metrics
- Revenue: ₹{revenue:,.0f}
- EBITDA: ₹{ebitda:,.0f}
- Total Debt: ₹{debt:,.0f}
- Equity: ₹{equity:,.0f}
- Interest Coverage: {interest_coverage:.2f}x

---

## Five Cs Assessment

"""

    five_cs_text = "\n".join([f"**{k}**: {v:.0f}/100" for k, v in five_cs_scores.items()])

    prompt = f"""
Create a professional Credit Appraisal Memo (CAM) in markdown format.

Company: {company}
Decision: {decision}
Risk Rating: {risk_rating}
Probability of Default: {pd_probability:.2%}

Financial Metrics:
- Revenue: {revenue:,.0f}
- EBITDA: {ebitda:,.0f}
- Debt: {debt:,.0f}
- Equity: {equity:,.0f}
- Interest Coverage: {interest_coverage:.2f}x

Five Cs Scores:
{five_cs_text}

Create a concise (max 300 words) Credit Appraisal Memo covering:
1. Executive Summary
2. Key Risk Factors
3. Financial Strength
4. Recommendation
5. Conditions of Disbursement
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except:
        return f"CAM generation unavailable."

# =======================================================
# INTEREST RATE CALCULATION
# =======================================================

def calculate_interest_rate(pd_probability, sector, interest_coverage, debt_to_equity):
    base_rate = 9.0
    pd_premium = pd_probability * 4
    sector_premium = SECTOR_CONFIG.get(sector, {}).get("regulatory_risk", 0.5) * 2

    if debt_to_equity > 3.0:
        leverage_premium = 1.5
    elif debt_to_equity > 2.0:
        leverage_premium = 1.0
    elif debt_to_equity > 1.5:
        leverage_premium = 0.5
    else:
        leverage_premium = 0.0

    if interest_coverage > 3.0:
        coverage_discount = -0.5
    elif interest_coverage > 2.0:
        coverage_discount = -0.25
    else:
        coverage_discount = 0.0

    total_rate = base_rate + pd_premium + sector_premium + leverage_premium + coverage_discount

    return {
        "total_rate": max(7.0, min(total_rate, 20.0)),
        "base_rate": base_rate,
        "pd_premium": pd_premium,
        "sector_premium": sector_premium,
        "leverage_premium": leverage_premium,
        "coverage_discount": coverage_discount
    }

# =======================================================
# FINANCIAL RATIO ENGINE
# =======================================================

def calculate_financial_ratios(revenue, ebitda, debt, equity, finance_cost):
    ratios = {}

    if revenue > 0:
        ratios["EBITDA Margin"] = ebitda / revenue
    else:
        ratios["EBITDA Margin"] = 0

    if equity > 0:
        ratios["Debt to Equity"] = debt / equity
    else:
        ratios["Debt to Equity"] = 0

    if finance_cost > 0:
        ratios["Interest Coverage"] = ebitda / finance_cost
    else:
        ratios["Interest Coverage"] = 0

    if debt > 0 and ebitda > 0:
        ratios["Debt to EBITDA"] = debt / ebitda
    else:
        ratios["Debt to EBITDA"] = 0

    return ratios

# =======================================================
# EARLY WARNING SIGNAL DETECTION
# =======================================================

def detect_early_warnings(interest_coverage, debt_to_equity, gst_variance, sentiment):
    warnings = []

    if interest_coverage < 1:
        warnings.append("⚠ Company unable to cover interest payments")

    if debt_to_equity > 3:
        warnings.append("⚠ Excessive leverage detected")

    if gst_variance > 0.2:
        warnings.append("⚠ Possible revenue inflation or circular trading")

    if sentiment < -0.4:
        warnings.append("⚠ Negative market sentiment detected")

    return warnings

# =======================================================
# CORPORATE CREDIT SCORE
# =======================================================

def generate_credit_score(pd_probability):
    score = int(900 - (pd_probability * 600))
    score = max(300, min(score, 900))
    return score

# =======================================================
# INDUSTRY BENCHMARKS
# =======================================================

INDUSTRY_BENCHMARKS = {
    "IT Services": {"Debt/Equity": 0.5, "Interest Coverage": 5},
    "Manufacturing": {"Debt/Equity": 1.5, "Interest Coverage": 3},
    "Real Estate": {"Debt/Equity": 2.5, "Interest Coverage": 2},
    "Retail": {"Debt/Equity": 1.8, "Interest Coverage": 2.5},
    "Pharma": {"Debt/Equity": 1.0, "Interest Coverage": 4}
}

def compare_to_industry(sector, debt_to_equity, interest_coverage):
    benchmark = INDUSTRY_BENCHMARKS.get(sector)

    if not benchmark:
        return []

    insights = []

    if debt_to_equity > benchmark["Debt/Equity"]:
        insights.append("⚠ Leverage above industry average")

    if interest_coverage < benchmark["Interest Coverage"]:
        insights.append("⚠ Interest coverage weaker than industry average")

    return insights

# =======================================================
# SHAP WATERFALL CHART — ALWAYS RENDERS
# Works with real XGBoost model OR falls back to
# rule-based simulation if model files are missing
# =======================================================

def render_shap_waterfall(df, input_values, pd_probability, feature_names_list,
                          explainer_obj, model_obj):
    """
    Renders a SHAP waterfall chart.
    - If real explainer is available: uses actual SHAP values
    - If not: simulates interpretable contributions from raw inputs
    """

    st.subheader("Feature Importance Analysis (SHAP Waterfall)")

    # ── CASE 1: Real SHAP from XGBoost ──────────────────────────────────────
    if explainer_obj is not None and model_obj is not None and df is not None:
        try:
            shap_values = explainer_obj(df)

            # Handle base_value being list (binary XGBoost classifier)
            base_value = explainer_obj.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1])
            else:
                base_value = float(base_value)

            # Handle shap values being 2D (binary classifier returns both classes)
            raw_shap = shap_values[0].values
            if raw_shap.ndim == 2:
                raw_shap = raw_shap[:, 1]

            feature_vals = df.iloc[0].values

            # Top 10 by absolute magnitude
            indices = np.argsort(np.abs(raw_shap))[::-1][:10]
            sorted_features = [feature_names_list[i] for i in indices]
            sorted_shap    = raw_shap[indices]
            sorted_vals    = feature_vals[indices]

            _draw_waterfall(sorted_features, sorted_shap, sorted_vals, base_value, source="XGBoost SHAP")
            return

        except Exception as e:
            st.warning(f"Real SHAP failed ({e}). Showing rule-based interpretation instead.")

    # ── CASE 2: Simulated interpretable contributions ────────────────────────
    # Build proxy SHAP values from the 10 input features using domain logic.
    # Signs and magnitudes are meaningful and match credit risk intuition.

    rev, ebitda_val, debt, ic, gst_var, litig, sent, sec_risk, mgmt, cap_util = input_values

    base_value = 0.30  # Approximate dataset average PD

    feature_display = [
        "Litigation Cases",
        "GST Variance",
        "Interest Coverage",
        "Sector Risk",
        "Sentiment Score",
        "Debt (Total)",
        "Management Quality",
        "Capacity Utilization",
        "EBITDA",
        "Revenue"
    ]

    # Each contribution: positive = increases default risk, negative = reduces it
    contrib = np.array([
        litig * 0.005,                              # More cases → higher risk
        gst_var * 0.3,                              # Higher variance → higher risk
        -min(ic, 10) * 0.015,                       # Higher coverage → lower risk
        sec_risk * 0.08,                            # Higher sector risk → higher risk
        -sent * 0.04,                               # Positive news → lower risk
        (debt / max(rev, 1)) * 0.02,               # Higher leverage → higher risk
        -(mgmt / 10) * 0.04,                        # Better mgmt → lower risk
        -(cap_util) * 0.03,                         # Higher utilization → lower risk
        -(ebitda_val / max(rev, 1)) * 0.05,        # Higher EBITDA margin → lower risk
        -(min(rev, 1e10) / 1e9) * 0.01             # Larger revenue → slightly lower risk
    ])

    # Scale contributions so they sum to (pd_probability - base_value)
    total_contrib = contrib.sum()
    actual_gap    = pd_probability - base_value
    if abs(total_contrib) > 1e-6:
        contrib = contrib * (actual_gap / total_contrib)

    raw_values = [litig, gst_var, ic, sec_risk, sent, debt, mgmt, cap_util, ebitda_val, rev]

    # Sort by absolute contribution
    indices = np.argsort(np.abs(contrib))[::-1][:10]
    sorted_features = [feature_display[i] for i in indices]
    sorted_shap     = contrib[indices]
    sorted_vals     = [raw_values[i] for i in indices]

    _draw_waterfall(sorted_features, sorted_shap, sorted_vals, base_value,
                    source="Rule-Based Interpretation")


def _draw_waterfall(sorted_features, sorted_shap, sorted_vals, base_value, source="SHAP"):
    """
    Core waterfall drawing — shared by both real SHAP and simulated paths.
    """
    n = len(sorted_features)

    # Build cumulative waterfall positions
    cumulative = np.zeros(n + 1)
    cumulative[0] = base_value
    for i in range(n):
        cumulative[i + 1] = cumulative[i] + sorted_shap[i]

    final_pred = cumulative[-1]

    colors = ['#d62728' if x > 0 else '#2ca02c' for x in sorted_shap]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    for i, (feat, sv, cv) in enumerate(zip(sorted_features, sorted_shap, cumulative[:-1])):
        bar = ax.barh(i, sv, left=cv, color=colors[i], alpha=0.88,
                      edgecolor='white', linewidth=0.4, height=0.6)
        mid = cv + sv / 2
        ax.text(mid, i, f'{sv:+.4f}', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')

    # Connector lines between bars
    for i in range(n - 1):
        ax.plot([cumulative[i + 1], cumulative[i + 1]],
                [i + 0.3, i + 0.7], color='white', linewidth=0.6, alpha=0.4)

    # Reference lines
    ax.axvline(x=base_value, color='#00bfff', linestyle='--', linewidth=2,
               label=f'Baseline PD: {base_value:.4f}', alpha=0.85)
    ax.axvline(x=final_pred, color='#ff4444', linestyle='--', linewidth=2,
               label=f'Predicted PD: {final_pred:.4f}', alpha=0.85)

    # Styling
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_features, fontsize=10, color='white')
    ax.set_xlabel('Contribution to Default Probability', fontsize=11,
                  fontweight='bold', color='white')
    ax.set_title(f'SHAP Waterfall — Top Feature Contributions to Default Risk\n({source})',
                 fontsize=12, fontweight='bold', color='white', pad=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.grid(axis='x', alpha=0.15, linestyle='--', color='white')
    ax.legend(loc='lower right', fontsize=9, facecolor='#1e1e2e',
              labelcolor='white', framealpha=0.8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("📖 How to interpret this chart"):
        st.markdown("""
- **Blue dashed line** — Baseline: average default probability across the dataset  
- **Red dashed line** — Final predicted default probability for this company  
- 🔴 **Red bars** — Features *increasing* default risk (pushing PD higher)  
- 🟢 **Green bars** — Features *reducing* default risk (pushing PD lower)  
- **Bar length** — Magnitude of impact; longer = stronger influence on the prediction
        """)


# =======================================================
# UI BEGINS HERE
# =======================================================

st.markdown("---")

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_pdf = st.file_uploader("Upload Financial Statement (PDF)", type=["pdf"])

ai_data = {}

if uploaded_pdf:
    st.sidebar.success("✅ PDF uploaded")
    text = extract_pdf_text(uploaded_pdf)
    ai_data = ai_extract_financials(text)
    if ai_data:
        st.sidebar.info("🤖 AI extracted financial values")

tab1, tab2, tab3 = st.tabs(["📊 Financial Analysis", "🔍 Research & Risk", "📋 Decision & CAM"])

# =======================================================
# TAB 1: FINANCIAL ANALYSIS
# =======================================================

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income Statement")
        revenue = st.number_input(
            "Revenue (₹)",
            value=float(ai_data.get("revenue", 500000000)),
            min_value=0.0,
            max_value=10000000000000.0,
            step=1000000.0
        )

        pbt = st.number_input(
            "Profit Before Tax (₹)",
            value=float(ai_data.get("profit_before_tax", 50000000)),
            min_value=0.0,
            step=1000000.0
        )

        finance_cost = st.number_input(
            "Finance Cost (₹)",
            value=float(ai_data.get("finance_cost", 10000000)),
            min_value=0.0,
            step=1000000.0
        )

        depreciation = st.number_input(
            "Depreciation (₹)",
            value=float(ai_data.get("depreciation", 10000000)),
            min_value=0.0,
            step=1000000.0
        )

    with col2:
        st.subheader("Balance Sheet")
        total_debt = st.number_input(
            "Total Debt (₹)",
            value=float(ai_data.get("total_debt", 200000000)),
            min_value=0.0,
            step=1000000.0
        )

        equity = st.number_input(
            "Total Equity (₹)",
            value=float(ai_data.get("total_equity", 200000000)),
            min_value=0.0,
            step=1000000.0
        )

        st.subheader("GST Compliance Check")
        gst_turnover = st.number_input(
            "GST Turnover (₹)",
            value=revenue * 0.95,
            min_value=0.0,
            step=1000000.0,
            help="From GST-3B returns"
        )

        bank_turnover = st.number_input(
            "Bank Statement Turnover (₹)",
            value=revenue,
            min_value=0.0,
            step=1000000.0,
            help="Total credits from bank statements"
        )

    ebitda = pbt + finance_cost + depreciation

    if finance_cost > 0:
        interest_coverage = ebitda / finance_cost
    else:
        interest_coverage = 0

    if equity > 0:
        debt_to_equity = total_debt / equity
    else:
        debt_to_equity = 10

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("EBITDA", f"₹{ebitda/1e7:.1f}Cr")
    with metric_col2:
        st.metric("Interest Coverage", f"{interest_coverage:.2f}x",
                 delta="Healthy" if interest_coverage > 1.5 else "Weak")
    with metric_col3:
        st.metric("Debt/Equity", f"{debt_to_equity:.2f}x",
                 delta="High" if debt_to_equity > 2.5 else "Moderate")
    with metric_col4:
        st.metric("Profit Margin", f"{(pbt/revenue)*100:.1f}%" if revenue > 0 else "N/A")

    ratios = calculate_financial_ratios(revenue, ebitda, total_debt, equity, finance_cost)

    st.subheader("Financial Ratio Analysis")

    ratio_cols = st.columns(len(ratios))

    for i, (name, value) in enumerate(ratios.items()):
        ratio_cols[i].metric(name, f"{value:.2f}")

# =======================================================
# TAB 2: RESEARCH & RISK
# =======================================================

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Company Details")
        company = st.text_input("Company Name", value="")
        promoter = st.text_input("Promoter Name (Optional)", value="")

        st.subheader("Litigation Risk")
        litigation = st.number_input(
            "Number of Active Legal Cases",
            value=1,
            min_value=0,
            max_value=1000,
            step=1
        )
        litigation_notes = st.text_area(
            "Litigation Details (e.g., fraud, criminal, NCLT)",
            value="",
            height=80
        )

    with col2:
        st.subheader("Operational Risk")
        sector = st.selectbox(
            "Industry Sector",
            list(SECTOR_CONFIG.keys()),
            index=4
        )

        capacity_util = st.slider(
            "Capacity Utilization",
            0.0, 1.0, 0.8,
            help="Operating capacity as % of maximum"
        )

        mgmt = st.slider(
            "Management Quality (1-10)",
            1, 10, 7,
            help="Assessment of promoter track record & experience"
        )

        gst_mismatch = st.slider(
            "GST Variance (Manual Override)",
            0.0, 0.5, 0.1,
            help="If GST inputs not available, use this estimate"
        )

    st.subheader("News Intelligence")

    if st.button("🔍 Fetch Market News"):
        with st.spinner("Searching news..."):
            sentiment, articles = get_news_sentiment(company, promoter, sector)
            st.session_state.sentiment = sentiment
            st.session_state.articles = articles

    if "articles" in st.session_state:
        articles = st.session_state.articles
        sentiment = st.session_state.sentiment

        st.metric("News Sentiment", f"{sentiment:+.2f}",
                 delta="Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral")

        if articles:
            st.write("**Recent Articles:**")
            for article in articles:
                title  = article.get("title","")
                url    = article.get("url","")
                source = article.get("source",{}).get("name","Unknown")
                st.markdown(f"📌 [{title}]({url})")
                st.caption(f"Source: {source}")
                st.write("---")
        else:
            st.info("No relevant news found")
    else:
        sentiment = 0
        articles  = []

    if sector in SECTOR_CONFIG:
        st.info(f"**Sector Headwinds**: {SECTOR_CONFIG[sector]['headwinds']}")

# =======================================================
# TAB 3: DECISION & CAM
# =======================================================

with tab3:
    st.subheader("Credit Decision")

    if st.button("⚡ Analyze & Generate CAM", key="analyze"):
        if not company:
            st.error("Please enter Company Name")
        else:
            # Sentiment
            if "sentiment" not in st.session_state:
                sentiment, articles = get_news_sentiment(company, promoter, sector)
                st.session_state.sentiment = sentiment
                st.session_state.articles  = articles
            else:
                sentiment = st.session_state.sentiment
                articles  = st.session_state.articles

            # Five Cs
            five_cs = calculate_five_cs(
                revenue, pbt, total_debt, equity, litigation,
                sentiment, mgmt, capacity_util, gst_mismatch,
                SECTOR_CONFIG[sector]["regulatory_risk"]
            )

            litig_risk = assess_litigation_risk(litigation, litigation_notes)
            gst_check  = validate_gst_compliance(gst_turnover, bank_turnover, gst_mismatch)

            # ── Model prediction ──────────────────────────────────────────
            input_values_raw = [
                revenue, ebitda, total_debt, interest_coverage,
                gst_mismatch, litigation, sentiment,
                SECTOR_CONFIG[sector]["regulatory_risk"],
                mgmt, capacity_util
            ]

            df_input = None

            if model and feature_names:
                input_array = np.array([input_values_raw])
                df_input = pd.DataFrame(input_array, columns=feature_names)
                pd_probability = float(model.predict_proba(df_input)[0][1])
            else:
                # Fallback: rule-based PD estimate when model files are absent
                base_pd  = 0.15
                base_pd += min(litigation * 0.01, 0.20)
                base_pd += gst_mismatch * 0.15
                base_pd -= min(interest_coverage * 0.01, 0.10)
                base_pd += SECTOR_CONFIG[sector]["regulatory_risk"] * 0.10
                base_pd -= (mgmt / 10) * 0.05
                base_pd -= capacity_util * 0.05
                pd_probability = max(0.01, min(base_pd, 0.99))

            pd_probability = max(0.0, min(pd_probability, 1.0))

            risk_rating, risk_description = calculate_risk_rating(pd_probability)
            decision = "Approved" if pd_probability <= threshold else "Rejected"

            # ── Result header ─────────────────────────────────────────────
            result_col1, result_col2, result_col3 = st.columns(3)
            credit_score = generate_credit_score(pd_probability)
            st.metric("Corporate Credit Score", credit_score)

            with result_col1:
                if decision == "Approved":
                    st.success(f"{decision}", icon="✅")
                else:
                    st.error(f"{decision}", icon="❌")

            with result_col2:
                st.metric("Risk Rating", risk_rating, delta=risk_description)

            with result_col3:
                st.metric("Probability of Default (PD)", f"{pd_probability:.2%}")

            # ── Five Cs ───────────────────────────────────────────────────
            st.subheader("Five Cs of Credit Assessment")
            c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns(5)

            c_metrics = [
                (c_col1, "Character"),
                (c_col2, "Capacity"),
                (c_col3, "Capital"),
                (c_col4, "Collateral"),
                (c_col5, "Conditions")
            ]

            for col, c_name in c_metrics:
                score = five_cs[c_name]
                with col:
                    if score >= 70:
                        status = "✅"
                    elif score >= 50:
                        status = "⚠️"
                    else:
                        status = "❌"
                    st.metric(f"{status} {c_name}", f"{score:.0f}/100")

            # ── GST Status ────────────────────────────────────────────────
            st.subheader("GST Compliance Status")
            gst_col1, gst_col2, gst_col3 = st.columns(3)

            with gst_col1:
                st.metric("Variance", f"{gst_check['variance']:.1%}")
            with gst_col2:
                risk_color = "🔴" if gst_check['risk_level'] == "High" else "🟡" if gst_check['risk_level'] == "Medium" else "🟢"
                st.metric("Risk Level", f"{risk_color} {gst_check['risk_level']}")
            with gst_col3:
                st.metric("Litigation Severity", litig_risk["severity"])

            # ── Early Warnings ────────────────────────────────────────────
            warnings = detect_early_warnings(interest_coverage, debt_to_equity,
                                             gst_check["variance"], sentiment)
            if warnings:
                st.subheader("Early Warning Signals")
            for w in warnings:
                st.error(w)

            # ── Industry Benchmarks ───────────────────────────────────────
            industry_insights = compare_to_industry(sector, debt_to_equity, interest_coverage)
            if industry_insights:
                st.subheader("Industry Benchmark Comparison")
            for insight in industry_insights:
                st.warning(insight)

            # ── Decision Roadmap ──────────────────────────────────────────
            st.subheader("Decision Roadmap")
            roadmap = generate_decision_roadmap(
                decision, pd_probability, five_cs, litig_risk, gst_check,
                sentiment, interest_coverage, debt_to_equity,
                SECTOR_CONFIG[sector].get("headwinds", "")
            )
            st.markdown(roadmap)

            # ── Loan Recommendation ───────────────────────────────────────
            st.subheader("Loan Recommendation")
            rate_details = calculate_interest_rate(
                pd_probability, sector, interest_coverage, debt_to_equity
            )

            rate_col1, rate_col2 = st.columns(2)

            with rate_col1:
                st.metric("Interest Rate", f"{rate_details['total_rate']:.2f}%")
                max_loan  = ebitda * 3.5
                loan_amount = max_loan * (1 - pd_probability)
                st.write(f"**Maximum Loan Amount**: ₹{loan_amount:,.0f}")
                try:
                    words = num2words(int(loan_amount), lang="en_IN")
                    st.caption(f"_{words}_")
                except:
                    pass

            with rate_col2:
                st.write("**Rate Breakdown:**")
                st.write(f"• Base Rate: {rate_details['base_rate']:.2f}%")
                st.write(f"• PD Premium: {rate_details['pd_premium']:.2f}%")
                st.write(f"• Sector Premium: {rate_details['sector_premium']:.2f}%")
                st.write(f"• Leverage Premium: {rate_details['leverage_premium']:.2f}%")
                st.write(f"• Coverage Discount: {rate_details['coverage_discount']:.2f}%")

            # ── SHAP WATERFALL ─────────────────────────────────────────────
            # Always renders — uses real SHAP if model loaded, rule-based otherwise
            render_shap_waterfall(
                df=df_input,
                input_values=input_values_raw,
                pd_probability=pd_probability,
                feature_names_list=feature_names if feature_names else [
                    "Revenue", "EBITDA", "Total Debt", "Interest Coverage",
                    "GST Variance", "Litigation", "Sentiment",
                    "Sector Risk", "Management Quality", "Capacity Utilization"
                ],
                explainer_obj=explainer,
                model_obj=model
            )

            # ── CAM ───────────────────────────────────────────────────────
            st.subheader("Credit Appraisal Memo")
            cam = generate_cam(
                company, revenue, ebitda, total_debt, equity, decision,
                five_cs, risk_rating, interest_coverage, pd_probability
            )
            st.markdown(cam)

            st.download_button(
                label="⬇️ Download CAM",
                data=cam,
                file_name=f"{company}_CAM_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

st.markdown("---")
st.caption("IntelliCredit-X | The Smart Credit Risk Analyzer")

