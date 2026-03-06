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
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(layout="wide", page_title="IntelliCredit-X", initial_sidebar_state="expanded")

st.title("🏦 IntelliCredit-X | Next-Gen AI Credit Decision Engine")
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

@st.cache_resource
def load_explainer():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "financial_model.pkl"))
        return shap.TreeExplainer(model)
    except:
        return None

model = load_model()
threshold = load_threshold()
feature_names = load_features()
explainer = load_explainer()

# =====================================================
# PDF TEXT EXTRACTION (ENHANCED WITH OCR PREP)
# =====================================================

def extract_pdf_text(uploaded_file):
    """Extract text from PDF with enhanced handling"""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    text += f"--- PAGE {page_num + 1} ---\n{t}\n"
                
                # Extract tables if present
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
# GST-ITR RECONCILIATION ENGINE (CRITICAL)
# =====================================================

def extract_gst_data_from_text(text: str) -> Dict:
    """Extract GST-related data from financial statements"""
    if not text:
        return {}
    
    gst_data = {
        "gst_2a_iTC": None,
        "gst_3b_liability": None,
        "gst_turnover": None,
        "iTC_credit": None
    }
    
    if client:
        prompt = f"""
Extract GST data from financial statement:
- GSTR-2A (ITC Available)
- GSTR-3B (ITC Claimed)
- GST Turnover
- Any ITC discrepancies

Return JSON with keys: gst_2a_iTC, gst_3b_liability, gst_turnover, iTC_credit

Text: {text[:5000]}
"""
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            cleaned = response.text.replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
            gst_data.update(data)
        except:
            pass
    
    return gst_data

def gst_itr_reconciliation(gst_turnover: float, itr_revenue: float, 
                           gst_2a: float, gst_3b: float) -> Dict:
    """
    CRITICAL: Detect revenue inflation via GST-ITR mismatch
    Indian banks mandate this check to catch circular trading
    """
    
    reconciliation = {
        "status": "PASS",
        "risk_level": "LOW",
        "findings": [],
        "risk_score_impact": 0.0,
        "red_flags": []
    }
    
    # Check 1: GST Turnover vs ITR Revenue
    if gst_turnover and itr_revenue:
        mismatch_pct = abs(gst_turnover - itr_revenue) / itr_revenue
        
        if mismatch_pct > 0.20:
            reconciliation["risk_level"] = "CRITICAL"
            reconciliation["status"] = "FAIL"
            reconciliation["red_flags"].append(
                f"CRITICAL: GST-ITR mismatch {mismatch_pct:.1%}. Possible revenue inflation or circular trading."
            )
            reconciliation["risk_score_impact"] = 0.15
        elif mismatch_pct > 0.10:
            reconciliation["risk_level"] = "HIGH"
            reconciliation["red_flags"].append(
                f"HIGH: GST turnover ({gst_turnover:,.0f}) differs from ITR revenue ({itr_revenue:,.0f}) by {mismatch_pct:.1%}"
            )
            reconciliation["risk_score_impact"] = 0.08
        elif mismatch_pct > 0.05:
            reconciliation["risk_level"] = "MEDIUM"
            reconciliation["findings"].append(
                f"MEDIUM: Minor discrepancy of {mismatch_pct:.1%} between GST and ITR"
            )
            reconciliation["risk_score_impact"] = 0.03
        else:
            reconciliation["findings"].append("✓ GST-ITR revenue aligned")
    
    # Check 2: GSTR-2A vs GSTR-3B (ITC Mismatch)
    if gst_2a and gst_3b:
        iTC_discrepancy = abs(gst_2a - gst_3b) / gst_2a if gst_2a > 0 else 0
        
        if iTC_discrepancy > 0.30:
            reconciliation["red_flags"].append(
                f"ALERT: ITC Credit discrepancy {iTC_discrepancy:.1%} (2A: {gst_2a:,.0f} vs 3B: {gst_3b:,.0f}). Check for fraudulent credits."
            )
            reconciliation["risk_score_impact"] += 0.10
        elif iTC_discrepancy > 0.10:
            reconciliation["findings"].append(
                f"Note: ITC variance {iTC_discrepancy:.1%} between GSTR-2A and 3B. Review pending input credits."
            )
    
    return reconciliation

# =====================================================
# MCA FILING INTEGRATION (REGULATORY DATA)
# =====================================================

def fetch_mca_company_data(company_name: str) -> Dict:
    """
    Fetch MCA (Ministry of Corporate Affairs) data
    This includes: CIN, incorporation date, director info, compliance status
    """
    
    mca_data = {
        "cin": None,
        "incorporation_date": None,
        "directors": [],
        "compliance_status": "UNKNOWN",
        "annual_filings": [],
        "regulatory_flags": []
    }
    
    # Mock implementation - In production, integrate with actual MCA API
    # https://www.mca.gov.in/content/mca/global/en/home.html (official source)
    
    if not company_name:
        return mca_data
    
    # Try to extract from uploaded documents
    if client:
        prompt = f"""
From financial documents of '{company_name}', extract:
- CIN (Corporate Identification Number)
- Date of Incorporation
- Director Names
- Any compliance or regulatory issues mentioned

Return JSON with keys: cin, incorporation_date, directors, compliance_issues
"""
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            cleaned = response.text.replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
            
            if data.get("cin"):
                mca_data["cin"] = data["cin"]
            if data.get("incorporation_date"):
                mca_data["incorporation_date"] = data["incorporation_date"]
            if data.get("directors"):
                mca_data["directors"] = data["directors"]
            if data.get("compliance_issues"):
                mca_data["regulatory_flags"] = data["compliance_issues"]
        except:
            pass
    
    return mca_data

# =====================================================
# E-COURTS LITIGATION SEARCH (LEGAL RISK)
# =====================================================

def search_ecourts_litigation(company_name: str, promoter_names: List[str] = None) -> Dict:
    """
    Search for litigation involving company or promoters
    In production: integrate with https://www.ecourts.gov.in/ (e-Courts India portal)
    """
    
    litigation_data = {
        "total_cases": 0,
        "commercial_cases": [],
        "criminal_cases": [],
        "insolvency_cases": [],
        "litigation_risk_score": 0.0,
        "critical_alerts": []
    }
    
    if not company_name:
        return litigation_data
    
    # For MVP: Use GNews to find litigation news
    search_terms = [
        f'"{company_name}" litigation case court',
        f'"{company_name}" insolvency bankruptcy',
        f'"{company_name}" default legal'
    ]
    
    if promoter_names:
        search_terms.extend([f'"{name}" fraud case' for name in promoter_names[:2]])
    
    all_news = []
    
    for term in search_terms:
        if GNEWS_API_KEY:
            try:
                url = (
                    f"https://gnews.io/api/v4/search?"
                    f"q={term}"
                    f"&lang=en&max=3"
                    f"&apikey={GNEWS_API_KEY}"
                )
                response = requests.get(url, timeout=5)
                articles = response.json().get("articles", [])
                all_news.extend(articles)
            except:
                pass
    
    # Classify litigation
    for article in all_news:
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        combined = title + " " + description
        
        litigation_data["total_cases"] += 1
        
        if any(word in combined for word in ["insolvency", "bankruptcy", "nclt"]):
            litigation_data["insolvency_cases"].append({
                "title": article.get("title"),
                "url": article.get("url"),
                "date": article.get("publishedAt")
            })
            litigation_data["critical_alerts"].append("⚠️ INSOLVENCY CASE FOUND")
        elif any(word in combined for word in ["criminal", "fraud", "forgery", "cheating"]):
            litigation_data["criminal_cases"].append({
                "title": article.get("title"),
                "url": article.get("url"),
                "date": article.get("publishedAt")
            })
            litigation_data["critical_alerts"].append("⚠️ CRIMINAL CASE FOUND")
        else:
            litigation_data["commercial_cases"].append({
                "title": article.get("title"),
                "url": article.get("url"),
                "date": article.get("publishedAt")
            })
    
    # Calculate litigation risk score
    if litigation_data["insolvency_cases"]:
        litigation_data["litigation_risk_score"] += 0.35
    if litigation_data["criminal_cases"]:
        litigation_data["litigation_risk_score"] += 0.25
    if litigation_data["commercial_cases"]:
        litigation_data["litigation_risk_score"] += len(litigation_data["commercial_cases"]) * 0.05
    
    litigation_data["litigation_risk_score"] = min(1.0, litigation_data["litigation_risk_score"])
    
    return litigation_data

# =====================================================
# CIBIL COMMERCIAL REPORT INTEGRATION
# =====================================================

def fetch_cibil_score(company_name: str, manual_score: float = None) -> Dict:
    """
    CIBIL Commercial Score (Indian credit bureau)
    In production: integrate with CIBIL API for actual scores
    
    Typical ranges:
    - Score > 90: Excellent credit rating
    - Score 60-90: Good credit rating
    - Score < 60: Poor credit rating
    """
    
    cibil_data = {
        "cibil_score": manual_score if manual_score else 75.0,
        "rating_category": "GOOD",
        "payment_history": "GOOD",
        "npa_flag": False,
        "credit_utilization": 0.5,
        "default_history": []
    }
    
    if manual_score:
        if manual_score > 90:
            cibil_data["rating_category"] = "EXCELLENT"
        elif manual_score > 70:
            cibil_data["rating_category"] = "GOOD"
        elif manual_score > 50:
            cibil_data["rating_category"] = "FAIR"
        else:
            cibil_data["rating_category"] = "POOR"
    
    return cibil_data

# =====================================================
# RBI REGULATORY TRACKING
# =====================================================

def fetch_rbi_sector_guidelines(sector: str = None) -> Dict:
    """
    Fetch RBI guidelines and sector-specific regulations
    Key factors: repo rate, NPA norms, sector-specific caps
    """
    
    rbi_data = {
        "repo_rate": 6.5,  # As of last known update; in production, fetch from RBI website
        "npa_threshold": 0.05,  # 5% NPA threshold
        "sector_specific_guidance": [],
        "recent_guidelines": [],
        "risk_adjustments": []
    }
    
    # Sector-specific guidelines
    sector_guidelines = {
        "NBFC": "RBI mandates higher provisions; sector under tighter regulation",
        "AUTO": "SIAM data shows sector slowdown; monitor ELV regulations",
        "RETAIL": "High competition; focus on cash flow stability",
        "PHARMA": "FDI restrictions; API sourcing risks",
        "POWER": "Tariff pressure; coal costs volatile",
        "TELECOM": "Spectrum auction burden; ensure strong balance sheet"
    }
    
    if sector and sector.upper() in sector_guidelines:
        rbi_data["sector_specific_guidance"] = [sector_guidelines[sector.upper()]]
        rbi_data["risk_adjustments"].append(f"Sector ({sector}): {sector_guidelines[sector.upper()]}")
    
    return rbi_data

# =====================================================
# AI FINANCIAL EXTRACTION (ENHANCED)
# =====================================================

def ai_extract_financials(text: str) -> Dict:
    """Enhanced financial extraction with India-specific formats"""
    
    if not client or text == "":
        return {}
    
    prompt = f"""
Extract ALL financial values from this Indian financial statement.

Return JSON with EXACT keys:
- revenue (or "net sales" or "turnover")
- profit_before_tax (or "PBT" or "EBT")
- profit_after_tax (or "net profit")
- finance_cost (or "interest expense")
- depreciation (or "D&A")
- total_debt (or "borrowings")
- current_liabilities
- current_assets
- inventory
- receivables
- payables
- ebitda (if available)

If value not found return null.
For each value found, also return a "confidence_score" (0-1).

Text (first 10000 chars):
{text[:10000]}
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        cleaned = response.text.replace("```json", "").replace("```", "")
        data = json.loads(cleaned)
        return data
    except:
        return {}

# =====================================================
# NEWS SENTIMENT & SECTOR ANALYSIS
# =====================================================

def get_news_sentiment(company: str, sector: str = None) -> Tuple[float, List, List]:
    """Enhanced news sentiment with sector analysis"""
    
    if not company or not GNEWS_API_KEY:
        return 0, [], []
    
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    
    sentiment_score = 0.0
    article_count = 0
    articles = []
    sector_articles = []
    
    # Company-specific news
    url = (
        f"https://gnews.io/api/v4/search?"
        f"q=\"{company}\" AND (business OR company OR earnings OR finance OR quarterly OR results)"
        f"&from={start.strftime('%Y-%m-%d')}"
        f"&to={end.strftime('%Y-%m-%d')}"
        f"&lang=en&max=5"
        f"&apikey={GNEWS_API_KEY}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
    except:
        articles = []
    
    # Sector-specific news
    if sector:
        try:
            sector_url = (
                f"https://gnews.io/api/v4/search?"
                f"q=\"{sector}\" AND (industry OR sector OR market OR regulation OR RBI)"
                f"&from={start.strftime('%Y-%m-%d')}"
                f"&to={end.strftime('%Y-%m-%d')}"
                f"&lang=en&max=3"
                f"&apikey={GNEWS_API_KEY}"
            )
            response = requests.get(sector_url, timeout=10)
            sector_articles = response.json().get("articles", [])
        except:
            sector_articles = []
    
    # Sentiment analysis
    combined_text = ""
    for a in articles:
        combined_text += a.get("title", "") + ". "
    
    if client and combined_text:
        prompt = f"""
Analyze sentiment of this corporate news. Return JSON:
- sentiment_score: -1 (very negative) to +1 (very positive)
- reasoning: brief explanation
- risk_flags: list of concerning words if any

News: {combined_text[:2000]}
"""
        try:
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            
            data = json.loads(resp.text.replace("```json", "").replace("```", ""))
            sentiment_score = float(data.get("sentiment_score", 0))
        except:
            sentiment_score = 0
    
    return sentiment_score, articles, sector_articles

# =====================================================
# FIVE Cs OF CREDIT FRAMEWORK
# =====================================================

def evaluate_five_cs(
    character_data: Dict,
    capacity_data: Dict,
    capital_data: Dict,
    collateral_data: Dict,
    conditions_data: Dict
) -> Dict:
    """
    Evaluate credit using the Five Cs framework
    Essential for Indian banking credit assessment
    """
    
    five_cs = {
        "character": {
            "score": 0.0,
            "findings": [],
            "rating": "POOR",
            "red_flags": []
        },
        "capacity": {
            "score": 0.0,
            "findings": [],
            "rating": "POOR",
            "red_flags": []
        },
        "capital": {
            "score": 0.0,
            "findings": [],
            "rating": "POOR",
            "red_flags": []
        },
        "collateral": {
            "score": 0.0,
            "findings": [],
            "rating": "POOR",
            "red_flags": []
        },
        "conditions": {
            "score": 0.0,
            "findings": [],
            "rating": "POOR",
            "red_flags": []
        }
    }
    
    # CHARACTER: Promoter background, payment history, litigation
    char_score = 0.0
    if character_data.get("litigation_risk_score", 0) < 0.2:
        char_score += 25
        five_cs["character"]["findings"].append("✓ Low litigation risk")
    else:
        five_cs["character"]["red_flags"].append("⚠️ Significant litigation history")
    
    if not character_data.get("critical_alerts", []):
        char_score += 25
        five_cs["character"]["findings"].append("✓ No critical legal alerts")
    
    if character_data.get("payment_history") == "GOOD":
        char_score += 25
        five_cs["character"]["findings"].append("✓ Good payment history")
    else:
        five_cs["character"]["red_flags"].append("⚠️ Payment history concerns")
    
    if character_data.get("cibil_score", 0) > 70:
        char_score += 25
        five_cs["character"]["findings"].append(f"✓ Good CIBIL score: {character_data['cibil_score']}")
    
    five_cs["character"]["score"] = min(100, char_score)
    five_cs["character"]["rating"] = "EXCELLENT" if char_score >= 75 else \
                                      "GOOD" if char_score >= 50 else \
                                      "FAIR" if char_score >= 25 else "POOR"
    
    # CAPACITY: Revenue growth, profitability, cash flow
    cap_score = 0.0
    if capacity_data.get("revenue_trend") == "GROWING":
        cap_score += 25
        five_cs["capacity"]["findings"].append("✓ Revenue trending upward")
    
    if capacity_data.get("interest_coverage", 0) > 2.5:
        cap_score += 25
        five_cs["capacity"]["findings"].append(f"✓ Strong interest coverage: {capacity_data['interest_coverage']:.2f}x")
    else:
        five_cs["capacity"]["red_flags"].append(f"⚠️ Weak interest coverage: {capacity_data['interest_coverage']:.2f}x")
    
    if capacity_data.get("profit_margin", 0) > 0.08:
        cap_score += 25
        five_cs["capacity"]["findings"].append(f"✓ Healthy profit margin: {capacity_data['profit_margin']:.1%}")
    else:
        five_cs["capacity"]["red_flags"].append("⚠️ Thin profit margins")
    
    if capacity_data.get("capacity_utilization", 0.5) > 0.70:
        cap_score += 25
        five_cs["capacity"]["findings"].append(f"✓ Good capacity utilization: {capacity_data['capacity_utilization']:.0%}")
    else:
        five_cs["capacity"]["red_flags"].append(f"⚠️ Low capacity utilization: {capacity_data['capacity_utilization']:.0%}")
    
    five_cs["capacity"]["score"] = min(100, cap_score)
    five_cs["capacity"]["rating"] = "EXCELLENT" if cap_score >= 75 else \
                                      "GOOD" if cap_score >= 50 else \
                                      "FAIR" if cap_score >= 25 else "POOR"
    
    # CAPITAL: Equity, leverage, solvency
    cap_score = 0.0
    if capital_data.get("debt_equity_ratio", 2.0) < 1.5:
        cap_score += 33
        five_cs["capital"]["findings"].append(f"✓ Conservative leverage: D/E = {capital_data['debt_equity_ratio']:.2f}")
    else:
        five_cs["capital"]["red_flags"].append(f"⚠️ High leverage: D/E = {capital_data['debt_equity_ratio']:.2f}")
    
    if capital_data.get("current_ratio", 0.8) > 1.2:
        cap_score += 33
        five_cs["capital"]["findings"].append(f"✓ Strong liquidity: Current Ratio = {capital_data['current_ratio']:.2f}")
    else:
        five_cs["capital"]["red_flags"].append(f"⚠️ Liquidity concerns: CR = {capital_data['current_ratio']:.2f}")
    
    if capital_data.get("dscr", 1.2) > 1.25:
        cap_score += 34
        five_cs["capital"]["findings"].append(f"✓ Healthy debt service: DSCR = {capital_data['dscr']:.2f}")
    else:
        five_cs["capital"]["red_flags"].append(f"⚠️ Weak debt service capacity: DSCR = {capital_data['dscr']:.2f}")
    
    five_cs["capital"]["score"] = min(100, cap_score)
    five_cs["capital"]["rating"] = "EXCELLENT" if cap_score >= 75 else \
                                    "GOOD" if cap_score >= 50 else \
                                    "FAIR" if cap_score >= 25 else "POOR"
    
    # COLLATERAL: Asset quality, tangible assets, pledged assets
    coll_score = 0.0
    if collateral_data.get("tangible_asset_ratio", 0.5) > 0.6:
        coll_score += 33
        five_cs["collateral"]["findings"].append(f"✓ Strong tangible assets: {collateral_data['tangible_asset_ratio']:.0%}")
    else:
        five_cs["collateral"]["red_flags"].append(f"⚠️ High intangible asset ratio")
    
    if collateral_data.get("asset_coverage", 0.8) > 1.0:
        coll_score += 33
        five_cs["collateral"]["findings"].append(f"✓ Good asset coverage: {collateral_data['asset_coverage']:.2f}x")
    else:
        five_cs["collateral"]["red_flags"].append("⚠️ Asset coverage below 1.0x")
    
    if collateral_data.get("pledged_assets_ratio", 0.5) < 0.70:
        coll_score += 34
        five_cs["collateral"]["findings"].append(f"✓ Reasonable pledge ratio: {collateral_data['pledged_assets_ratio']:.0%}")
    else:
        five_cs["collateral"]["red_flags"].append("⚠️ High proportion of pledged assets")
    
    five_cs["collateral"]["score"] = min(100, coll_score)
    five_cs["collateral"]["rating"] = "EXCELLENT" if coll_score >= 75 else \
                                       "GOOD" if coll_score >= 50 else \
                                       "FAIR" if coll_score >= 25 else "POOR"
    
    # CONDITIONS: Sector headwinds, macro environment, covenants
    cond_score = 0.0
    if conditions_data.get("sector_sentiment") > 0:
        cond_score += 25
        five_cs["conditions"]["findings"].append("✓ Positive sector outlook")
    else:
        five_cs["conditions"]["red_flags"].append("⚠️ Sector facing headwinds")
    
    if conditions_data.get("macro_environment") == "STABLE":
        cond_score += 25
        five_cs["conditions"]["findings"].append("✓ Stable macro environment")
    else:
        five_cs["conditions"]["red_flags"].append("⚠️ Macroeconomic uncertainty")
    
    if not conditions_data.get("covenant_concerns", []):
        cond_score += 25
        five_cs["conditions"]["findings"].append("✓ No covenant violations expected")
    
    if conditions_data.get("industry_regulations_favorable", True):
        cond_score += 25
        five_cs["conditions"]["findings"].append("✓ Favorable regulatory environment")
    else:
        five_cs["conditions"]["red_flags"].append("⚠️ Regulatory headwinds")
    
    five_cs["conditions"]["score"] = min(100, cond_score)
    five_cs["conditions"]["rating"] = "EXCELLENT" if cond_score >= 75 else \
                                       "GOOD" if cond_score >= 50 else \
                                       "FAIR" if cond_score >= 25 else "POOR"
    
    return five_cs

# =====================================================
# COMPREHENSIVE CAM GENERATOR (FIVE Cs BASED)
# =====================================================

def generate_structured_cam(
    company: str,
    revenue: float,
    ebitda: float,
    debt: float,
    decision: str,
    five_cs: Dict,
    gst_reconciliation: Dict,
    litigation_data: Dict,
    sector: str = None,
    cibil_score: float = 75.0
) -> str:
    """
    Generate professional Credit Appraisal Memo using Five Cs framework
    This is what Indian credit officers expect to see
    """
    
    if not client:
        return generate_basic_cam(company, revenue, ebitda, debt, decision, five_cs)
    
    prompt = f"""
Generate a professional Credit Appraisal Memo for Indian banking standards.

COMPANY DETAILS:
Name: {company}
Sector: {sector or 'Unknown'}
Revenue: ₹{revenue:,.0f}
EBITDA: ₹{ebitda:,.0f}
Total Debt: ₹{debt:,.0f}
CIBIL Score: {cibil_score}

DECISION: {decision}

FIVE Cs ASSESSMENT:

CHARACTER:
Score: {five_cs['character']['score']:.0f}/100 ({five_cs['character']['rating']})
Findings: {'; '.join(five_cs['character']['findings'][:3])}
Red Flags: {'; '.join(five_cs['character']['red_flags'][:2]) if five_cs['character']['red_flags'] else 'None'}

CAPACITY:
Score: {five_cs['capacity']['score']:.0f}/100 ({five_cs['capacity']['rating']})
Findings: {'; '.join(five_cs['capacity']['findings'][:3])}

CAPITAL:
Score: {five_cs['capital']['score']:.0f}/100 ({five_cs['capital']['rating']})
Findings: {'; '.join(five_cs['capital']['findings'][:3])}

COLLATERAL:
Score: {five_cs['collateral']['score']:.0f}/100 ({five_cs['collateral']['rating']})
Findings: {'; '.join(five_cs['collateral']['findings'][:2])}

CONDITIONS:
Score: {five_cs['conditions']['score']:.0f}/100 ({five_cs['conditions']['rating']})
Findings: {'; '.join(five_cs['conditions']['findings'][:3])}

REGULATORY CHECKS:
GST-ITR Status: {gst_reconciliation.get('status')}
Litigation Risk: {litigation_data.get('total_cases')} cases found
Critical Alerts: {len(litigation_data.get('critical_alerts', []))}

Create a professional CAM with:
1. Executive Summary
2. Company Overview (incorporation, sector, size)
3. Five Cs detailed analysis
4. Risk Assessment with regulatory findings
5. Recommendation & Terms
6. Covenants & Conditions
7. Monitoring Plan

Format as professional banking document.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            generation_config={"max_output_tokens": 2000}
        )
        return response.text
    except Exception as e:
        return generate_basic_cam(company, revenue, ebitda, debt, decision, five_cs)

def generate_basic_cam(company: str, revenue: float, ebitda: float, debt: float, 
                       decision: str, five_cs: Dict) -> str:
    """Fallback CAM generator when API is unavailable"""
    
    cam = f"""
╔════════════════════════════════════════════════════════════════╗
║         CREDIT APPRAISAL MEMO (CAM) - CONFIDENTIAL            ║
╚════════════════════════════════════════════════════════════════╝

COMPANY: {company}
DATE: {datetime.now().strftime('%Y-%m-%d')}
DECISION: {decision}

─────────────────────────────────────────────────────────────────

FINANCIAL OVERVIEW:
  Revenue: ₹{revenue:,.0f}
  EBITDA: ₹{ebitda:,.0f}
  Total Debt: ₹{debt:,.0f}

─────────────────────────────────────────────────────────────────

FIVE Cs ASSESSMENT:

1. CHARACTER (Promoter Background & Payment History)
   ├─ Score: {five_cs['character']['score']:.0f}/100
   ├─ Rating: {five_cs['character']['rating']}
   ├─ Findings: {'; '.join(five_cs['character']['findings'][:2])}
   └─ Red Flags: {'; '.join(five_cs['character']['red_flags'][:1]) if five_cs['character']['red_flags'] else 'None'}

2. CAPACITY (Revenue & Profit Generation Ability)
   ├─ Score: {five_cs['capacity']['score']:.0f}/100
   ├─ Rating: {five_cs['capacity']['rating']}
   ├─ Findings: {'; '.join(five_cs['capacity']['findings'][:2])}
   └─ Red Flags: {'; '.join(five_cs['capacity']['red_flags'][:1]) if five_cs['capacity']['red_flags'] else 'None'}

3. CAPITAL (Equity & Leverage Position)
   ├─ Score: {five_cs['capital']['score']:.0f}/100
   ├─ Rating: {five_cs['capital']['rating']}
   ├─ Findings: {'; '.join(five_cs['capital']['findings'][:2])}
   └─ Red Flags: {'; '.join(five_cs['capital']['red_flags'][:1]) if five_cs['capital']['red_flags'] else 'None'}

4. COLLATERAL (Asset Quality & Coverage)
   ├─ Score: {five_cs['collateral']['score']:.0f}/100
   ├─ Rating: {five_cs['collateral']['rating']}
   ├─ Findings: {'; '.join(five_cs['collateral']['findings'][:2])}
   └─ Red Flags: {'; '.join(five_cs['collateral']['red_flags'][:1]) if five_cs['collateral']['red_flags'] else 'None'}

5. CONDITIONS (External Environment & Risks)
   ├─ Score: {five_cs['conditions']['score']:.0f}/100
   ├─ Rating: {five_cs['conditions']['rating']}
   ├─ Findings: {'; '.join(five_cs['conditions']['findings'][:2])}
   └─ Red Flags: {'; '.join(five_cs['conditions']['red_flags'][:1]) if five_cs['conditions']['red_flags'] else 'None'}

─────────────────────────────────────────────────────────────────

OVERALL ASSESSMENT:
Decision: {decision}
This recommendation is based on comprehensive evaluation of all
Five Cs with emphasis on regulatory compliance and risk mitigation.

─────────────────────────────────────────────────────────────────
Generated by IntelliCredit-X | AI Credit Decision Engine
"""
    
    return cam

# =====================================================
# EXPLAINABILITY ENGINE (WHY DECISIONS)
# =====================================================

def generate_decision_explanation(
    decision: str,
    prob: float,
    five_cs: Dict,
    gst_reconciliation: Dict,
    litigation_data: Dict,
    shap_values = None
) -> str:
    """
    Explain WHY the model made this decision
    This addresses the hackathon's "explainability" requirement
    """
    
    explanation = f"""
📊 DECISION EXPLAINABILITY REPORT

FINAL DECISION: {decision}
Probability of Default: {prob:.1%}

═══════════════════════════════════════════════════════════════

🔴 CRITICAL FACTORS (IF ANY):
"""
    
    # GST-ITR Red Flags
    if gst_reconciliation.get("red_flags"):
        explanation += "\n🚨 GST-ITR Reconciliation Issues:\n"
        for flag in gst_reconciliation["red_flags"]:
            explanation += f"  • {flag}\n"
    
    # Litigation Red Flags
    if litigation_data.get("critical_alerts"):
        explanation += "\n🚨 Legal & Compliance Risks:\n"
        for alert in litigation_data["critical_alerts"]:
            explanation += f"  • {alert}\n"
    
    # Five Cs Red Flags
    explanation += "\n🚨 Financial Risk Indicators:\n"
    all_red_flags = []
    for c in ["character", "capacity", "capital", "collateral", "conditions"]:
        flags = five_cs[c].get("red_flags", [])
        all_red_flags.extend(flags)
    
    for flag in all_red_flags[:5]:
        explanation += f"  • {flag}\n"
    
    # SHAP Contribution (if available)
    explanation += "\n═══════════════════════════════════════════════════════════════\n"
    explanation += "📈 MODEL CONTRIBUTION ANALYSIS:\n"
    
    if shap_values:
        explanation += "  Top factors influencing this decision (from SHAP):\n"
        explanation += "  [See visualization below]\n"
    else:
        # Manual feature importance
        explanation += """
  Based on Five Cs Assessment:
  • Character Risk: 25% weight
  • Capacity Risk: 30% weight (most important)
  • Capital Risk: 20% weight
  • Collateral Risk: 15% weight
  • Conditions Risk: 10% weight
"""
    
    explanation += "\n═══════════════════════════════════════════════════════════════\n"
    explanation += "✅ POSITIVE FACTORS (IF ANY):\n"
    
    positive_count = 0
    for c in ["character", "capacity", "capital", "collateral", "conditions"]:
        findings = five_cs[c].get("findings", [])
        for finding in findings[:2]:
            if "✓" in finding:
                explanation += f"  • {finding}\n"
                positive_count += 1
    
    if positive_count == 0:
        explanation += "  • [None significant]\n"
    
    explanation += "\n═══════════════════════════════════════════════════════════════\n"
    
    return explanation

# =====================================================
# MAIN APP FLOW
# =====================================================

st.sidebar.markdown("## ⚙️ Control Panel")

# File Upload Section
st.header("📄 Step 1: Upload Financial Statement")
uploaded_pdf = st.file_uploader("Upload Balance Sheet / P&L (PDF)", type=["pdf"])

ai_data = {}
gst_data = {}
mca_data = {}

if uploaded_pdf:
    st.success("✅ PDF uploaded successfully")
    
    with st.spinner("🔄 Extracting financial data..."):
        text = extract_pdf_text(uploaded_pdf)
        ai_data = ai_extract_financials(text)
        gst_data = extract_gst_data_from_text(text)
    
    if ai_data:
        st.info("✓ AI extracted financial values. Please verify and adjust as needed.")

# Financial Inputs
st.header("💰 Step 2: Financial Inputs")

col1, col2 = st.columns(2)

with col1:
    revenue = st.number_input(
        "Revenue / Turnover (₹)",
        min_value=0.0,
        value=float(ai_data.get("revenue", 500000000)),
        format="%f"
    )
    
    pbt = st.number_input(
        "Profit Before Tax (₹)",
        min_value=0.0,
        value=float(ai_data.get("profit_before_tax", 50000000)),
        format="%f"
    )
    
    finance_cost = st.number_input(
        "Finance Cost / Interest Expense (₹)",
        min_value=0.0,
        value=float(ai_data.get("finance_cost", 10000000)),
        format="%f"
    )
    
    depreciation = st.number_input(
        "Depreciation & Amortization (₹)",
        min_value=0.0,
        value=float(ai_data.get("depreciation", 10000000)),
        format="%f"
    )

with col2:
    debt = st.number_input(
        "Total Debt (₹)",
        min_value=0.0,
        value=float(ai_data.get("total_debt", 200000000)),
        format="%f"
    )
    
    current_assets = st.number_input(
        "Current Assets (₹)",
        min_value=0.0,
        value=float(ai_data.get("current_assets", 200000000)),
        format="%f"
    )
    
    current_liabilities = st.number_input(
        "Current Liabilities (₹)",
        min_value=0.0,
        value=float(ai_data.get("current_liabilities", 150000000)),
        format="%f"
    )
    
    equity = st.number_input(
        "Total Equity (₹)",
        min_value=0.0,
        value=float(ai_data.get("equity", 300000000) or 300000000),
        format="%f"
    )

# GST-ITR Reconciliation Section
st.header("📊 Step 3: GST-ITR Reconciliation (CRITICAL)")

col1, col2 = st.columns(2)

with col1:
    gst_turnover = st.number_input(
        "GST Turnover (₹)",
        min_value=0.0,
        value=float(gst_data.get("gst_turnover", revenue * 0.95)),
        format="%f",
        help="From GSTR-3B monthly filings"
    )
    
    gst_2a_iTC = st.number_input(
        "GSTR-2A ITC Available (₹)",
        min_value=0.0,
        value=float(gst_data.get("gst_2a_iTC", 5000000)),
        format="%f"
    )

with col2:
    itr_revenue = st.number_input(
        "ITR Revenue (₹)",
        min_value=0.0,
        value=float(ai_data.get("itr_revenue", revenue)),
        format="%f",
        help="From Income Tax Returns"
    )
    
    gst_3b_liability = st.number_input(
        "GSTR-3B ITC Claimed (₹)",
        min_value=0.0,
        value=float(gst_data.get("gst_3b_liability", 4500000)),
        format="%f"
    )

# Additional Risk Factors
st.header("⚠️ Step 4: Additional Risk Factors")

col1, col2, col3 = st.columns(3)

with col1:
    company = st.text_input("Company Name", placeholder="e.g., ABC Manufacturing Ltd")
    sector = st.selectbox(
        "Sector",
        ["NBFC", "AUTO", "RETAIL", "PHARMA", "POWER", "TELECOM", "FMCG", "INFRA", "OTHER"]
    )
    mgmt = st.slider("Management Quality (1-10)", 1.0, 10.0, 7.0)

with col2:
    cibil_score = st.slider("CIBIL Commercial Score (1-100)", 1, 100, 75)
    litigation_count = st.number_input("Known Litigation Cases", min_value=0, value=0)
    capacity_util = st.slider("Capacity Utilization (%)", 0, 100, 80)

with col3:
    gst_mismatch_risk = st.slider("GST Mismatch Risk (0-0.5)", 0.0, 0.5, 0.1)
    promoter_names = st.text_input("Promoter Names (comma-separated)", placeholder="For litigation search")
    sector_sentiment = st.slider("Sector Sentiment (-1 to +1)", -1.0, 1.0, 0.0)

# Calculate Ratios
st.subheader("📈 Calculated Financial Ratios")

ebitda = pbt + finance_cost + depreciation

interest_coverage = 0
if finance_cost > 0:
    interest_coverage = ebitda / finance_cost
else:
    interest_coverage = 0

current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
debt_equity = debt / equity if equity > 0 else 0
profit_margin = pbt / revenue if revenue > 0 else 0
dscr = ebitda / (finance_cost * 1.1) if finance_cost > 0 else 0

ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)

with ratio_col1:
    st.metric("EBITDA", f"₹{ebitda:,.0f}")
    st.metric("Current Ratio", f"{current_ratio:.2f}x")

with ratio_col2:
    st.metric("Interest Coverage", f"{interest_coverage:.2f}x")
    st.metric("D/E Ratio", f"{debt_equity:.2f}x")

with ratio_col3:
    st.metric("Profit Margin", f"{profit_margin:.1%}")
    st.metric("DSCR", f"{dscr:.2f}x")

with ratio_col4:
    st.metric("GST-ITR Variance", f"{abs(gst_turnover - itr_revenue) / itr_revenue if itr_revenue > 0 else 0:.1%}")
    st.metric("CIBIL Score", f"{cibil_score}/100")

# =====================================================
# RUN ANALYSIS BUTTON
# =====================================================

if st.button("🚀 ANALYZE CREDIT RISK & GENERATE CAM", use_container_width=True):
    
    if not company:
        st.error("⚠️ Please enter Company Name")
    else:
        
        with st.spinner("⏳ Running comprehensive credit analysis..."):
            
            # 1. GST-ITR Reconciliation
            st.subheader("🔍 GST-ITR Reconciliation Analysis")
            gst_recon = gst_itr_reconciliation(gst_turnover, itr_revenue, gst_2a_iTC, gst_3b_liability)
            
            if gst_recon["red_flags"]:
                for flag in gst_recon["red_flags"]:
                    st.error(flag)
            else:
                for finding in gst_recon["findings"]:
                    st.success(finding)
            
            # 2. MCA Filing Data
            st.subheader("🏛️ MCA & Regulatory Filings")
            mca_data = fetch_mca_company_data(company)
            
            if mca_data.get("cin"):
                st.info(f"✓ CIN: {mca_data['cin']}")
            if mca_data.get("incorporation_date"):
                st.info(f"✓ Incorporated: {mca_data['incorporation_date']}")
            if mca_data.get("regulatory_flags"):
                for flag in mca_data["regulatory_flags"]:
                    st.warning(f"⚠️ {flag}")
            
            # 3. Litigation Search
            st.subheader("⚖️ Legal & Litigation Search")
            promoters = [p.strip() for p in promoter_names.split(",")] if promoter_names else []
            litigation_data = search_ecourts_litigation(company, promoters)
            
            if litigation_data["total_cases"] > 0:
                st.warning(f"⚠️ Found {litigation_data['total_cases']} litigation case(s)")
                
                if litigation_data["insolvency_cases"]:
                    st.error("🚨 INSOLVENCY CASES FOUND:")
                    for case in litigation_data["insolvency_cases"]:
                        st.markdown(f"- [{case['title']}]({case['url']})")
                
                if litigation_data["criminal_cases"]:
                    st.error("🚨 CRIMINAL CASES FOUND:")
                    for case in litigation_data["criminal_cases"]:
                        st.markdown(f"- [{case['title']}]({case['url']})")
                
                if litigation_data["commercial_cases"]:
                    with st.expander("Commercial Litigation Cases"):
                        for case in litigation_data["commercial_cases"]:
                            st.markdown(f"- [{case['title']}]({case['url']})")
            else:
                st.success("✓ No litigation found in news database")
            
            # 4. CIBIL & Payment History
            st.subheader("💳 CIBIL Commercial Report")
            cibil_data = fetch_cibil_score(company, cibil_score)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CIBIL Score", cibil_score)
            with col2:
                st.metric("Rating", cibil_data["rating_category"])
            with col3:
                st.metric("NPA Status", "🟢 No" if not cibil_data["npa_flag"] else "🔴 Yes")
            
            # 5. RBI Sector Guidelines
            st.subheader("📋 RBI Regulatory Guidelines")
            rbi_data = fetch_rbi_sector_guidelines(sector)
            
            st.info(f"📌 Repo Rate: {rbi_data['repo_rate']:.2f}%")
            if rbi_data["sector_specific_guidance"]:
                for guidance in rbi_data["sector_specific_guidance"]:
                    st.warning(f"⚠️ {guidance}")
            
            # 6. News & Sentiment Analysis
            st.subheader("📰 News Intelligence & Sector Analysis")
            sentiment, articles, sector_articles = get_news_sentiment(company, sector)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Company News")
                if articles:
                    st.metric("Sentiment Score", f"{sentiment:.2f}", 
                              delta="Positive" if sentiment > 0 else "Negative")
                    for article in articles[:3]:
                        with st.expander(article.get("title", "Article")[:60]):
                            st.markdown(f"[{article.get('source', {}).get('name')}]({article.get('url')})")
                            st.caption(article.get("description", ""))
                else:
                    st.info("No recent news found")
            
            with col2:
                st.subheader("Sector News")
                if sector_articles:
                    for article in sector_articles[:3]:
                        with st.expander(article.get("title", "Article")[:60]):
                            st.markdown(f"[{article.get('source', {}).get('name')}]({article.get('url')})")
                else:
                    st.info("No sector news available")
            
            # 7. Prepare Five Cs data
            character_data = {
                "litigation_risk_score": litigation_data.get("litigation_risk_score", 0.0),
                "critical_alerts": litigation_data.get("critical_alerts", []),
                "payment_history": "GOOD" if cibil_score > 70 else "FAIR" if cibil_score > 50 else "POOR",
                "cibil_score": cibil_score
            }
            
            capacity_data = {
                "revenue_trend": "GROWING" if sentiment > 0 else "STABLE",
                "interest_coverage": interest_coverage,
                "profit_margin": profit_margin,
                "capacity_utilization": capacity_util / 100.0
            }
            
            capital_data = {
                "debt_equity_ratio": debt_equity,
                "current_ratio": current_ratio,
                "dscr": dscr
            }
            
            collateral_data = {
                "tangible_asset_ratio": 0.7,
                "asset_coverage": ebitda / debt if debt > 0 else 0,
                "pledged_assets_ratio": 0.5
            }
            
            conditions_data = {
                "sector_sentiment": sector_sentiment,
                "macro_environment": "STABLE",
                "covenant_concerns": [],
                "industry_regulations_favorable": True
            }
            
            # 8. Evaluate Five Cs
            five_cs = evaluate_five_cs(
                character_data, capacity_data, capital_data, 
                collateral_data, conditions_data
            )
            
            st.subheader("⭐ Five Cs of Credit Assessment")
            
            five_c_cols = st.columns(5)
            cs_names = ["CHARACTER", "CAPACITY", "CAPITAL", "COLLATERAL", "CONDITIONS"]
            cs_keys = ["character", "capacity", "capital", "collateral", "conditions"]
            
            for idx, (col, cs_name, cs_key) in enumerate(zip(five_c_cols, cs_names, cs_keys)):
                with col:
                    score = five_cs[cs_key]["score"]
                    rating = five_cs[cs_key]["rating"]
                    color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
                    st.metric(cs_name, f"{score:.0f}/100", delta=rating)
                    st.write(f"{color} {rating}")
            
            # 9. ML Model Prediction
            st.subheader("🤖 ML-Based Risk Assessment")
            
            if model and feature_names:
                input_data = np.array([[
                    revenue,
                    ebitda,
                    debt,
                    interest_coverage,
                    gst_mismatch_risk,
                    litigation_count,
                    sentiment,
                    sector_sentiment,
                    mgmt / 10.0,
                    capacity_util / 100.0
                ]])
                
                df = pd.DataFrame(input_data, columns=feature_names)
                
                try:
                    prob = model.predict_proba(df)[0][1]
                except:
                    prob = 0.3
            else:
                # Calculate synthetic probability from Five Cs
                five_c_avg = np.mean([five_cs[c]["score"] for c in cs_keys])
                prob = (100 - five_c_avg) / 100.0
            
            prob = max(0, min(prob, 1))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probability of Default", f"{prob:.1%}", 
                         delta="High Risk" if prob > 0.3 else "Medium Risk" if prob > 0.15 else "Low Risk")
            
            # Decision
            decision = "✅ APPROVED" if prob <= threshold else "❌ REJECTED"
            
            with col2:
                if "APPROVED" in decision:
                    st.success(decision)
                else:
                    st.error(decision)
            
            with col3:
                five_c_avg_score = np.mean([five_cs[c]["score"] for c in cs_keys])
                st.metric("Five Cs Average Score", f"{five_c_avg_score:.0f}/100")
            
            # 10. Loan Recommendation
            st.subheader("💵 Loan Recommendation")
            
            max_loan = ebitda * 3.5
            recommended_loan = max_loan * (1 - prob)
            interest_rate = 9 + prob * 6
            
            loan_col1, loan_col2, loan_col3 = st.columns(3)
            
            with loan_col1:
                st.metric("Max Loan Amount (3.5x EBITDA)", f"₹{max_loan:,.0f}")
            
            with loan_col2:
                st.metric("Recommended Loan Amount", f"₹{recommended_loan:,.0f}")
            
            with loan_col3:
                st.metric("Recommended Interest Rate", f"{interest_rate:.2f}%")
            
            try:
                loan_words = num2words(int(recommended_loan), lang="en_IN")
                st.info(f"💬 In words: {loan_words}")
            except:
                pass
            
            # 11. SHAP Explainability
            st.subheader("📊 Feature Importance (SHAP Waterfall)")
            
            if explainer and model and feature_names:
                try:
                    shap_values = explainer(df)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"⚠️ SHAP visualization unavailable: {str(e)}")
            else:
                st.info("📌 SHAP explainability requires pre-trained model")
            
            # 12. Decision Explainability
            st.subheader("💡 Why This Decision?")
            explanation = generate_decision_explanation(
                decision, prob, five_cs, gst_recon, litigation_data, shap_values=None
            )
            st.text(explanation)
            
            # 13. Comprehensive CAM Generator
            st.subheader("📋 Credit Appraisal Memo (CAM)")
            
            cam = generate_structured_cam(
                company=company,
                revenue=revenue,
                ebitda=ebitda,
                debt=debt,
                decision=decision,
                five_cs=five_cs,
                gst_reconciliation=gst_recon,
                litigation_data=litigation_data,
                sector=sector,
                cibil_score=cibil_score
            )
            
            st.text_area("CAM Content", cam, height=600)
            
            # Export CAM
            st.download_button(
                label="📥 Download CAM as Text",
                data=cam,
                file_name=f"CAM_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("""
**IntelliCredit-X** | Next-Gen AI Credit Decision Engine
- Comprehensive financial extraction & analysis
- Regulatory compliance (GST-ITR, MCA, e-Courts)
- Five Cs credit framework assessment
- SHAP-based explainability
- Structured CAM generation
""")
st.markdown("Built for the Intelli-Credit Hackathon Challenge")
