import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import json

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

# SIMPLE GST-ITR CHECK
def gst_itr_check(gst_turnover, itr_revenue):
    if not gst_turnover or not itr_revenue:
        return "PASS", "LOW", []
    
    variance = abs(gst_turnover - itr_revenue) / itr_revenue
    
    if variance > 0.20:
        return "FAIL", "CRITICAL", [f"🚨 Critical GST-ITR mismatch: {variance:.1%}"]
    elif variance > 0.10:
        return "FAIL", "HIGH", [f"⚠️ High variance: {variance:.1%}"]
    elif variance > 0.05:
        return "PASS", "MEDIUM", [f"Minor variance: {variance:.1%}"]
    else:
        return "PASS", "LOW", ["✓ GST-ITR aligned"]

# MAIN UI
st.header("💰 Financial Inputs")

col1, col2 = st.columns(2)

with col1:
    company = st.text_input("Company Name")
    revenue = st.number_input("Revenue (₹)", min_value=0.0, value=500000000.0)
    pbt = st.number_input("Profit Before Tax (₹)", min_value=0.0, value=50000000.0)
    finance_cost = st.number_input("Finance Cost (₹)", min_value=0.0, value=10000000.0)
    depreciation = st.number_input("Depreciation (₹)", min_value=0.0, value=10000000.0)

with col2:
    debt = st.number_input("Total Debt (₹)", min_value=0.0, value=200000000.0)
    current_assets = st.number_input("Current Assets (₹)", min_value=0.0, value=200000000.0)
    current_liab = st.number_input("Current Liabilities (₹)", min_value=0.0, value=150000000.0)
    equity = st.number_input("Total Equity (₹)", min_value=0.0, value=300000000.0)

# CALCULATE RATIOS
ebitda = pbt + finance_cost + depreciation
ic = ebitda / finance_cost if finance_cost > 0 else 0
cr = current_assets / current_liab if current_liab > 0 else 0
de = debt / equity if equity > 0 else 0
pm = pbt / revenue if revenue > 0 else 0

st.subheader("📈 Ratios")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("EBITDA", f"₹{ebitda:,.0f}")
with c2:
    st.metric("Interest Coverage", f"{ic:.2f}x")
with c3:
    st.metric("Current Ratio", f"{cr:.2f}x")
with c4:
    st.metric("D/E Ratio", f"{de:.2f}x")

# GST-ITR
st.header("📊 GST-ITR")
col1, col2 = st.columns(2)

with col1:
    gst_turnover = st.number_input("GST Turnover (₹)", min_value=0.0, value=revenue * 0.95)

with col2:
    itr_revenue = st.number_input("ITR Revenue (₹)", min_value=0.0, value=revenue)

# RISK FACTORS
st.header("⚠️ Risk Factors")
cibil_score = st.slider("CIBIL Score", 1, 100, 75)
litigation = st.number_input("Litigation Cases", min_value=0, value=0)
sector_sentiment = st.slider("Sector Sentiment (-1 to +1)", -1.0, 1.0, 0.0)

# ANALYSIS
if st.button("🚀 ANALYZE CREDIT RISK", use_container_width=True):
    
    if not company:
        st.error("Enter company name")
    else:
        st.subheader("📊 Results")
        
        # GST-ITR
        st.subheader("GST-ITR Check")
        status, risk, flags = gst_itr_check(gst_turnover, itr_revenue)
        
        for flag in flags:
            if "Critical" in flag or "High" in flag:
                st.error(flag)
            else:
                st.success(flag)
        
        # Litigation
        st.subheader("Litigation Risk")
        if litigation > 0:
            st.warning(f"⚠️ {litigation} case(s)")
        else:
            st.success("✓ No litigation")
        
        # CIBIL
        st.subheader("CIBIL Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", cibil_score)
        with col2:
            rating = "EXCELLENT" if cibil_score > 90 else "GOOD" if cibil_score > 70 else "FAIR"
            st.metric("Rating", rating)
        
        # Calculate PD
        base_pd = (100 - cibil_score) / 100.0
        litigation_risk = litigation * 0.05
        gst_var = abs(gst_turnover - itr_revenue) / itr_revenue if itr_revenue > 0 else 0
        gst_risk = 0.15 if gst_var > 0.20 else 0.08 if gst_var > 0.10 else 0
        
        prob = min(1.0, max(0.0, base_pd + litigation_risk + gst_risk))
        
        # Decision
        st.subheader("🤖 Decision")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probability of Default", f"{prob:.1%}")
        
        decision = "✅ APPROVED" if prob <= 0.5 else "❌ REJECTED"
        
        with col2:
            if "APPROVED" in decision:
                st.success(decision)
            else:
                st.error(decision)
        
        with col3:
            risk_level = "LOW" if prob < 0.2 else "MEDIUM" if prob < 0.4 else "HIGH"
            st.metric("Risk Level", risk_level)
        
        # Loan
        st.subheader("💵 Loan Terms")
        max_loan = ebitda * 3.5
        recommended = max_loan * (1 - prob)
        rate = 9 + prob * 6
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Loan", f"₹{max_loan:,.0f}")
        with col2:
            st.metric("Recommended", f"₹{recommended:,.0f}")
        with col3:
            st.metric("Interest Rate", f"{rate:.2f}%")
        
        # CAM
        st.subheader("📋 Credit Appraisal Memo")
        cam = f"""
CREDIT APPRAISAL MEMO

Company: {company}
Date: {datetime.now().strftime('%Y-%m-%d')}

FINANCIAL SUMMARY:
Revenue: ₹{revenue:,.0f}
EBITDA: ₹{ebitda:,.0f}
Total Debt: ₹{debt:,.0f}

RATIOS:
Interest Coverage: {ic:.2f}x
Current Ratio: {cr:.2f}x
D/E Ratio: {de:.2f}x
Profit Margin: {pm:.1%}

RISK ASSESSMENT:
CIBIL Score: {cibil_score}
Litigation Cases: {litigation}
GST-ITR Variance: {gst_var:.1%}
Probability of Default: {prob:.1%}

RECOMMENDATION:
Decision: {decision}
Recommended Loan: ₹{recommended:,.0f}
Interest Rate: {rate:.2f}%

This appraisal is based on comprehensive financial analysis and
risk assessment per banking standards.
"""
        
        st.text_area("CAM", cam, height=400)
        
        st.download_button(
            label="📥 Download CAM",
            data=cam,
            file_name=f"CAM_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown("**IntelliCredit-X** | AI Credit Decision Engine")
