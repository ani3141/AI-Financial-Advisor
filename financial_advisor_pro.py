import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import requests
from io import StringIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import time
import json
import os
import tensorflow as tf
def save_login_details(name, phone, occupation):
    """Save login details to a CSV file."""
    file_path = "data/login_details.csv"
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    # Prepare the row as a DataFrame
    row = pd.DataFrame([{
        "Name": name,
        "Phone": phone,
        "Occupation": occupation
    }])
    # Append to CSV (create if not exists)
    if os.path.exists(file_path):
        row.to_csv(file_path, mode='a', header=False, index=False)
    else:
        row.to_csv(file_path, mode='w', header=True, index=False)
# Page configuration
st.set_page_config(
    page_title="Financial Advisor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

if 'completed_topics' not in st.session_state:
    st.session_state.completed_topics = set()

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'onboarding'

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        pass

download_nltk_resources()

# Constants
INCOME_CATEGORIES = [
    "0-5 Lakhs",
    "5-10 Lakhs",
    "10-15 Lakhs",
    "15-20 Lakhs",
    "20-25 Lakhs",
    "25-30 Lakhs",
    "Above 30 Lakhs"
]

INDIAN_SECTORS = {
    "Banking & Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Pharma & Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Automobile": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS"],
    "Energy & Oil": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "NTPC.NS", "POWERGRID.NS"],
    "Consumer Goods": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS"],
    "Technology": ["TECHM.NS", "MINDTREE.NS", "LTI.NS", "MPHASIS.NS", "PERSISTENT.NS"]
}

INDIAN_STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Oil & Natural Gas Corp": "ONGC.NS",
    "Indian Oil Corporation": "IOC.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Dabur India": "DABUR.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Vedanta": "VEDL.NS",
    "Coal India": "COALINDIA.NS"
}

FINANCIAL_BASICS = {
    "Investment Fundamentals": {
        "What is Investment?": """
        Investment is the act of allocating resources (usually money) with the expectation of generating income or profit.
        Key points:
        - Long-term wealth creation
        - Power of compounding
        - Risk vs. Return trade-off
        - Importance of diversification
        """,
        "Power of Compound Interest": """
        Compound Interest is interest earned on both the principal and accumulated interest.
        
        Key Concepts:
        1. Rule of 72: Divide 72 by interest rate to find years needed to double money
        2. Impact of Time: Earlier you start, more wealth you create
        3. Frequency Matters: More frequent compounding = higher returns
        
        Example:
        ₹10,000 invested at 10% p.a.
        - Simple Interest (5 years): ₹15,000
        - Compound Interest (5 years): ₹16,105
        - Compound Interest (10 years): ₹25,937
        
        This demonstrates how compound interest creates exponential growth over time.
        """,
        "Types of Investments": """
        Different investment options available in India:
        1. Stocks (Equity)
        2. Bonds
        3. Mutual Funds
        4. Fixed Deposits
        5. Public Provident Fund (PPF)
        6. Real Estate
        7. Gold
        Each has its own risk-return profile and investment horizon.
        """
    },
    "Stock Market Basics": {
        "Understanding Stocks": """
        A stock represents ownership in a company.
        Key concepts:
        - Share prices and market capitalization
        - Stock exchanges (NSE/BSE)
        - Trading and settlement
        - Corporate actions (dividends, splits)
        """,
        "Stock Analysis": """
        Methods to analyze stocks:
        1. Fundamental Analysis
           - Company financials
           - Industry analysis
           - Economic factors
        2. Technical Analysis
           - Price patterns
           - Trading volumes
           - Technical indicators
        """,
        "Market Indices": """
        Understanding market indices:
        - NIFTY 50
        - SENSEX
        - Sector indices
        These help track market performance and sentiment.
        """
    },
    "Risk Management": {
        "Understanding Risk": """
        Risk in financial markets:
        - Market risk
        - Credit risk
        - Liquidity risk
        - Operational risk
        Importance of risk assessment and management.
        """,
        "Risk Mitigation": """
        Strategies to manage risk:
        1. Diversification
        2. Asset allocation
        3. Stop-loss orders
        4. Regular portfolio review
        5. Emergency fund maintenance
        """
    },
    "Banking Basics": {
        "Types of Bank Accounts": """
        Understanding different bank accounts:

        1. Savings Account:
        - For personal savings
        - Earns interest
        - Limited transactions
        - Minimum balance requirement

        2. Current Account:
        - For businesses
        - No interest earned
        - Unlimited transactions
        - Higher minimum balance

        3. Debit Account:
        - Linked to savings/current account
        - Direct access to your money
        - Used for purchases and ATM withdrawals
        - Daily transaction limits apply

        4. Credit Account:
        - Borrowed money from bank
        - Need to repay with interest
        - Credit limit based on creditworthiness
        - Builds credit history

        5. Overdraft Account:
        - Allows withdrawing more than balance
        - Pre-approved credit limit
        - Higher interest rates
        - Usually for business accounts
        """,
        "Banking Services": """
        Essential banking services:
        1. NEFT/RTGS/IMPS Transfers
        2. Mobile Banking
        3. Internet Banking
        4. UPI Payments
        5. Cheque Services
        6. Lockers
        7. Insurance Products
        """,
        "Credit Management": """
        Managing credit effectively:
        1. Credit Score Importance
        2. Credit Card Usage
        3. Loan Management
        4. Debt Consolidation
        5. Credit Report Monitoring
        """
    },
    "Taxation and Compliance": {
        "Income Tax Basics": """
        Understanding Income Tax in India:

        Tax Slabs (FY 2023-24):
        Old Regime:
        - Up to ₹2.5L: No tax
        - ₹2.5L to ₹5L: 5%
        - ₹5L to ₹7.5L: 10%
        - ₹7.5L to ₹10L: 15%
        - ₹10L to ₹12.5L: 20%
        - ₹12.5L to ₹15L: 25%
        - Above ₹15L: 30%

        New Regime:
        - Up to ₹3L: No tax
        - ₹3L to ₹6L: 5%
        - ₹6L to ₹9L: 10%
        - ₹9L to ₹12L: 15%
        - ₹12L to ₹15L: 20%
        - Above ₹15L: 30%

        Additional:
        - Standard Deduction: ₹50,000
        - Section 80C Deductions: Up to ₹1.5L
        - Health Insurance (80D): Up to ₹25,000
        """,
        "TDS (Tax Deducted at Source)": """
        Understanding TDS:

        1. What is TDS?
        - Tax deducted by payer at source
        - Collected before paying income
        - Prevents tax evasion
        
        2. Common TDS Rates:
        - Salary: As per slab rates
        - Interest: 10%
        - Professional Fees: 10%
        - Rent: 10%
        - Commission: 5%

        3. TDS Certificates:
        - Form 16 for salary
        - Form 16A for other income
        - Form 26AS for tax credit
        """,
        "ITR Filing": """
        Income Tax Return Filing:

        1. Types of ITR:
        - ITR-1: For salaried individuals
        - ITR-2: For individuals with capital gains
        - ITR-3: For business income
        - ITR-4: For presumptive income
        
        2. Filing Process:
        - Collect documents (Form 16, 26AS)
        - Choose correct ITR form
        - Fill income details
        - Claim deductions
        - Verify return (e-verify/send ITR-V)

        3. Important Dates:
        - Regular filing: July 31
        - Audit cases: October 31
        - Belated filing: December 31
        """,
        "PAN Card": """
        Permanent Account Number (PAN):

        1. Uses:
        - Primary tax identification
        - Required for bank accounts
        - Mandatory for investments
        - High-value transactions
        - Credit card applications

        2. When Required:
        - Income above basic exemption
        - Business/Professional start
        - High-value transactions
        - Foreign travel
        - Property purchase/sale
        """,
        "GST and Other Taxes": """
        Goods and Services Tax:

        1. GST Rates:
        - 0% (Essential goods)
        - 5% (Basic necessities)
        - 12% (Standard goods)
        - 18% (Most goods)
        - 28% (Luxury items)

        2. GST Returns:
        - GSTR-1: Monthly outward supplies
        - GSTR-2B: Input tax credit
        - GSTR-3B: Monthly summary
        
        3. E-way Bill:
        - Required for goods movement
        - Value exceeding ₹50,000
        - Valid for specific duration
        - Generated through portal

        4. Sales Tax vs VAT vs GST:
        - Sales Tax: Old system, state-level
        - VAT: Value Added Tax, predecessor to GST
        - GST: One nation, one tax system
        """
    },
    "Financial Planning": {
        "Budgeting": """
        Creating and maintaining a budget:
        - Income tracking
        - Expense categorization
        - Savings goals
        - Debt management
        - 50-30-20 Rule (Needs-Wants-Savings)
        - Monthly expense tracking
        - Budget review and adjustment
        """,
        "Emergency Fund": """
        Building an emergency fund:
        - 6-12 months of expenses
        - Liquid investments
        - Regular contributions
        - Separate from investments
        - Where to keep emergency fund
        - When to use emergency fund
        - How to replenish used funds
        """,
        "Investment Planning": """
        Creating an investment plan:
        1. Goal setting
           - Short-term goals (1-3 years)
           - Medium-term goals (3-7 years)
           - Long-term goals (7+ years)
        
        2. Risk Assessment
           - Risk capacity
           - Risk tolerance
           - Risk requirement
        
        3. Asset Allocation
           - Equity allocation
           - Debt allocation
           - Gold and alternatives
        
        4. Investment Strategy
           - Lump sum vs SIP
           - Direct vs Regular plans
           - Active vs Passive funds
        
        5. Regular Review
           - Portfolio rebalancing
           - Goal tracking
           - Performance monitoring
        """
    },
    "Retirement Planning": {
        "Retirement Basics": """
        Understanding retirement planning:
        1. Why plan for retirement
        2. When to start planning
        3. Estimating retirement corpus
        4. Impact of inflation
        5. Life expectancy considerations
        """,
        "Retirement Investment Options": """
        Popular retirement investment options:
        1. Employee Provident Fund (EPF)
        2. Public Provident Fund (PPF)
        3. National Pension System (NPS)
        4. Mutual Fund Pension Plans
        5. Senior Citizens Savings Scheme
        6. Post Office Monthly Income Scheme
        """,
        "Post-Retirement Planning": """
        Managing finances post retirement:
        1. Regular Income Generation
        2. Healthcare Planning
        3. Estate Planning
        4. Tax Planning
        5. Insurance Needs
        """
    },
    "Financial Terms and Abbreviations": {
        "Common Financial Terms": """
        Essential Financial Terms:
        
        1. Banking Terms:
        - KYC: Know Your Customer
        - CASA: Current Account Savings Account
        - NEFT: National Electronic Funds Transfer
        - RTGS: Real Time Gross Settlement
        - IMPS: Immediate Payment Service
        - UPI: Unified Payments Interface
        - ATM: Automated Teller Machine
        
        2. Investment Terms:
        - ROI: Return on Investment
        - CAGR: Compound Annual Growth Rate
        - NAV: Net Asset Value
        - SIP: Systematic Investment Plan
        - ELSS: Equity Linked Savings Scheme
        - IPO: Initial Public Offering
        - FD: Fixed Deposit
        - RD: Recurring Deposit
        
        3. Tax Terms:
        - TDS: Tax Deducted at Source
        - TCS: Tax Collected at Source
        - GST: Goods and Services Tax
        - ITR: Income Tax Return
        - PAN: Permanent Account Number
        - DTAA: Double Taxation Avoidance Agreement
        
        4. Insurance Terms:
        - LIC: Life Insurance Corporation
        - ULIPs: Unit Linked Insurance Plans
        - EMI: Equated Monthly Installment
        - NCB: No Claim Bonus
        
        5. Market Terms:
        - BSE: Bombay Stock Exchange
        - NSE: National Stock Exchange
        - SEBI: Securities and Exchange Board of India
        - RBI: Reserve Bank of India
        - P/E: Price to Earnings Ratio
        - EPS: Earnings Per Share
        """,
        "Financial Ratios": """
        Important Financial Ratios:
        
        1. Profitability Ratios:
        - Gross Profit Margin
        - Net Profit Margin
        - Return on Equity (ROE)
        - Return on Assets (ROA)
        
        2. Liquidity Ratios:
        - Current Ratio
        - Quick Ratio
        - Working Capital Ratio
        
        3. Market Ratios:
        - P/E Ratio (Price/Earnings)
        - P/B Ratio (Price/Book)
        - Dividend Yield
        - Earnings Per Share (EPS)
        
        4. Debt Ratios:
        - Debt-to-Equity Ratio
        - Interest Coverage Ratio
        - Debt Service Coverage Ratio
        """
    }
}

# Mutual Fund Categories and Data
MUTUAL_FUND_CATEGORIES = {
    "Large Cap": ["101206", "119598", "120503"],
    "Mid Cap": ["118834", "119551"],
    "Small Cap": ["119598"],
    "Multi Cap": ["120503", "119551"],
    "Sectoral": ["119551"],
    "Hybrid": ["118834"],
}

# Updated Mutual Fund Data with risk and return metrics
MUTUAL_FUND_DATA = {
    "HDFC Top 100 Fund - Direct Plan-Growth": {
        "code": "101206",
        "category": "Large Cap",
        "risk_level": "Moderate",
        "min_investment": 5000,
        "return_1y": 15.8,
        "return_3y": 12.5,
        "return_5y": 11.2,
        "risk_score": 6,
        "expense_ratio": 0.95,
    },
    "SBI Small Cap Fund - Direct Plan-Growth": {
        "code": "119598",
        "category": "Small Cap",
        "risk_level": "High",
        "min_investment": 5000,
        "return_1y": 22.4,
        "return_3y": 15.8,
        "return_5y": 14.2,
        "risk_score": 8,
        "expense_ratio": 1.15,
    },
    "Axis Bluechip Fund - Direct Plan-Growth": {
        "code": "120503",
        "category": "Large Cap",
        "risk_level": "Low",
        "min_investment": 5000,
        "return_1y": 12.5,
        "return_3y": 10.2,
        "return_5y": 9.8,
        "risk_score": 4,
        "expense_ratio": 0.85,
    },
    "ICICI Prudential Technology Fund - Direct Plan-Growth": {
        "code": "119551",
        "category": "Sectoral",
        "risk_level": "Very High",
        "min_investment": 5000,
        "return_1y": 25.6,
        "return_3y": 18.9,
        "return_5y": 16.5,
        "risk_score": 9,
        "expense_ratio": 1.25,
    },
    "Parag Parikh Conservative Hybrid Fund - Direct Plan-Growth": {
        "code": "118834",
        "category": "Hybrid",
        "risk_level": "Low",
        "min_investment": 5000,
        "return_1y": 8.5,
        "return_3y": 7.8,
        "return_5y": 7.2,
        "risk_score": 3,
        "expense_ratio": 0.75,
    },
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .subheader {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0;
    }
    .buy-signal {
        color: green;
        font-weight: bold;
    }
    .sell-signal {
        color: red;
        font-weight: bold;
    }
    .neutral-signal {
        color: gray;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data fetching functions
@st.cache_data(ttl=300)  # 5 minutes cache
def get_stock_data(symbol, period='6mo', max_retries=3):
    """Fetch stock data for the given symbol and period."""
    for attempt in range(max_retries):
        try:
            # Add .NS suffix if not present for Indian stocks
            if not symbol.endswith(('.NS', '.BO', '^NSEI', '^BSESN')):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                st.warning(f"Attempt {attempt + 1}: No data available for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
            return data
        except Exception as e:
            st.warning(f"Attempt {attempt + 1}: Error fetching data for {symbol}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol, max_retries=3):
    """Get detailed information about a stock."""
    for attempt in range(max_retries):
        try:
            # Add .NS suffix if not present for Indian stocks
            if not symbol.endswith(('.NS', '.BO', '^NSEI', '^BSESN')):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                st.warning(f"Attempt {attempt + 1}: No information available for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
            return info
        except Exception as e:
            st.warning(f"Attempt {attempt + 1}: Error fetching information for {symbol}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    st.error(f"Failed to fetch information for {symbol} after {max_retries} attempts")
    return {}

@st.cache_data(ttl=300)
def get_nifty_data(period='1y', max_retries=3):
    """Get NIFTY 50 index data."""
    for attempt in range(max_retries):
        try:
            nifty = yf.Ticker("^NSEI")
            data = nifty.history(period=period)
            
            if data.empty:
                st.warning(f"Attempt {attempt + 1}: No NIFTY data available")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
            return data
        except Exception as e:
            st.warning(f"Attempt {attempt + 1}: Error fetching NIFTY data: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    st.error(f"Failed to fetch NIFTY data after {max_retries} attempts")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_sensex_data(period='1y', max_retries=3):
    """Get SENSEX index data."""
    for attempt in range(max_retries):
        try:
            sensex = yf.Ticker("^BSESN")
            data = sensex.history(period=period)
            
            if data.empty:
                st.warning(f"Attempt {attempt + 1}: No SENSEX data available")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
            return data
        except Exception as e:
            st.warning(f"Attempt {attempt + 1}: Error fetching SENSEX data: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    st.error(f"Failed to fetch SENSEX data after {max_retries} attempts")
    return pd.DataFrame()

def search_indian_stocks(query):
    """Search Indian stocks by name."""
    query_lower = query.lower()
    results = {}
    
    for name, symbol in INDIAN_STOCKS.items():
        if query_lower in name.lower() or query_lower in symbol.lower():
            results[name] = symbol
    
    return results

def calculate_rsi(data, periods=14):
    """Calculate RSI technical indicator."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD technical indicator."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return pd.DataFrame({
        'BB_Middle': sma,
        'BB_Upper': upper_band,
        'BB_Lower': lower_band
    })

def calculate_all_indicators(data):
    """Calculate all technical indicators."""
    if data.empty:
        return data
    
    result = data.copy()
    
    # Calculate RSI
    result['RSI'] = calculate_rsi(data)
    
    # Calculate MACD
    macd_data = calculate_macd(data)
    for col in macd_data.columns:
        result[col] = macd_data[col]
    
    # Calculate Bollinger Bands
    bb_data = calculate_bollinger_bands(data)
    for col in bb_data.columns:
        result[col] = bb_data[col]
    
    # Calculate Simple Moving Averages
    result['SMA_50'] = data['Close'].rolling(window=50).mean()
    result['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    return result

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER."""
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        return sentiment
    except:
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}

def get_stock_news_sentiment(symbol, company_name):
    """Get news sentiment for a stock."""
    try:
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Get news from Yahoo Finance
        stock = yf.Ticker(symbol)
        news_items = stock.news
        
        if not news_items:
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'NEUTRAL',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'news_items': [],
                'sentiment_trend': pd.DataFrame({'date': [], 'sentiment': []})
            }
        
        # Process news items
        processed_news = []
        sentiments = []
        dates = []
        
        for item in news_items[:10]:  # Process last 10 news items
            try:
                # Extract news content safely with fallbacks
                title = item.get('title', '')
                summary = item.get('summary', '') or item.get('description', '') or ''
                source = item.get('source', 'Unknown')
                publish_time = item.get('providerPublishTime', None)
                
                # Skip items without essential information
                if not title and not summary:
                    continue
                
                # Get sentiment scores
                text_to_analyze = f"{title} {summary}"
                sentiment = analyzer.polarity_scores(text_to_analyze)
                compound_score = sentiment['compound']
                
                # Format date
                if publish_time:
                    date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d')
                else:
                    date = datetime.now().strftime('%Y-%m-%d')
                
                processed_news.append({
                    'title': title or 'No Title Available',
                    'summary': summary or 'No Summary Available',
                    'source': source,
                    'date': date,
                    'sentiment': compound_score
                })
                
                sentiments.append(compound_score)
                dates.append(datetime.strptime(date, '%Y-%m-%d'))
            
            except Exception as e:
                st.warning(f"Skipped processing one news item due to: {str(e)}")
                continue
        
        if not processed_news:
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'NEUTRAL',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'news_items': [],
                'sentiment_trend': pd.DataFrame({'date': [], 'sentiment': []})
            }
        
        # Calculate overall sentiment
        overall_sentiment = sum(sentiments) / len(sentiments)
        
        # Count sentiment distribution
        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Generate sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = "BULLISH"
        elif overall_sentiment < -0.2:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        # Create sentiment trend
        sentiment_trend = pd.DataFrame({
            'date': dates,
            'sentiment': sentiments
        })
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_trend': sentiment_trend,
            'news_items': processed_news
        }
    
    except Exception as e:
        st.error(f"Error analyzing sentiment for {company_name}: {str(e)}")
        # Return neutral default values
        return {
            'overall_sentiment': 0,
            'sentiment_label': 'NEUTRAL',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'news_items': [],  # Always include empty news_items list
            'sentiment_trend': pd.DataFrame({'date': [], 'sentiment': []})
        }

def generate_signals(data):
    """Generate trading signals based on technical indicators."""
    if data.empty:
        return data
    
    signals = data.copy()
    
    # RSI signals
    signals['RSI_Signal'] = 0
    signals.loc[signals['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold
    signals.loc[signals['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought
    
    # MACD signals
    signals['MACD_Signal'] = 0
    signals.loc[signals['MACD'] > signals['MACD_Signal'], 'MACD_Signal'] = 1
    signals.loc[signals['MACD'] < signals['MACD_Signal'], 'MACD_Signal'] = -1
    
    # Moving Average signals
    signals['MA_Signal'] = 0
    signals.loc[signals['SMA_50'] > signals['SMA_200'], 'MA_Signal'] = 1
    signals.loc[signals['SMA_50'] < signals['SMA_200'], 'MA_Signal'] = -1
    
    # Bollinger Bands signals
    signals['BB_Signal'] = 0
    signals.loc[signals['Close'] < signals['BB_Lower'], 'BB_Signal'] = 1
    signals.loc[signals['Close'] > signals['BB_Upper'], 'BB_Signal'] = -1
    
    # Overall signal
    signals['Overall_Signal'] = (
        signals['RSI_Signal'] +
        signals['MACD_Signal'] +
        signals['MA_Signal'] +
        signals['BB_Signal']
    ) / 4
    
    return signals

def interpret_signal(signal):
    """Interpret the trading signal."""
    if signal > 0.2:
        return "BUY", int(abs(signal * 100))
    elif signal < -0.2:
        return "SELL", int(abs(signal * 100))
    else:
        return "HOLD", int(abs(signal * 100))

# Add new function to save user data
def save_user_activity(activity_type, activity_data):
    """Save user activity to Excel file with enhanced error handling"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_name = st.session_state.user_data.get('name', 'Anonymous') if hasattr(st.session_state, 'user_data') else 'Anonymous'
    
    # Create the data directory if it doesn't exist
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except Exception as e:
        st.error(f"Error creating data directory: {str(e)}")
        return
    
    # Define the Excel file path
    excel_file = 'data/user_activity_log.xlsx'
    
    try:
        # Load existing data if file exists
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                st.warning(f"Could not read existing activity log. Creating new file. Error: {str(e)}")
                df = pd.DataFrame(columns=['Timestamp', 'User', 'Activity Type', 'Activity Data'])
        else:
            df = pd.DataFrame(columns=['Timestamp', 'User', 'Activity Type', 'Activity Data'])
        
        # Add new activity
        new_row = pd.DataFrame({
            'Timestamp': [timestamp],
            'User': [user_name],
            'Activity Type': [activity_type],
            'Activity Data': [json.dumps(activity_data)]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to Excel with error handling
        try:
            df.to_excel(excel_file, index=False, engine='openpyxl')
        except PermissionError:
            st.error("Cannot save activity log: File is open in another program. Please close it and try again.")
        except Exception as e:
            st.error(f"Error saving activity log: {str(e)}")
        
    except Exception as e:
        st.error(f"Error processing activity log: {str(e)}")

def save_user_profile(user_data):
    """Save user profile to Excel file with enhanced error handling"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the data directory if it doesn't exist
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except Exception as e:
        st.error(f"Error creating data directory: {str(e)}")
        return
    
    # Define the Excel file path
    excel_file = 'data/user_profiles.xlsx'
    
    try:
        # Load existing data if file exists
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                st.warning(f"Could not read existing profiles. Creating new file. Error: {str(e)}")
                df = pd.DataFrame(columns=['Timestamp', 'Name', 'Phone', 'Occupation', 'Income Category', 
                                         'Monthly Income', 'Monthly Savings', 'Savings Ratio'])
        else:
            df = pd.DataFrame(columns=['Timestamp', 'Name', 'Phone', 'Occupation', 'Income Category', 
                                     'Monthly Income', 'Monthly Savings', 'Savings Ratio'])
        
        # Prepare user data
        profile_data = {
            'Timestamp': timestamp,
            'Name': user_data.get('name', ''),
            'Phone': user_data.get('phone', ''),
            'Occupation': user_data.get('occupation', ''),
            'Income Category': user_data.get('income_category', ''),
            'Monthly Income': user_data.get('monthly_income', 0),
            'Monthly Savings': user_data.get('monthly_savings', 0),
            'Savings Ratio': user_data.get('savings_ratio', 0)
        }
        
        # Add new profile
        new_row = pd.DataFrame([profile_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to Excel with error handling
        try:
            df.to_excel(excel_file, index=False, engine='openpyxl')
        except PermissionError:
            st.error("Cannot save user profile: File is open in another program. Please close it and try again.")
        except Exception as e:
            st.error(f"Error saving user profile: {str(e)}")
        
    except Exception as e:
        st.error(f"Error processing user profile: {str(e)}")

def show_onboarding():
    """Display the onboarding page"""
    st.title("Welcome to Financial Advisor Pro")
    st.write("Let's get to know you better to provide personalized financial guidance.")
    
    # Initialize form data from session state if it exists
    initial_data = st.session_state.user_data if hasattr(st.session_state, 'user_data') else {}
    
    with st.form("user_details"):
        st.subheader("👤 Personal Information")
        name = st.text_input("Name", value=initial_data.get('name', ''), placeholder="Enter your full name")
        phone = st.text_input("Phone Number", value=initial_data.get('phone', ''), placeholder="Enter your 10-digit phone number")
        
        # Basic user information
        col1, col2 = st.columns(2)
        with col1:
            occupation = st.selectbox(
                "Occupation",
                ["Select Occupation", "Student", "Salaried", "Self-Employed"],
                index=["Select Occupation", "Student", "Salaried", "Self-Employed"].index(initial_data.get('occupation', 'Select Occupation'))
            )
        
        # Additional fields for working professionals
        if occupation in ["Salaried", "Self-Employed"]:
            with col2:
                income_category = st.selectbox(
                    "Annual Income Category",
                    INCOME_CATEGORIES,
                    help="Select your annual income range"
                )
            
            st.subheader("💰 Financial Details")
            col3, col4 = st.columns(2)
            with col3:
                monthly_income = st.number_input(
                    "Monthly Income (₹)",
                    min_value=10000,
                    max_value=None,  # Remove upper limit
                    value=initial_data.get('monthly_income', 100000),
                    step=10000,
                    help="Your monthly take-home income"
                )
            
            with col4:
                monthly_savings = st.number_input(
                    "Monthly Savings Capacity (₹)",
                    min_value=0,
                    max_value=None,  # Remove upper limit
                    value=min(initial_data.get('monthly_savings', 50000), monthly_income),
                    step=5000,
                    help="How much can you save monthly?"
                )
            
            # Show savings analysis
            if monthly_income > 0:
                savings_ratio = (monthly_savings / monthly_income) * 100
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Savings Ratio", f"{savings_ratio:.1f}%")
                with col6:
                    st.metric("Monthly Disposable Income", f"₹{monthly_income - monthly_savings:,.2f}")
                
                # Savings feedback
                if savings_ratio < 20:
                    st.warning("⚠️ Your savings ratio is below the recommended 20%. Consider reducing expenses to save more.")
                elif savings_ratio > 50:
                    st.success("🌟 Excellent savings ratio! You're well-positioned for investments.")
                else:
                    st.info("✅ Good savings ratio. You're on track for financial stability.")
                
                # Investment potential
                recommended_monthly_investment = monthly_savings * 0.7
                st.write("#### 💡 Investment Potential")
                st.info(f"""
                Based on your savings of ₹{monthly_savings:,}, you could consider:
                - Monthly Investment: ₹{recommended_monthly_investment:,.0f}
                - Emergency Fund: ₹{monthly_savings * 0.2:,.0f}
                - Additional Savings: ₹{monthly_savings * 0.1:,.0f}
                """)
        
        elif occupation == "Student":
            st.info("""
            🎓 As a student, you'll get access to:
            - Basic financial education
            - Investment concepts
            - Budgeting tutorials
            - Market understanding
            """)
        
        # Submit button
        submitted = st.form_submit_button("Continue")
        
        if submitted:
            if not name or not phone:
                st.error("Please fill in your name and phone number.")
            elif occupation == "Select Occupation":
                st.error("Please select your occupation.")
            elif occupation in ["Salaried", "Self-Employed"]:
                if not income_category:
                    st.error("Please select your income category.")
                elif monthly_savings <= 0:
                    st.error("Please enter your monthly savings capacity.")
                elif monthly_savings > monthly_income:
                    st.error("Monthly savings cannot be greater than monthly income.")

                else:
                    # Save user data
                    user_data = {
                        "name": name,
                        "phone": phone,
                        "occupation": occupation,
                        "income_category": income_category,
                        "monthly_income": monthly_income,
                        "monthly_savings": monthly_savings,
                        "savings_ratio": savings_ratio,
                        "recommended_monthly_investment": recommended_monthly_investment
                    }
                    st.session_state.user_data = user_data
                    st.session_state.is_authenticated = True
                    # Save login details to CSV
                    save_login_details(name, phone, occupation)

                
                    # Save user profile to Excel
                    save_user_profile(user_data)
                    
                    # Log the onboarding activity
                    save_user_activity('onboarding_completed', user_data)
                    
                    st.success("Profile created successfully! Redirecting to Financial Literacy Hub...")
                    st.rerun()
            else:
                # Save student data
                user_data = {
                    "name": name,
                    "phone": phone,
                    "occupation": occupation
                }
                st.session_state.user_data = user_data
                st.session_state.is_authenticated = True
                
                # Save login details to CSV
                save_login_details(name, phone, occupation)

                # Save user profile to Excel
                save_user_profile(user_data)
                
                # Log the onboarding activity
                save_user_activity('student_onboarding_completed', user_data)
                
                st.success("Profile created successfully! Redirecting to Financial Literacy Hub...")
                st.rerun()

def show_financial_literacy():
    """Display comprehensive financial literacy content"""
    st.title("Financial Literacy Hub")
    
    # Get user name safely from session state
    user_name = st.session_state.user_data.get('name', 'User')
    occupation = st.session_state.user_data.get('occupation', '')
    
    st.write(f"Welcome {user_name}! Let's master your financial knowledge.")
    
    # Student-specific investment section
    if occupation == "Student":
        st.header("🎓 Student Investment Planner")
        amount = st.number_input(
            "Enter the amount you have for investment (₹)",
            min_value=0,
            max_value=None,
            value=1000,
            step=500,
            help="Enter the total amount you have available for investment"
        )
        
        if amount > 0:
            recommendations = get_student_investment_recommendations(amount)
            
            st.subheader("📊 Investment Recommendations")
            
            # Display recommendations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Safe Options", "Moderate Options", "Growth Options", "Educational"])
            
            with tab1:
                st.markdown("### 🛡️ Safe Investment Options")
                for option in recommendations["Safe Options"]:
                    with st.expander(f"{option['type']} (Min: ₹{option['min_amount']:,})"):
                        st.markdown(f"""
                        **Description:** {option['description']}
                        **Expected Returns:** {option['returns']}
                        **Liquidity:** {option['liquidity']}
                        **Risk Level:** {option['risk']}
                        """)
            
            with tab2:
                st.markdown("### 🔄 Moderate Risk Options")
                for option in recommendations["Moderate Options"]:
                    with st.expander(f"{option['type']} (Min: ₹{option['min_amount']:,})"):
                        st.markdown(f"""
                        **Description:** {option['description']}
                        **Expected Returns:** {option['returns']}
                        **Liquidity:** {option['liquidity']}
                        **Risk Level:** {option['risk']}
                        """)
            
            with tab3:
                st.markdown("### 📈 Growth Options")
                for option in recommendations["Growth Options"]:
                    with st.expander(f"{option['type']} (Min: ₹{option['min_amount']:,})"):
                        st.markdown(f"""
                        **Description:** {option['description']}
                        **Expected Returns:** {option['returns']}
                        **Liquidity:** {option['liquidity']}
                        **Risk Level:** {option['risk']}
                        """)
            
            with tab4:
                st.markdown("### 📚 Educational Investment")
                for option in recommendations["Educational"]:
                    with st.expander(f"{option['type']} (Min: ₹{option['min_amount']:,})"):
                        st.markdown(f"""
                        **Description:** {option['description']}
                        **Expected Returns:** {option['returns']}
                        **Liquidity:** {option['liquidity']}
                        **Risk Level:** {option['risk']}
                        """)
            
            # Investment tips for students
            st.info("""
            💡 **Tips for Student Investors:**
            1. Start small but start early
            2. Focus on learning before big investments
            3. Keep emergency fund before investing
            4. Research thoroughly before investing
            5. Consider investment horizon
            """)
    
    # Left sidebar for main topics
    selected_topic = st.sidebar.selectbox(
        "Choose Learning Module",
        list(FINANCIAL_BASICS.keys())
    )
    
    # Main content area
    st.header(f"📚 {selected_topic}")
    
    # Display subtopics with interactive elements
    for subtopic, content in FINANCIAL_BASICS[selected_topic].items():
        with st.expander(f"📖 {subtopic}", expanded=True):
            st.markdown(content)
            
            # Interactive elements based on topic
            if "Compound Interest" in subtopic:
                st.subheader("💡 Try Compound Interest Calculator")
                principal = st.number_input("Principal Amount (₹)", 1000, 1000000, 10000, key=f"principal_{subtopic}")
                rate = st.number_input("Annual Interest Rate (%)", 1.0, 20.0, 8.0, key=f"rate_{subtopic}")
                years = st.number_input("Time Period (Years)", 1, 30, 5, key=f"years_{subtopic}")
                
                if st.button("Calculate", key=f"calc_{subtopic}"):
                    final_amount = principal * (1 + rate/100)**years
                    st.metric("Final Amount", f"₹{final_amount:,.2f}")
                    st.metric("Interest Earned", f"₹{(final_amount - principal):,.2f}")
            
            elif "EMI" in subtopic:
                st.subheader("💡 Try EMI Calculator")
                loan_amount = st.number_input("Loan Amount (₹)", 10000, 10000000, 100000, key=f"loan_{subtopic}")
                interest_rate = st.number_input("Annual Interest Rate (%)", 1.0, 30.0, 10.0, key=f"interest_{subtopic}")
                tenure_years = st.number_input("Loan Tenure (Years)", 1, 30, 5, key=f"tenure_{subtopic}")
                
                if st.button("Calculate EMI", key=f"calc_{subtopic}"):
                    r = interest_rate/12/100
                    n = tenure_years * 12
                    emi = loan_amount * r * (1 + r)**n/((1 + r)**n - 1)
                    total_payment = emi * n
                    st.metric("Monthly EMI", f"₹{emi:,.2f}")
                    st.metric("Total Payment", f"₹{total_payment:,.2f}")
                    st.metric("Total Interest", f"₹{(total_payment - loan_amount):,.2f}")
            
            elif "Stock Market" in subtopic:
                st.subheader("📈 Live Market Overview")
                nifty_data = get_nifty_data(period='1d')
                sensex_data = get_sensex_data(period='1d')
                
                if not nifty_data.empty and not sensex_data.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        nifty_current = nifty_data['Close'].iloc[-1]
                        nifty_prev = nifty_data['Close'].iloc[-2]
                        nifty_change = (nifty_current - nifty_prev) / nifty_prev * 100
                        st.metric("NIFTY 50", f"₹{nifty_current:,.2f}", f"{nifty_change:+.2f}%")
                    
                    with col2:
                        sensex_current = sensex_data['Close'].iloc[-1]
                        sensex_prev = sensex_data['Close'].iloc[-2]
                        sensex_change = (sensex_current - sensex_prev) / sensex_prev * 100
                        st.metric("SENSEX", f"₹{sensex_current:,.2f}", f"{sensex_change:+.2f}%")
            
            # Mark topic as completed
            if st.button(f"Mark '{subtopic}' as Complete", key=f"complete_{subtopic}"):
                st.session_state.completed_topics.add(subtopic)
                st.success(f"🎉 Completed {subtopic}!")
    
    # Show progress
    total_topics = sum(len(topics) for topics in FINANCIAL_BASICS.values())
    completed = len(st.session_state.completed_topics)
    st.sidebar.progress(completed/total_topics)
    st.sidebar.write(f"Progress: {completed}/{total_topics} topics completed")

def show_stock_analysis():
    """Display stock analysis page"""
    # Initialize session state for stock analysis if not already done
    if 'last_stock_search' not in st.session_state:
        st.session_state.last_stock_search = None
    
    st.markdown("<h1 class='main-header'>Indian Stock Market Analysis</h1>", unsafe_allow_html=True)
    
    # Get user's financial data
    monthly_savings = st.session_state.user_data.get('monthly_savings', 0)
    recommended_monthly_investment = st.session_state.user_data.get('recommended_monthly_investment', monthly_savings * 0.7)
    
    # Display financial profile
    st.subheader("💰 Your Investment Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Savings", f"₹{monthly_savings:,}")
    with col2:
        st.metric("Recommended Stock Investment", f"₹{recommended_monthly_investment * 0.4:,.0f}")
    with col3:
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["Short Term (< 1 year)", "Medium Term (1-3 years)", "Long Term (> 3 years)"],
            help="Choose your investment timeframe"
        )
    
    # Investment amount selection
    st.subheader("💵 Stock Investment Planning")
    
    # Get recommended investment amount from session state or use default
    recommended_monthly_investment = (
        st.session_state.user_data.get('recommended_monthly_investment', 0) 
        if hasattr(st.session_state, 'user_data') 
        else 0
    )
    
    # Calculate default investment amount (minimum 1000)
    default_investment = max(1000, int(recommended_monthly_investment * 0.4) if recommended_monthly_investment else 1000)
    
    stock_investment = st.number_input(
        "Amount to Invest in Stocks (₹)",
        min_value=1000,
        max_value=None,
        value=default_investment,
        step=1000,
        help="Recommended: 40% of your investment capacity for stocks"
    )
    
    # Investment strategy based on amount
    if recommended_monthly_investment > 0 and stock_investment > recommended_monthly_investment * 0.4:
        st.warning("""
        ⚠️ Your selected stock investment amount is higher than recommended.
        Consider diversifying into other investment options to reduce risk.
        """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Search bar for stocks
        search_query = st.text_input("Search for Indian stocks", placeholder="Type company name...")
        if search_query:
            search_results = search_indian_stocks(search_query)
            if search_results:
                stock_options = {f"{name} ({symbol})": symbol for name, symbol in search_results.items()}
                selected_option = st.selectbox("Select a stock", list(stock_options.keys()))
                selected_symbol = stock_options[selected_option]
                selected_stock_name = selected_option.split(' (')[0]
            else:
                st.warning("No matching stocks found. Please try another search term.")
                st.stop()
        else:
            # Sector selection
            selected_sector = st.selectbox("Select Sector", list(INDIAN_SECTORS.keys()))
            selected_stocks = INDIAN_SECTORS[selected_sector]
            
            # Stock selection
            stock_dict = {k: v for k, v in INDIAN_STOCKS.items() if v in selected_stocks}
            stock_names = list(stock_dict.keys())
            if stock_names:
                selected_stock_name = st.selectbox("Select Stock", stock_names)
                selected_symbol = stock_dict[selected_stock_name]
            else:
                st.warning(f"No stocks available for sector: {selected_sector}")
                st.stop()
        
        # Time period
        period = st.selectbox(
            "Select Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
    
    with col2:
        st.markdown("### Stock Market Indices")
        
        # Fetch index data
        nifty_data = get_nifty_data()
        sensex_data = get_sensex_data()
        
        # Display current index values
        if not nifty_data.empty and not sensex_data.empty:
            nifty_current = nifty_data['Close'].iloc[-1]
            nifty_prev = nifty_data['Close'].iloc[-2]
            nifty_change = (nifty_current - nifty_prev) / nifty_prev * 100
            
            sensex_current = sensex_data['Close'].iloc[-1]
            sensex_prev = sensex_data['Close'].iloc[-2]
            sensex_change = (sensex_current - sensex_prev) / sensex_prev * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "NIFTY 50",
                    f"₹{nifty_current:.2f}",
                    f"{nifty_change:.2f}%"
                )
            
            with col2:
                st.metric(
                    "SENSEX",
                    f"₹{sensex_current:.2f}",
                    f"{sensex_change:.2f}%"
                )
    
    # Fetch and process stock data
    with st.spinner(f"Analyzing {selected_stock_name}..."):
        # Get stock data and info
        stock_data = get_stock_data(selected_symbol, period=period)
        stock_info = get_stock_info(selected_symbol)
        
        if stock_data.empty:
            st.error(f"No data available for {selected_stock_name}")
            st.stop()
        
        # Calculate indicators
        data_with_indicators = calculate_all_indicators(stock_data)
        
        # Generate signals
        signals = generate_signals(data_with_indicators)
        
        # Display stock information and price
        st.markdown(f"## {selected_stock_name} ({selected_symbol})")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if stock_info:
                sector = stock_info.get('sector', 'N/A')
                industry = stock_info.get('industry', 'N/A')
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
            
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            st.metric(
                "Current Price",
                f"₹{current_price:.2f}",
                f"{price_change:.2f}%"
            )
        
        with col2:
            if stock_info:
                market_cap = stock_info.get('marketCap', 0)
                pe_ratio = stock_info.get('trailingPE', 0)
                
                if market_cap:
                    market_cap_cr = market_cap / 10000000  # Convert to crores
                    st.metric("Market Cap", f"₹{market_cap_cr:.2f} Cr")
                
                if pe_ratio:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        
        with col3:
            if stock_info:
                day_low = stock_info.get('dayLow', 0)
                day_high = stock_info.get('dayHigh', 0)
                
                if day_low and day_high:
                    st.metric("Day Range", f"₹{day_low:.2f} - ₹{day_high:.2f}")
                
                volume = stock_info.get('volume', 0)
                if volume:
                    st.metric("Volume", f"{volume:,}")
        
        # Investment Recommendation
        st.subheader("💡 Investment Recommendation")
        
        # Get the latest signal
        if 'Overall_Signal' in signals.columns:
            latest_signal = signals['Overall_Signal'].iloc[-1]
            recommendation, confidence = interpret_signal(latest_signal)
            
            # Calculate number of shares possible
            possible_shares = int(stock_investment / current_price)
            investment_amount = possible_shares * current_price
            remaining_amount = stock_investment - investment_amount
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                signal_color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "gray"
                signal_emoji = "🔺" if recommendation == "BUY" else "🔻" if recommendation == "SELL" else "➡️"
                
                st.markdown(f"""
                <div style='background-color: rgba({0 if signal_color != 'red' else 255}, {0 if signal_color != 'green' else 255}, 0, 0.1); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h1 style='color: {signal_color};'>{signal_emoji} {recommendation}</h1>
                    <p>Confidence: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Investment Plan")
                st.markdown(f"""
                **For your investment of ₹{stock_investment:,.2f}:**
                - Number of shares possible: {possible_shares}
                - Total investment needed: ₹{investment_amount:,.2f}
                - Remaining amount: ₹{remaining_amount:,.2f}
                
                **Recommended Strategy ({investment_horizon}):**
                """)
                
                if investment_horizon == "Short Term (< 1 year)":
                    st.markdown("""
                    - Set strict stop-loss at 5-7% below purchase price
                    - Book profits at 10-15% gains
                    - Monitor daily for market movements
                    - Keep track of quarterly results
                    """)
                elif investment_horizon == "Medium Term (1-3 years)":
                    st.markdown("""
                    - Set wider stop-loss at 10-12% below purchase price
                    - Consider averaging on dips
                    - Monitor weekly for major changes
                    - Focus on company fundamentals
                    """)
                else:  # Long Term
                    st.markdown("""
                    - Focus on company fundamentals and growth
                    - Consider systematic investment (monthly/quarterly)
                    - Monitor quarterly for major changes
                    - Hold through market cycles
                    """)
        
        # Display technical indicators
        st.markdown("### Technical Indicators")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Price & Moving Averages", "RSI", "MACD & Bollinger Bands"])
        
        with tab1:
            fig = go.Figure()
            
            # Price
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['Close'],
                name="Price",
                line=dict(color='blue')
            ))
            
            # Moving Averages
            if 'SMA_50' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['SMA_50'],
                    name="SMA 50",
                    line=dict(color='orange')
                ))
            
            if 'SMA_200' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['SMA_200'],
                    name="SMA 200",
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title=f"{selected_stock_name} Stock Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'RSI' in data_with_indicators.columns:
                fig = go.Figure()
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['RSI'],
                    name="RSI",
                    line=dict(color='purple')
                ))
                
                # Add overbought and oversold lines
                fig.add_shape(
                    type="line",
                    x0=data_with_indicators.index[0],
                    y0=70,
                    x1=data_with_indicators.index[-1],
                    y1=70,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=data_with_indicators.index[0],
                    y0=30,
                    x1=data_with_indicators.index[-1],
                    y1=30,
                    line=dict(color="green", width=2, dash="dash")
                )
                
                fig.update_layout(
                    title=f"{selected_stock_name} RSI Indicator",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("RSI data not available for the selected timeframe")
        
        with tab3:
            # Price and Bollinger Bands
            if all(col in data_with_indicators.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['Close'],
                    name="Price",
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Upper'],
                    name="Upper Band",
                    line=dict(color='gray', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Lower'],
                    name="Lower Band",
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.2)'
                ))
                
                fig.update_layout(
                    title=f"{selected_stock_name} Price with Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bollinger Bands data not available for the selected timeframe")
            
            # MACD chart
            if all(col in data_with_indicators.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD'],
                    name="MACD",
                    line=dict(color='blue')
                ))
                
                fig2.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD_Signal'],
                    name="Signal",
                    line=dict(color='red')
                ))
                
                # Add histogram for MACD histogram
                fig2.add_trace(go.Bar(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD_Histogram'],
                    name="Histogram",
                    marker_color=np.where(data_with_indicators['MACD_Histogram'] > 0, 'green', 'red')
                ))
                
                fig2.update_layout(
                    title=f"{selected_stock_name} MACD Indicator",
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("MACD data not available for the selected timeframe")
        
        # Risk Warning
        st.warning("""
        ⚠️ **Investment Risk Disclaimer:**
        - Past performance is not indicative of future returns
        - Stock investments carry market risks
        - Always diversify your portfolio
        - Consider consulting a financial advisor
        """)
    
    # Add predictions section after technical indicators
    if not stock_data.empty:
        show_predictions(stock_data, selected_stock_name)
    
    # Log stock analysis activity
    if selected_symbol and selected_symbol != st.session_state.last_stock_search:
        activity_data = {
            'stock_symbol': selected_symbol,
            'stock_name': selected_stock_name,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_activity('stock_analysis', activity_data)
        st.session_state.last_stock_search = selected_symbol

def show_portfolio_optimization():
    """Display portfolio optimization page"""
    st.markdown("<h1 class='main-header'>Portfolio Optimization</h1>", unsafe_allow_html=True)
    
    st.write("""
    Optimize your investment portfolio to maximize returns or minimize risk based on historical performance.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sector selection
        selected_sector = st.selectbox("Select Sector", list(INDIAN_SECTORS.keys()))
        selected_stocks = INDIAN_SECTORS[selected_sector]
        
        # Stock selection
        stock_dict = {k: v for k, v in INDIAN_STOCKS.items() if v in selected_stocks}
        stock_names = list(stock_dict.keys())
        symbols = st.multiselect("Select Stocks for Portfolio", stock_names, default=stock_names[:3])
        
        if not symbols or len(symbols) < 2:
            st.warning("Please select at least two stocks for portfolio optimization.")
            st.stop()
        
        selected_symbols = [stock_dict[name] for name in symbols]
        
        # Investment amount
        investment_amount = st.number_input("Investment Amount (₹)", 10000.0, 10000000.0, 100000.0)
        
        # Optimization goal
        optimization_goal = st.radio(
            "Optimization Goal",
            ["Maximize Sharpe Ratio", "Maximize Return", "Minimize Volatility"]
        )
        
        # Time period
        period = st.selectbox(
            "Select Data Period",
            ["1y", "2y", "3y", "5y"],
            index=0
        )
    
    with col2:
        st.image("https://images.unsplash.com/photo-1594135356513-14291e55162a", width=400)
        st.write("Optimize your investment portfolio to maximize returns or minimize risk based on historical performance.")
    
    # Store the button click in a variable
    submitted = st.button("Optimize Portfolio")
    
    if submitted:
        with st.spinner("Optimizing portfolio allocation..."):
            # Fetch data for all selected symbols
            data = pd.DataFrame()
            for i, symbol in enumerate(selected_symbols):
                stock_data = get_stock_data(symbol, period=period)
                if not stock_data.empty:
                    data[symbols[i]] = stock_data['Close']
            
            if data.empty:
                st.error("No data found for the selected stocks.")
                st.stop()
            
            # Calculate returns and covariance
            returns = data.pct_change().dropna()
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Run Monte Carlo simulation
            np.random.seed(42)
            num_portfolios = 5000
            results = np.zeros((num_portfolios, len(symbols) + 3))
            
            for i in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(len(symbols))
                weights /= np.sum(weights)
                
                # Calculate portfolio return and volatility
                portfolio_return = np.sum(returns.mean() * weights) * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Calculate Sharpe ratio (assuming 0 risk-free rate)
                sharpe_ratio = portfolio_return / portfolio_volatility
                
                # Store results
                results[i, :len(symbols)] = weights
                results[i, len(symbols)] = portfolio_return
                results[i, len(symbols) + 1] = portfolio_volatility
                results[i, len(symbols) + 2] = sharpe_ratio
            
            # Convert results to DataFrame
            columns = symbols + ['Return', 'Volatility', 'Sharpe']
            portfolios = pd.DataFrame(results, columns=columns)
            
            # Find optimal portfolio based on goal
            if optimization_goal == "Maximize Sharpe Ratio":
                optimal_idx = portfolios['Sharpe'].idxmax()
            elif optimization_goal == "Maximize Return":
                optimal_idx = portfolios['Return'].idxmax()
            else:  # Minimize Volatility
                optimal_idx = portfolios['Volatility'].idxmin()
            
            optimal_portfolio = portfolios.iloc[optimal_idx]
            
            # Display results
            st.subheader("Optimal Portfolio Allocation")
            
            # Get weights and calculate amounts
            weights = optimal_portfolio[:len(symbols)]
            amounts = weights * investment_amount
            
            # Create allocation dataframe
            allocation_data = []
            for i, symbol in enumerate(symbols):
                allocation_data.append({
                    'Stock': symbol,
                    'Weight': f"{weights[i]:.2%}",
                    'Amount (₹)': f"₹{amounts[i]:.2f}"
                })
            
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, use_container_width=True)
            
            # Display portfolio metrics
            st.subheader("Portfolio Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected Annual Return",
                    f"{optimal_portfolio['Return']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Annual Volatility",
                    f"{optimal_portfolio['Volatility']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{optimal_portfolio['Sharpe']:.2f}"
                )
            
            # Plot efficient frontier
            st.subheader("Efficient Frontier")
            
            fig = px.scatter(
                x=portfolios['Volatility'],
                y=portfolios['Return'],
                color=portfolios['Sharpe'],
                color_continuous_scale='viridis',
                title="Risk vs Return - Efficient Frontier",
                labels={
                    'x': 'Annualized Volatility',
                    'y': 'Annualized Return',
                    'color': 'Sharpe Ratio'
                }
            )
            
            # Highlight the optimal portfolio
            fig.add_trace(
                go.Scatter(
                    x=[optimal_portfolio['Volatility']],
                    y=[optimal_portfolio['Return']],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        line=dict(
                            color='black',
                            width=2
                        ),
                        symbol='star'
                    ),
                    name='Optimal Portfolio'
                )
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart of allocation
            fig = px.pie(
                names=symbols,
                values=weights,
                title="Portfolio Allocation",
                hole=0.4
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Log portfolio optimization activity when form is submitted
    if submitted:
        activity_data = {
            'optimization_goal': optimization_goal,
            'investment_amount': investment_amount,
            'selected_stocks': symbols,
            'period': period,
            'optimization_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_activity('portfolio_optimization', activity_data)

def get_fund_recommendations(risk_preference, return_preference, investment_amount, filtered_funds=None):
    """Get fund recommendations based on user preferences"""
    scored_funds = []
    funds_to_analyze = filtered_funds if filtered_funds else MUTUAL_FUND_DATA

    for fund_name, fund_data in funds_to_analyze.items():
        # Skip funds with minimum investment higher than user's amount
        if fund_data["min_investment"] > investment_amount:
            continue

        # Calculate preference score
        risk_match = 10 - abs(risk_preference - fund_data["risk_score"])
        return_match = (fund_data["return_3y"] / 25) * 10  # Normalize returns to 0-10 scale

        # Weight the scores based on user's return preference
        final_score = risk_match * (1 - return_preference) + return_match * return_preference

        scored_funds.append({
            "name": fund_name,
            "score": final_score,
            "data": fund_data
        })

    # Sort funds by score
    return sorted(scored_funds, key=lambda x: x["score"], reverse=True)

def display_fund_recommendations(recommendations):
    """Display fund recommendations in the Streamlit interface"""
    if not recommendations:
        st.warning("No funds match your criteria. Try adjusting your preferences or investment amount.")
        return

    st.subheader("🌟 Recommended Funds")
    
    # Display top recommendations
    for i, fund in enumerate(recommendations[:5]):
        fund_data = fund['data']
        
        with st.expander(f"{i+1}. {fund['name']} (Match Score: {fund['score']:.1f}/10)", expanded=i==0):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("**Key Metrics**")
                st.markdown(f"""
                - Category: {fund_data['category']}
                - Risk Level: {fund_data['risk_level']} (Risk Score: {fund_data['risk_score']})
                - Minimum Investment: ₹{fund_data['min_investment']:,}
                """)
            
            with col2:
                st.markdown("**Returns**")
                st.markdown(f"""
                - 1 Year: {fund_data['return_1y']}%
                - 3 Years: {fund_data['return_3y']}%
                - 5 Years: {fund_data['return_5y']}%
                """)
            
            with col3:
                st.markdown("**Expense Ratio**")
                st.markdown(f"{fund_data['expense_ratio']}%")
            
            # Risk-Return visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[fund_data['risk_score']],
                y=[fund_data['return_3y']],
                mode='markers+text',
                marker=dict(size=15, color='blue'),
                text=['Current Fund'],
                textposition='top center'
            ))
            
            fig.update_layout(
                title='Risk-Return Profile',
                xaxis_title='Risk Score (1-10)',
                yaxis_title='3-Year Return (%)',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Chart
    st.subheader("📊 Performance Comparison")
    
    comparison_data = pd.DataFrame([
        {
            'Fund': fund['name'],
            '1Y Return': fund['data']['return_1y'],
            '3Y Return': fund['data']['return_3y'],
            '5Y Return': fund['data']['return_5y'],
            'Risk Score': fund['data']['risk_score'],
            'Expense Ratio': fund['data']['expense_ratio']
        }
        for fund in recommendations[:5]
    ])
    
    # Returns comparison
    fig = px.bar(
        comparison_data,
        x='Fund',
        y=['1Y Return', '3Y Return', '5Y Return'],
        title='Returns Comparison',
        barmode='group'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk vs Return scatter plot
    fig = px.scatter(
        comparison_data,
        x='Risk Score',
        y='3Y Return',
        size='Expense Ratio',
        hover_data=['Fund'],
        title='Risk vs Return Analysis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download recommendations
    st.markdown("### 📥 Download Detailed Report")
    csv = comparison_data.to_csv(index=False)
    st.download_button(
        label="Download Fund Comparison (CSV)",
        data=csv,
        file_name="mutual_fund_recommendations.csv",
        mime="text/csv"
    )
    
    # Disclaimer
    st.info("""
    **Disclaimer:** Past performance does not guarantee future returns. These recommendations are based on historical data 
    and should not be the sole basis for investment decisions. Please read scheme-related documents carefully and consult 
    with a financial advisor before investing.
    """)

def calculate_sip_returns(initial_amount, monthly_contribution, investment_years, expected_return):
    """Calculate SIP returns based on given parameters"""
    total_invested = initial_amount + (monthly_contribution * 12 * investment_years)
    future_values = [initial_amount]

    for year in range(1, investment_years + 1):
        prev_value = future_values[-1]
        year_start = prev_value
        year_end = year_start * (1 + expected_return / 100)

        # Add monthly contributions for the year
        monthly_return = (1 + expected_return / 100) ** (1 / 12) - 1
        for month in range(12):
            year_end += monthly_contribution * (1 + monthly_return) ** (11 - month)

        future_values.append(year_end)

    future_value = future_values[-1]
    wealth_gained = future_value - total_invested
    absolute_return = (wealth_gained / total_invested) * 100

    return {
        "future_value": future_value,
        "total_invested": total_invested,
        "wealth_gained": wealth_gained,
        "absolute_return": absolute_return,
        "year_wise_values": future_values,
    }

def calculate_required_sip(target_amount, investment_years, expected_return):
    """Calculate required monthly SIP to reach target amount"""
    monthly_return = (1 + expected_return / 100) ** (1 / 12) - 1
    months = investment_years * 12

    # PMT formula: PMT = FV * r / ((1 + r)^n - 1)
    # where FV is future value, r is monthly return rate, n is number of months
    required_sip = target_amount * monthly_return / ((1 + monthly_return) ** months - 1)
    return required_sip

def mutual_fund_analysis():
    """Handle Mutual Fund Analysis section"""
    if not st.session_state.user_data.get('monthly_savings'):
        st.warning("⚠️ Please complete your financial profile first!")
        if st.button("Update Profile"):
            st.session_state.is_authenticated = False
            st.rerun()
        return

    st.markdown("<h1 class='main-header'>Smart Mutual Fund Recommendations</h1>", unsafe_allow_html=True)

    # Get user's financial data
    monthly_savings = st.session_state.user_data.get('monthly_savings', 0)
    income_category = st.session_state.user_data.get('income_category', '')
    recommended_monthly_investment = st.session_state.user_data.get('recommended_monthly_investment', monthly_savings * 0.7)

    # Display financial profile
    st.subheader("💰 Your Financial Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Savings", f"₹{monthly_savings:,}")
    with col2:
        st.metric("Income Category", income_category)
    with col3:
        st.metric("Recommended Investment", f"₹{recommended_monthly_investment:,.0f}")

    # Investment amount selection
    st.subheader("💵 Investment Planning")
    investment_amount = st.number_input(
        "Monthly SIP Amount",
        min_value=1000,
        max_value=None,  # Remove upper limit
        value=int(recommended_monthly_investment),
        step=1000,
        help="Recommended: 70% of your monthly savings"
    )

    # Investment amount feedback
    if investment_amount > recommended_monthly_investment:
        st.warning("""
        ⚠️ Your selected investment amount is higher than recommended.
        Consider keeping some funds for emergencies.
        """)
    elif investment_amount < recommended_monthly_investment * 0.5:
        st.info("""
        💡 You could potentially invest more based on your savings.
        Consider increasing your investment for better long-term returns.
        """)

    # Category and risk preference
    st.subheader("🎯 Investment Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox(
            "Fund Category",
            list(MUTUAL_FUND_CATEGORIES.keys()),
            help="Choose the type of mutual fund you want to invest in"
        )
    
    with col2:
        risk_preference = st.slider(
            "Risk Appetite",
            min_value=1,
            max_value=10,
            value=5,
            help="1: Very Conservative, 10: Very Aggressive"
        )

    # Return preference
    return_preference = st.slider(
        "Return Priority",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0: Focus on Risk Management, 1: Focus on Maximum Returns"
    )

    if st.button("Get Personalized Recommendations"):
        with st.spinner("Analyzing funds based on your preferences..."):
            # Filter funds by category and investment amount
            category_funds = MUTUAL_FUND_CATEGORIES[selected_category]
            filtered_funds = {
                name: data
                for name, data in MUTUAL_FUND_DATA.items()
                if data["code"] in category_funds and data["min_investment"] <= investment_amount
            }

            if not filtered_funds:
                st.error(f"""
                No suitable funds found. This might be because:
                1. Minimum investment requirement is higher than ₹{investment_amount:,}
                2. No funds available in {selected_category} category
                
                Suggestions:
                - Try increasing your investment amount
                - Select a different fund category
                - Adjust your risk preferences
                """)
                return

            # Get recommendations
            recommendations = get_fund_recommendations(
                risk_preference,
                return_preference,
                investment_amount,
                filtered_funds
            )

            # Display recommendations
            display_fund_recommendations(recommendations)

            # Investment summary
            st.subheader("📊 Investment Summary")
            st.info(f"""
            Based on your profile:
            - Monthly Investment: ₹{investment_amount:,}
            - Remaining Savings: ₹{monthly_savings - investment_amount:,}
            - Emergency Fund: Keep at least ₹{monthly_savings * 0.2:,.0f} for emergencies
            """)

def sip_calculator():
    """Handle SIP Calculator section"""
    st.markdown("<h1 class='main-header'>SIP Calculator</h1>", unsafe_allow_html=True)

    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Calculate Future Value", "Calculate Required SIP"],
        horizontal=True,
    )

    if mode == "Calculate Future Value":
        st.subheader("💰 Calculate Future Value")

        col1, col2 = st.columns(2)
        with col1:
            initial_amount = st.number_input(
                "Initial Investment (₹)", 0, None, 10000, 1000
            )
            monthly_contribution = st.number_input(
                "Monthly SIP Amount (₹)", 500, None, 5000, 500
            )

        with col2:
            investment_years = st.slider("Investment Duration (Years)", 1, 40, 10)
            expected_return = st.slider(
                "Expected Annual Return (%)", 1.0, 30.0, 12.0, 0.5
            )

        if st.button("Calculate Returns"):
            results = calculate_sip_returns(
                initial_amount, monthly_contribution, investment_years, expected_return
            )

            # Display results
            st.subheader("📊 Investment Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Future Value", f"₹{results['future_value']:,.2f}")
            with col2:
                st.metric("Total Investment", f"₹{results['total_invested']:,.2f}")
            with col3:
                st.metric("Wealth Gained", f"₹{results['wealth_gained']:,.2f}")

            # Year-wise growth visualization
            years = list(range(investment_years + 1))
            invested_amount = [
                initial_amount + (monthly_contribution * 12 * year) for year in years
            ]

            growth_data = pd.DataFrame(
                {
                    "Year": years,
                    "Total Investment": invested_amount,
                    "Future Value": results["year_wise_values"],
                    "Returns": [
                        fv - inv
                        for fv, inv in zip(results["year_wise_values"], invested_amount)
                    ],
                }
            )

            # Plot stacked bar chart
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=growth_data["Year"],
                    y=growth_data["Total Investment"],
                    name="Amount Invested",
                    marker_color="blue",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=growth_data["Year"],
                    y=growth_data["Returns"],
                    name="Returns Earned",
                    marker_color="green",
                )
            )
            fig.update_layout(
                barmode="stack",
                title="Year-wise Investment Growth",
                xaxis_title="Years",
                yaxis_title="Amount (₹)",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display detailed table
            st.subheader("📋 Year-wise Breakdown")
            detailed_data = pd.DataFrame(
                {
                    "Year": years,
                    "Total Investment": invested_amount,
                    "Future Value": results["year_wise_values"],
                    "Returns": [
                        fv - inv
                        for fv, inv in zip(results["year_wise_values"], invested_amount)
                    ],
                    "Returns (%)": [
                        (fv - inv) / inv * 100 if inv > 0 else 0
                        for fv, inv in zip(results["year_wise_values"], invested_amount)
                    ],
                }
            )
            detailed_data = detailed_data.round(2)
            st.dataframe(detailed_data, use_container_width=True)

    else:  # Calculate Required SIP
        st.subheader("🎯 Calculate Required Monthly SIP")

        col1, col2 = st.columns(2)
        with col1:
            target_amount = st.number_input(
                "Target Amount (₹)", 10000, None, 1000000, 10000
            )
            investment_years = st.slider("Investment Duration (Years)", 1, 40, 10)

        with col2:
            expected_return = st.slider(
                "Expected Annual Return (%)", 1.0, 30.0, 12.0, 0.5
            )

        if st.button("Calculate Required SIP"):
            required_sip = calculate_required_sip(
                target_amount, investment_years, expected_return
            )

            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Required Monthly SIP", f"₹{required_sip:,.2f}")
            with col2:
                total_investment = required_sip * 12 * investment_years
                st.metric("Total Investment", f"₹{total_investment:,.2f}")
            with col3:
                returns = target_amount - total_investment
                st.metric("Expected Returns", f"₹{returns:,.2f}")

            # Visualization
            monthly_data = pd.DataFrame(
                {
                    "Month": range(1, investment_years * 12 + 1),
                    "Investment": [
                        required_sip * m for m in range(1, investment_years * 12 + 1)
                    ],
                }
            )

            fig = px.line(
                monthly_data,
                x="Month",
                y="Investment",
                title="Cumulative Investment Over Time",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Add disclaimer
    st.info("""
    **Disclaimer:** The calculations are based on the assumption of a constant rate of return. 
    Actual returns may vary due to market conditions and other factors. 
    This calculator is for illustration purposes only and should not be considered as financial advice.
    """)

def show_sentiment_analysis():
    """Display sentiment analysis page with finance news website interface"""
    st.markdown("<h1 class='main-header'>Market Pulse & News Analysis</h1>", unsafe_allow_html=True)
    
    # Initialize variables
    selected_symbol = None
    selected_stock_name = None
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📰 Market News", "📊 Sentiment Dashboard", "🔍 Stock-Specific Analysis"])
    
    # Initialize session state for last sentiment search if not exists
    if 'last_sentiment_search' not in st.session_state:
        st.session_state.last_sentiment_search = None
    
    with tab1:
        st.subheader("Today's Market News")
        
        # Market Overview
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Fetch NIFTY and SENSEX data
            nifty_data = get_nifty_data(period='1d')
            sensex_data = get_sensex_data(period='1d')
            
            if not nifty_data.empty and not sensex_data.empty:
                nifty_change = ((nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[0]) / nifty_data['Close'].iloc[0]) * 100
                sensex_change = ((sensex_data['Close'].iloc[-1] - sensex_data['Close'].iloc[0]) / sensex_data['Close'].iloc[0]) * 100
                
                market_status = "🟢 Bullish" if nifty_change > 0 and sensex_change > 0 else "🔴 Bearish" if nifty_change < 0 and sensex_change < 0 else "⚪ Mixed"
                st.markdown(f"### Market Status: {market_status}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("NIFTY 50", f"₹{nifty_data['Close'].iloc[-1]:,.2f}", f"{nifty_change:+.2f}%")
                with col_b:
                    st.metric("SENSEX", f"₹{sensex_data['Close'].iloc[-1]:,.2f}", f"{sensex_change:+.2f}%")
        
        with col2:
            st.markdown("### Trending Topics")
            trending_topics = ["#MarketCrash", "#Inflation", "#FedRate", "#IPO", "#Earnings"]
            for topic in trending_topics:
                st.markdown(f"- {topic}")
        
        with col3:
            st.markdown("### Market Hours")
            st.markdown("""
            - Pre-Market: 9:00 AM
            - Market Open: 9:15 AM
            - Market Close: 3:30 PM
            - After Hours: 4:00 PM
            """)
        
        # News Categories
        st.markdown("### News Categories")
        categories = ["Top Stories", "Markets", "Economy", "Companies", "Global", "Expert Views"]
        selected_category = st.selectbox("Select Category", categories)
        
        # Display news based on category
        st.markdown(f"### {selected_category}")
        
        # Fetch news for top companies in each sector
        sectors_news = {}
        for sector, stocks in INDIAN_SECTORS.items():
            sector_sentiment = 0
            sector_news = []
            
            for stock in stocks[:2]:  # Get news for top 2 stocks in each sector
                try:
                    company_name = [name for name, symbol in INDIAN_STOCKS.items() if symbol == stock][0]
                    news_data = get_stock_news_sentiment(stock, company_name)
                    
                    if news_data and news_data['news_items']:
                        sector_sentiment += news_data['overall_sentiment']
                        sector_news.extend(news_data['news_items'])
                except:
                    continue
            
            if sector_news:
                sectors_news[sector] = {
                    'sentiment': sector_sentiment / len(sector_news),
                    'news': sorted(sector_news, key=lambda x: x['date'], reverse=True)
                }
        
        # Display news in a modern layout
        for sector, data in sectors_news.items():
            with st.expander(f"{sector} News", expanded=sector == list(sectors_news.keys())[0]):
                sentiment = data['sentiment']
                sentiment_color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "gray"
                st.markdown(f"Sector Sentiment: <span style='color: {sentiment_color};'>{'🔺' if sentiment > 0 else '🔻' if sentiment < 0 else '➡️'}</span>", unsafe_allow_html=True)
                
                for news in data['news'][:5]:  # Show top 5 news per sector
                    st.markdown(f"""
                    <div style='border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 10px 0;'>
                        <h4>{news['title']}</h4>
                        <p>{news['summary']}</p>
                        <p><small>Source: {news['source']} | {news['date']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Market Sentiment Dashboard")
        
        # Overall market sentiment
        market_sentiments = {}
        for sector, stocks in INDIAN_SECTORS.items():
            sector_sentiment = 0
            count = 0
            
            for stock in stocks:
                try:
                    company_name = [name for name, symbol in INDIAN_STOCKS.items() if symbol == stock][0]
                    sentiment_data = get_stock_news_sentiment(stock, company_name)
                    
                    if sentiment_data:
                        sector_sentiment += sentiment_data['overall_sentiment']
                        count += 1
                except:
                    continue
            
            if count > 0:
                market_sentiments[sector] = sector_sentiment / count
        
        # Create sentiment heatmap
        if market_sentiments:
            sentiment_df = pd.DataFrame(list(market_sentiments.items()), columns=['Sector', 'Sentiment'])
            # Filter out sectors with zero sentiment and add a small epsilon to avoid division by zero
            sentiment_df = sentiment_df[sentiment_df['Sentiment'] != 0].copy()
            if not sentiment_df.empty:
                # Add a small constant to avoid zero values while preserving sign
                sentiment_df['AbsValue'] = sentiment_df['Sentiment'].abs() + 1e-10
                
                fig = px.treemap(
                    sentiment_df,
                    path=['Sector'],
                    values='AbsValue',  # Use the adjusted absolute values for size
                    color='Sentiment',  # Use original sentiment for color
                    color_continuous_scale='RdYlGn',
                    title='Market Sentiment Heatmap'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment trends
                st.subheader("Sector-wise Sentiment Trends")
                
                fig = go.Figure()
                for sector, sentiment in market_sentiments.items():
                    if sentiment != 0:  # Only plot non-zero sentiments
                        fig.add_trace(go.Bar(
                            name=sector,
                            x=[sector],
                            y=[sentiment],
                            marker_color='green' if sentiment > 0 else 'red'
                        ))
                
                fig.update_layout(
                    title="Sector Sentiment Analysis",
                    xaxis_title="Sectors",
                    yaxis_title="Sentiment Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No significant market sentiment data available for visualization.")
    
    with tab3:
        st.subheader("Stock-Specific Sentiment Analysis")
        
        # Stock selection
        search_query = st.text_input("Search for Indian stocks", placeholder="Type company name...")
        if search_query:
            search_results = search_indian_stocks(search_query)
            if search_results:
                stock_options = {f"{name} ({symbol})": symbol for name, symbol in search_results.items()}
                selected_option = st.selectbox("Select a stock", list(stock_options.keys()))
                if selected_option:  # Add check for selected_option
                    selected_symbol = stock_options[selected_option]
                    selected_stock_name = selected_option.split(' (')[0]
                    
                    # Only proceed with sentiment analysis if we have a valid symbol
                    if selected_symbol and selected_symbol != st.session_state.last_sentiment_search:
                        sentiment_data = get_stock_news_sentiment(selected_symbol, selected_stock_name)
                        
                        if sentiment_data:
                            # Create columns for metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_score = sentiment_data['overall_sentiment']
                                st.metric(
                                    "Overall Sentiment",
                                    f"{sentiment_score:.2f}",
                                    delta=f"{sentiment_score*100:.1f}% {'Positive' if sentiment_score > 0 else 'Negative'}"
                                )
                            
                            with col2:
                                total_count = sentiment_data['positive_count'] + sentiment_data['negative_count'] + sentiment_data['neutral_count']
                                positive_ratio = sentiment_data['positive_count'] / total_count if total_count > 0 else 0
                                st.metric("Positive News Ratio", f"{positive_ratio:.1%}")
                            
                            with col3:
                                news_count = len(sentiment_data.get('news_items', []))
                                st.metric("Total News Count", news_count)
                            
                            # News Timeline
                            if sentiment_data.get('news_items'):
                                st.subheader("News Timeline")
                                
                                for news in sentiment_data['news_items']:
                                    sentiment = news['sentiment']
                                    sentiment_color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "gray"
                                    
                                    st.markdown(f"""
                                    <div style='border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                                        <h4 style='color: {sentiment_color};'>{news['title']}</h4>
                                        <p>{news['summary']}</p>
                                        <div style='display: flex; justify-content: space-between;'>
                                            <small>Source: {news['source']}</small>
                                            <small>Date: {news['date']}</small>
                                            <small>Sentiment: {sentiment:.2f}</small>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No news items available for this stock at the moment.")
                            
                            # Sentiment Distribution
                            sentiment_dist = pd.DataFrame({
                                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                                'Count': [
                                    sentiment_data['positive_count'],
                                    sentiment_data['neutral_count'],
                                    sentiment_data['negative_count']
                                ]
                            })
                            
                            fig = px.pie(
                                sentiment_dist,
                                values='Count',
                                names='Sentiment',
                                title='News Sentiment Distribution',
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': 'green',
                                    'Neutral': 'gray',
                                    'Negative': 'red'
                                },
                                hole=0.4
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Sentiment Trend
                            if 'sentiment_trend' in sentiment_data:
                                st.subheader("Sentiment Trend")
                                
                                fig = px.line(
                                    sentiment_data['sentiment_trend'],
                                    x='date',
                                    y='sentiment',
                                    title='Sentiment Trend Over Time'
                                )
                                
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig.update_layout(height=400)
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Unable to fetch sentiment data. Please try again later.")
            else:
                st.warning("No matching stocks found. Please try another search term.")
        
        # Add disclaimer
        st.info("""
        **Note on Sentiment Analysis:**
        - Sentiment scores range from -1 (very negative) to +1 (very positive)
        - Analysis is based on news headlines and summaries
        - Market sentiment may not directly correlate with stock performance
        - Consider multiple factors before making investment decisions
        """)
    
    # Log sentiment analysis activity
    if selected_symbol and selected_symbol != st.session_state.last_sentiment_search:
        activity_data = {
            'stock_symbol': selected_symbol,
            'stock_name': selected_stock_name,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_activity('sentiment_analysis', activity_data)
        st.session_state.last_sentiment_search = selected_symbol

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM model with enhanced features"""
    # Calculate additional features
    returns = np.diff(data) / data[:-1]
    
    # Align all features to have the same length
    prices = data[1:]  # Remove first element to match returns length
    
    # Calculate rolling features starting from the returns array length
    returns_series = pd.Series(returns)
    volatility = returns_series.rolling(window=lookback).std().fillna(method='bfill')
    momentum = returns_series.rolling(window=lookback).mean().fillna(method='bfill')
    
    # Combine features with aligned lengths
    features = np.column_stack([
        prices,          # Prices (n-1 length)
        returns,         # Returns (n-1 length)
        volatility,      # Volatility (n-1 length)
        momentum         # Momentum (n-1 length)
    ])
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Predict only the price
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data with validation set
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def build_lstm_model(lookback, n_features=4):
    """Build enhanced LSTM model with regularization"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.3),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Use Adam optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')  # Huber loss is more robust to outliers
    
    return model

def quantum_monte_carlo_prediction(data, n_simulations=1000, days=30):
    """Perform Quantum-inspired Monte Carlo prediction"""
    returns = np.log(data[1:] / data[:-1])
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate quantum-inspired random numbers using superposition principle
    quantum_random = np.random.normal(mu, sigma, (n_simulations, days))
    
    # Apply interference pattern
    interference = np.cos(np.linspace(0, np.pi/2, days))
    quantum_random *= interference
    
    last_price = data[-1]
    price_paths = last_price * np.exp(np.cumsum(quantum_random, axis=1))
    
    # Calculate predictions and confidence intervals
    mean_prediction = np.mean(price_paths, axis=0)
    lower_bound = np.percentile(price_paths, 5, axis=0)
    upper_bound = np.percentile(price_paths, 95, axis=0)
    
    return mean_prediction, lower_bound, upper_bound

def calculate_prediction_accuracy(actual, predicted):
    """Calculate comprehensive prediction accuracy metrics"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate directional accuracy
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

def show_predictions(data, stock_name):
    """Display stock predictions with enhanced accuracy metrics"""
    st.subheader("📈 Price Predictions")
    
    # Prepare data
    prices = data['Close'].values
    lookback = 60
    prediction_days = 30
    
    try:
        # LSTM Prediction with early stopping and validation
        with st.spinner("Training LSTM model..."):
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_lstm_data(prices, lookback)
            
            model = build_lstm_model(lookback)
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model with validation data
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions
            lstm_pred = model.predict(X_test)
            lstm_pred = scaler.inverse_transform(np.column_stack([lstm_pred, np.zeros((len(lstm_pred), 3))]))[..., 0]
            actual_test = scaler.inverse_transform(np.column_stack([y_test.reshape(-1, 1), np.zeros((len(y_test), 3))]))[..., 0]
            
            # Calculate accuracy
            lstm_accuracy = calculate_prediction_accuracy(actual_test, lstm_pred)
        
        # QMC Prediction
        with st.spinner("Performing Quantum Monte Carlo simulation..."):
            qmc_mean, qmc_lower, qmc_upper = quantum_monte_carlo_prediction(prices, days=prediction_days)
            
            # Calculate accuracy for available data
            qmc_accuracy = calculate_prediction_accuracy(
                prices[-len(qmc_mean):],
                qmc_mean[:len(prices[-len(qmc_mean):])]
            )
        
        # Display accuracy metrics with confidence scores
        st.subheader("Model Performance Metrics")
        
        # Create accuracy score based on multiple metrics
        def calculate_confidence_score(metrics):
            weights = {
                'R2': 0.3,
                'Directional_Accuracy': 0.3,
                'MAPE': 0.2,
                'MSE': 0.2
            }
            
            # Normalize MAPE and MSE (lower is better)
            normalized_mape = max(0, 100 - metrics['MAPE']) / 100
            normalized_mse = max(0, 1 - metrics['MSE'] / (prices.std() ** 2))
            
            score = (
                weights['R2'] * max(0, metrics['R2']) +
                weights['Directional_Accuracy'] * metrics['Directional_Accuracy'] / 100 +
                weights['MAPE'] * normalized_mape +
                weights['MSE'] * normalized_mse
            )
            
            return score * 100
        
        # Calculate confidence scores
        lstm_confidence = calculate_confidence_score(lstm_accuracy)
        qmc_confidence = calculate_confidence_score(qmc_accuracy)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### LSTM Model")
            st.metric("Confidence Score", f"{lstm_confidence:.1f}%")
            metrics_df = pd.DataFrame({
                'Metric': list(lstm_accuracy.keys()),
                'Value': list(lstm_accuracy.values())
            })
            st.dataframe(metrics_df.style.format({
                'Value': '{:.2f}'
            }))
        
        with col2:
            st.markdown("### QMC Model")
            st.metric("Confidence Score", f"{qmc_confidence:.1f}%")
            metrics_df = pd.DataFrame({
                'Metric': list(qmc_accuracy.keys()),
                'Value': list(qmc_accuracy.values())
            })
            st.dataframe(metrics_df.style.format({
                'Value': '{:.2f}'
            }))
        
        # Plot training history
        st.subheader("LSTM Training History")
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            line=dict(color='blue')
        ))
        fig_history.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation Loss',
            line=dict(color='red')
        ))
        fig_history.update_layout(
            title='Model Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400
        )
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Plot predictions
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index[-100:],
            y=data['Close'].values[-100:],
            name="Historical",
            line=dict(color='blue')
        ))
        
        # LSTM predictions
        future_dates = pd.date_range(
            start=data.index[-1],
            periods=len(lstm_pred),
            freq='D'
        )
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lstm_pred.flatten(),
            name="LSTM Prediction",
            line=dict(color='red', dash='dash')
        ))
        
        # QMC predictions
        future_dates_qmc = pd.date_range(
            start=data.index[-1],
            periods=len(qmc_mean),
            freq='D'
        )
        fig.add_trace(go.Scatter(
            x=future_dates_qmc,
            y=qmc_mean,
            name="QMC Prediction",
            line=dict(color='green', dash='dash')
        ))
        
        # QMC confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates_qmc.tolist() + future_dates_qmc.tolist()[::-1],
            y=np.concatenate([qmc_upper, qmc_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='QMC 90% Confidence'
        ))
        
        fig.update_layout(
            title=f"{stock_name} Price Predictions",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Summary
        st.subheader("🔍 Prediction Summary")
        
        # Calculate price changes
        lstm_change = float(((lstm_pred[-1] - prices[-1]) / prices[-1]) * 100)
        qmc_change = float(((qmc_mean[-1] - prices[-1]) / prices[-1]) * 100)
        
        summary_data = pd.DataFrame({
            'Model': ['LSTM', 'QMC'],
            'Current Price': [prices[-1], prices[-1]],
            'Predicted Price': [float(lstm_pred[-1]), float(qmc_mean[-1])],
            'Change (%)': [lstm_change, qmc_change]
        })
        
        summary_data = summary_data.round(2)
        st.dataframe(summary_data)
        
        # Prediction Consensus
        st.subheader("📊 Prediction Consensus")
        
        if lstm_change > 0 and qmc_change > 0:
            st.success("Strong Bullish: Both LSTM and QMC predict price increase")
        elif lstm_change < 0 and qmc_change < 0:
            st.error("Strong Bearish: Both LSTM and QMC predict price decrease")
        else:
            st.warning("Mixed Signals: Models show different predictions")
        
        # QMC Confidence Range
        confidence_range = float(((qmc_upper[-1] - qmc_lower[-1]) / prices[-1]) * 100)
        st.info(f"QMC 90% Confidence Range: ±{confidence_range/2:.2f}%")
        
        # Prediction disclaimer
        st.warning("""
        ⚠️ **Prediction Disclaimer:**
        - These predictions are based on historical data and mathematical models
        - Past performance does not guarantee future results
        - Multiple factors can affect stock prices
        - Always conduct thorough research before making investment decisions
        """)
    
    except Exception as e:
        st.error(f"Error in prediction calculations: {str(e)}")
        st.info("Please try again with a different stock or time period.")

# Add student investment recommendation function
def get_student_investment_recommendations(amount):
    """Generate investment recommendations for students based on available amount"""
    recommendations = {
        "Safe Options": [],
        "Moderate Options": [],
        "Growth Options": [],
        "Educational": []
    }
    
    # Safe Options (Any amount)
    recommendations["Safe Options"].append({
        "type": "Savings Account",
        "description": "Open a high-yield student savings account",
        "min_amount": 0,
        "returns": "3-4% p.a.",
        "liquidity": "High",
        "risk": "Very Low"
    })
    
    if amount >= 500:
        recommendations["Safe Options"].append({
            "type": "Recurring Deposit",
            "description": "Start a monthly recurring deposit",
            "min_amount": 500,
            "returns": "5-6% p.a.",
            "liquidity": "Medium",
            "risk": "Very Low"
        })
    
    if amount >= 1000:
        recommendations["Moderate Options"].append({
            "type": "Liquid Mutual Funds",
            "description": "Invest in liquid funds for better returns than savings account",
            "min_amount": 1000,
            "returns": "6-7% p.a.",
            "liquidity": "High",
            "risk": "Low"
        })
    
    if amount >= 5000:
        recommendations["Growth Options"].append({
            "type": "Index Fund SIP",
            "description": "Start SIP in Nifty/Sensex index funds",
            "min_amount": 5000,
            "returns": "10-12% p.a. (long term)",
            "liquidity": "Medium",
            "risk": "Moderate"
        })
        
        recommendations["Educational"].append({
            "type": "Skill Development",
            "description": "Invest in online courses or certifications",
            "min_amount": 5000,
            "returns": "Knowledge & Career Growth",
            "liquidity": "N/A",
            "risk": "Low"
        })
    
    if amount >= 10000:
        recommendations["Growth Options"].append({
            "type": "Diversified Mutual Funds",
            "description": "Invest in balanced mutual funds",
            "min_amount": 10000,
            "returns": "9-11% p.a.",
            "liquidity": "Medium",
            "risk": "Moderate"
        })
    
    return recommendations

# Main app logic
def main():
    """Main application logic"""
    # Initialize session state variables
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'completed_topics' not in st.session_state:
        st.session_state.completed_topics = set()
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'last_stock_search' not in st.session_state:
        st.session_state.last_stock_search = None
    if 'last_sentiment_search' not in st.session_state:
        st.session_state.last_sentiment_search = None
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .subheader {
            color: #2c3e50;
            font-size: 1.8rem;
            font-weight: bold;
            margin: 1.5rem 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    if not st.session_state.is_authenticated:
        show_onboarding()
    else:
        # Get user name from session state
        user_name = st.session_state.user_data.get('name', 'User')
        
        st.sidebar.title(f"Welcome, {user_name}!")
        st.sidebar.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3", width=250)
        
        # Updated Navigation
        pages = {
            "Financial Literacy": show_financial_literacy,
            "Stock Analysis": show_stock_analysis,
            "Portfolio Optimization": show_portfolio_optimization,
            "Mutual Fund Analysis": mutual_fund_analysis,  # Add new page
            "SIP Calculator": sip_calculator,  # Add new page
            "Sentiment Analysis": show_sentiment_analysis
        }
        
        selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))
        
        # Display selected page
        pages[selected_page]()
        
        # Logout option
        if st.sidebar.button("Logout"):
            st.session_state.is_authenticated = False
            st.session_state.user_data = {}
            st.rerun()

if __name__ == "__main__":
    main() 