# config.py
"""
Configuration file for the Streamlit Financial Dashboard.
Contains default values, constants, API key management, and example data.
"""
import streamlit as st
import pandas as pd
from fredapi import Fred 
from datetime import datetime, timedelta

# --- General App Configuration ---
DEFAULT_ROLLING_WINDOW_CORR = 30 
MIN_ROLLING_WINDOW_CORR = 1
MAX_ROLLING_WINDOW_CORR = 250
YEARS_OF_DATA_CORR = 10  

FED_JAWS_DURATION_DAYS = 120 
FFR_PCE_THRESHOLD = 2.0 

# --- Default Ticker Symbols ---
DEFAULT_TICKER_1_CORR = 'EURUSD=X'
DEFAULT_TICKER_2_CORR = 'GLD'
DEFAULT_TICKER_SKEW = 'AAPL'
DEFAULT_FRED_SERIES_NAME = "Effective Federal Funds Rate (Daily)"

# --- FRED API Key Configuration & Client Initialization ---
FRED_API_KEY = None
fred = None
fred_error_message = None

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except (KeyError, FileNotFoundError): 
    fred_error_message = "FRED API Key not found in Streamlit Secrets. Please create/check `.streamlit/secrets.toml` and add: FRED_API_KEY = \"YOUR_API_KEY_HERE\" to use FRED data features."
    print(fred_error_message) 
if FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        print("FRED client initialized successfully in config.py.") 
    except Exception as e:
        fred_error_message = f"Failed to initialize FRED API client: {e}. Check your API key and its validity."
        print(fred_error_message) 
        fred = None 
        FRED_API_KEY = None 
else:
    if not fred_error_message: 
        fred_error_message = "FRED API Key is not configured. Please set it in Streamlit Secrets to use FRED data features."
    print(fred_error_message) 


# --- Ticker Examples Data ---
ticker_examples = {
    "Equity Indices": {"S&P 500 (US)": "^GSPC", "Nasdaq Composite (US)": "^IXIC", "Dow Jones Industrial Avg (US)": "^DJI", "FTSE 100 (UK)": "^FTSE", "DAX (Germany)": "^GDAXI", "Nikkei 225 (Japan)": "^N225", "Straits Times Index (SG)": "^STI"},
    "Forex (vs USD)": {"Euro": "EURUSD=X", "British Pound": "GBPUSD=X", "Japanese Yen (USD/JPY)": "JPY=X", "Australian Dollar": "AUDUSD=X", "Singapore Dollar (USD/SGD)": "SGD=X", "Canadian Dollar (USD/CAD)": "CAD=X"},
    "Commodities": {"Gold Futures": "GC=F", "Crude Oil Futures (WTI)": "CL=F", "Silver Futures": "SI=F", "Copper Futures": "HG=F", "Natural Gas Futures": "NG=F", "SPDR Gold Shares ETF": "GLD", "US Oil Fund ETF": "USO"},
    "Cryptocurrencies (vs USD)": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD"},
    "US Treasury Yields": {"10 Year (^TNX)": "^TNX", "5 Year (^FVX)": "^FVX", "30 Year (^TYX)": "^TYX"},
    "Example Stocks":{"Apple (US)": "AAPL", "Microsoft (US)": "MSFT", "Google (US)": "GOOGL", "DBS Group (SG)": "D05.SI", "Tesla (US)": "TSLA"}
}
ticker_data_list = []
for category, tickers_in_category in ticker_examples.items():
    for name, symbol in tickers_in_category.items():
        ticker_data_list.append({"Asset Class": category, "Description": name, "Yahoo Ticker": symbol})
ticker_df = pd.DataFrame(ticker_data_list)
ticker_df = ticker_df.sort_values(by=["Asset Class", "Description"]).reset_index(drop=True)

# --- FRED Series Definitions ---
FRED_SERIES_EXAMPLES = {
    "Effective Federal Funds Rate (Daily)": "DFF", "Nominal GDP (Quarterly)": "GDP", "Real GDP (Quarterly)": "GDPC1",
    "CPI - All Urban Consumers (Monthly)": "CPIAUCSL", "Core CPI (Less Food & Energy, Monthly)": "CPILFESL",
    "PCE Price Index (Monthly)": "PCEPI", "Core PCE Price Index (Monthly)": "PCEPILFE", 
    "Core PCE Year-on-Year (Monthly)": "PCEPILFECHPYA", "Unemployment Rate (Monthly)": "UNRATE",
    "Initial Claims (Weekly)": "ICSA", "10-Year Treasury Constant Maturity Rate (Daily)": "DGS10",
    "M1 Money Stock (Weekly)": "WM1NS", "M2 Money Stock (Weekly)": "WM2NS",
    "Industrial Production Index (Monthly)": "INDPRO", "Retail Sales - Total (Monthly)": "RSAFS",
    "Gold Price (London Bullion, Daily)": "GOLDAMGBD228NLBM", "VIX (Volatility Index, Daily)": "VIXCLS",
    "Effective Federal Funds Rate (Monthly Avg)": "FEDFUNDS",
    "10-Year Treasury Inflation-Indexed Security (Daily)": "DFII10", # For Real Yield
    "NBER based Recession Indicators (Monthly)": "USRECM" # NBER Recession Indicators (USRECM for monthly)
}
fred_series_options = list(FRED_SERIES_EXAMPLES.keys())

USREC_SERIES_ID = "USRECM" # NBER based Recession Indicators for the United States (Monthly)

FED_JAWS_SERIES_IDS = ['DFEDTARU', 'IORB', 'DPCREDIT', 'SOFR', 'DFF', 'OBFR', 'DFEDTARL']

FFR_VS_PCE_SERIES_IDS = {"ffr": "FEDFUNDS", "core_pce_index": "PCEPILFE"}
FFR_VS_PCE_NAMES = {"ffr": "Effective Federal Funds Rate (Monthly)", "core_pce_index": "Core PCE Price Index (Monthly)", "core_pce_yoy_calculated": "Core PCE Inflation YoY (Calculated)"}

GOLD_VS_REAL_YIELD_SERIES_IDS = {"gold": "GOLDAMGBD228NLBM", "real_yield_10y": "DFII10"}
GOLD_VS_REAL_YIELD_NAMES = {"gold": "Gold Price (London Bullion, Daily)", "real_yield_10y": "10-Year Treasury Inflation-Indexed Security (Daily)"}

# Default date range for FRED charts (e.g., last 20 years)
DEFAULT_FRED_START_DATE = datetime.now() - timedelta(days=20*365)
DEFAULT_FRED_END_DATE = datetime.now()

# --- LinkedIn Footer Configuration ---
LINKEDIN_URL = "https://www.linkedin.com/in/kennethquah/"
YOUR_NAME = "Kenneth Quah"
LINKEDIN_SVG = """<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#0077B5" style="vertical-align: middle;"><title>LinkedIn</title><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>"""
