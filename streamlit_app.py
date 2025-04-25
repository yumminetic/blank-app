# --- Imports ---
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback # For detailed error printing
import numpy as np # Needed for checking numeric types
from fredapi import Fred # Import FRED API library

# --- Configuration ---
DEFAULT_TICKER_1_CORR = 'EURUSD=X'
DEFAULT_TICKER_2_CORR = 'GLD'
DEFAULT_TICKER_SKEW = 'AAPL' # Default for skew plot
ROLLING_WINDOW = 30
YEARS_OF_DATA = 10

# --- FRED API Key Configuration ---
try:
    FRED_API_KEY = "55934e55334d8efd02d713b44a8aed3b"
except (KeyError, FileNotFoundError): # Handle missing file/key
    FRED_API_KEY = None # Set to None if not found

# Instantiate Fred object (only if key is available)
fred = None
fred_error = None
if FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
    except Exception as e:
        fred_error = f"Failed to initialize FRED API: {e}. Check your API key."
        FRED_API_KEY = None # Disable further attempts
else:
     # No key provided at all
     fred_error = "FRED API Key not found. Please configure it in Streamlit Secrets (`.streamlit/secrets.toml`) to use the FRED data section."

# --- Data Fetching Functions ---
@st.cache_data
def calculate_rolling_correlation(ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Calculates rolling correlation (uses yfinance)."""
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2}...")
    # ... (Keep the robust implementation from previous version) ...
    end_date = datetime.now(); start_date = end_date - timedelta(days=years*365 + window + 50)
    try:
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)
        if data is None or data.empty or 'Close' not in data.columns: return None
        close_data = data.get('Close', pd.DataFrame());
        if close_data.empty: return None
        close_data.ffill(inplace=True)
        missing_tickers = [];
        if ticker1 not in close_data.columns or close_data[ticker1].isnull().all(): missing_tickers.append(ticker1)
        if ticker2 not in close_data.columns or close_data[ticker2].isnull().all(): missing_tickers.append(ticker2)
        if missing_tickers: st.error(f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}."); return None
        returns = close_data.pct_change().dropna();
        if ticker1 not in returns.columns or ticker2 not in returns.columns or returns.empty: return None
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all(): return None
        if len(returns) < window: return None
        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna();
        rolling_corr.name = f'{window}d Rolling Corr';
        print(f"Successfully calculated rolling correlation. Shape: {rolling_corr.shape}"); return rolling_corr
    except Exception as e: print(f"Error in correlation: {e}"); print(traceback.format_exc()); st.error(f"Error processing correlation for {ticker1}/{ticker2}."); return None

@st.cache_data
def get_expiration_dates(ticker):
    """Fetches option expiration dates (uses yfinance)."""
    if not ticker: return None, "Ticker is empty."
    try:
        expirations = yf.Ticker(ticker).options
        return list(expirations), None if expirations else (None, f"No options found for {ticker}.")
    except Exception as e: return None, f"Error fetching expirations for {ticker}."

@st.cache_data
def get_option_chain_data(ticker, expiry_date):
    """Fetches option chain data and price (uses yfinance)."""
    if not ticker or not expiry_date: return None, None, None, "Ticker or expiry date missing."
    try:
        print(f"Fetching option chain for {ticker} expiring {expiry_date}...")
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry_date)
        calls_df = chain.calls; puts_df = chain.puts
        info = stock.info; current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))
        if current_price is None or not isinstance(current_price, (int, float)): current_price = None
        if calls_df.empty and puts_df.empty: return None, None, current_price, f"No options data for {ticker} on {expiry_date}."
        print(f"Fetched chain for {ticker} {expiry_date}. Calls: {len(calls_df)}, Puts: {len(puts_df)}, Price: {current_price}")
        return calls_df, puts_df, current_price, None
    except Exception as e: print(f"Error fetching option chain: {e}"); print(traceback.format_exc()); return None, None, None, f"Error fetching chain for {ticker} on {expiry_date}."

@st.cache_data(show_spinner=False) # Using spinner outside function
def get_fred_data(_fred_instance, series_id):
    """Fetches data for a given FRED series ID."""
    # Note: _fred_instance is passed to help caching recognize changes if the instance were mutable, though unlikely here.
    if not _fred_instance: return None, "FRED API client not initialized."
    if not series_id: return None, "FRED Series ID is empty."
    try:
        print(f"Fetching FRED data for {series_id}...")
        data = _fred_instance.get_series(series_id)
        data = data.dropna() # Remove trailing NaNs often present
        print(f"Successfully fetched FRED data for {series_id}. Shape: {data.shape}")
        return data, None
    except Exception as e:
        error_msg = f"Failed to fetch data for FRED series '{series_id}': {e}"
        print(error_msg)
        print(traceback.format_exc())
        return None, error_msg

@st.cache_data(show_spinner=False)
def get_fred_series_info(_fred_instance, series_id):
    """Fetches metadata/information for a FRED series ID."""
    if not _fred_instance: return None, "FRED API client not initialized."
    if not series_id: return None, "FRED Series ID is empty."
    try:
        info = _fred_instance.get_series_info(series_id)
        return info, None
    except Exception as e:
        error_msg = f"Failed to fetch info for FRED series '{series_id}': {e}"
        print(error_msg)
        return None, error_msg


# --- Plotting Functions ---
def create_corr_plot(rolling_corr, ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Creates the Plotly figure for rolling correlation."""
    # ... (Keep implementation from previous version) ...
    fig = go.Figure(); plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"
    if rolling_corr is not None and not rolling_corr.empty:
        pos_corr = rolling_corr.copy(); pos_corr[pos_corr < 0] = None; neg_corr = rolling_corr.copy(); neg_corr[neg_corr >= 0] = None
        fig.add_trace(go.Scatter(x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation', line=dict(color='red')))
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    elif rolling_corr is not None and rolling_corr.empty: error_message = f"No correlation data available for {ticker1} / {ticker2}."; fig.update_layout(title=plot_title, annotations=[dict(text=error_message, showarrow=False, align='center')])
    else: error_message = f"Could not load or process data for {ticker1} / {ticker2}."; fig.update_layout(title=plot_title, annotations=[dict(text=error_message, showarrow=False, align='center')])
    if rolling_corr is None or rolling_corr.empty: fig.update_layout(xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500)
    return fig

def create_iv_skew_plot(calls_df, puts_df, ticker, expiry_date, current_price):
    """Creates the Plotly figure for IV Skew."""
    # ... (Keep implementation from previous version) ...
    fig = go.Figure(); plot_title = f"{ticker} Implied Volatility Skew (Expiry: {expiry_date})"; data_plotted = False
    if calls_df is not None and not calls_df.empty and 'strike' in calls_df.columns and 'impliedVolatility' in calls_df.columns:
        calls_df = calls_df[pd.to_numeric(calls_df['strike'], errors='coerce').notna()]; calls_df = calls_df[pd.to_numeric(calls_df['impliedVolatility'], errors='coerce').notna()]
        if not calls_df.empty: fig.add_trace(go.Scatter(x=calls_df['strike'], y=calls_df['impliedVolatility'] * 100, mode='markers+lines', name='Calls IV (%)', marker=dict(color='blue'), line=dict(color='blue'))); data_plotted = True
    if puts_df is not None and not puts_df.empty and 'strike' in puts_df.columns and 'impliedVolatility' in puts_df.columns:
        puts_df = puts_df[pd.to_numeric(puts_df['strike'], errors='coerce').notna()]; puts_df = puts_df[pd.to_numeric(puts_df['impliedVolatility'], errors='coerce').notna()]
        if not puts_df.empty: fig.add_trace(go.Scatter(x=puts_df['strike'], y=puts_df['impliedVolatility'] * 100, mode='markers+lines', name='Puts IV (%)', marker=dict(color='orange'), line=dict(color='orange'))); data_plotted = True
    fig.update_layout(title=plot_title, xaxis_title='Strike Price', yaxis_title='Implied Volatility (%)', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if current_price is not None: fig.add_vline(x=current_price, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Current Price: {current_price:.2f}", annotation_position="top right")
    if not data_plotted: error_message = f"No valid options IV data found for {ticker} expiring {expiry_date}."; fig.add_annotation(text=error_message, showarrow=False, align='center'); fig.update_layout(yaxis_range=[0, 100])
    return fig

def create_fred_plot(series_data, series_id, series_info):
    """Creates the Plotly figure for a FRED time series."""
    fig = go.Figure()
    plot_title = f"FRED Series: {series_id}"
    y_axis_label = "Value"

    if series_info is not None:
        plot_title = series_info.get('title', plot_title)
        y_axis_label = series_info.get('units_short', y_axis_label) + f" ({series_info.get('frequency_short', '')}, {series_info.get('seasonal_adjustment_short', 'NSA')})"


    if series_data is not None and not series_data.empty:
        fig.add_trace(go.Scatter(
            x=series_data.index,
            y=series_data.values,
            mode='lines', name=series_id
        ))
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title=y_axis_label, height=500)