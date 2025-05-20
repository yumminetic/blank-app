# data_fetchers.py
"""
Data fetching functions for the Streamlit Financial Dashboard.
Uses yfinance for market data and fredapi for economic data.
"""
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import traceback
import numpy as np
# No direct fredapi import here, the 'fred' object is passed in

# Import constants from config
from config import ROLLING_WINDOW, YEARS_OF_DATA # Used by calculate_rolling_correlation

@st.cache_data
def calculate_rolling_correlation(ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Calculates rolling correlation (uses yfinance)."""
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + window + 50) # Buffer
    try:
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            # st.error(f"Failed to download data for {ticker1} or {ticker2}.") # UI feedback in app.py
            return None, f"Failed to download data for {ticker1} or {ticker2}."
        if 'Close' not in data.columns:
            # st.error("Could not find 'Close' price column in downloaded data.")
            return None, "Could not find 'Close' price column in downloaded data."

        close_data = data.get('Close', pd.DataFrame())
        if close_data.empty:
            # st.warning(f"Could not retrieve 'Close' price data for {ticker1} or {ticker2}.")
            return None, f"Could not retrieve 'Close' price data for {ticker1} or {ticker2}."

        close_data.ffill(inplace=True)
        missing_tickers = []
        if ticker1 not in close_data.columns or close_data[ticker1].isnull().all():
            missing_tickers.append(ticker1)
        if ticker2 not in close_data.columns or close_data[ticker2].isnull().all():
            missing_tickers.append(ticker2)
        if missing_tickers:
            # st.error(f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}.")
            return None, f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}."

        returns = close_data.pct_change().dropna()
        if ticker1 not in returns.columns or ticker2 not in returns.columns or returns.empty:
            # st.warning(f"Could not calculate returns for {ticker1} or {ticker2}.")
            return None, f"Could not calculate returns for {ticker1} or {ticker2}."
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all():
            # st.warning(f"Returns data contains only NaNs for {ticker1} or {ticker2}.")
            return None, f"Returns data contains only NaNs for {ticker1} or {ticker2}."
        if len(returns) < window:
            # st.warning(f"Not enough data points ({len(returns)}) for the specified {window}-day rolling window.")
            return None, f"Not enough data points ({len(returns)}) for the specified {window}-day rolling window."

        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()
        rolling_corr.name = f'{window}d Rolling Corr'
        print(f"Successfully calculated rolling correlation. Shape: {rolling_corr.shape}")
        return rolling_corr, None # Data, No error message
    except Exception as e:
        print(f"Error calculating rolling correlation for {ticker1}/{ticker2}: {e}")
        print(traceback.format_exc())
        # st.error(f"An error occurred while calculating correlation for {ticker1}/{ticker2}. Check ticker symbols and data availability.")
        return None, f"An error occurred while calculating correlation for {ticker1}/{ticker2}: {e}"

@st.cache_data
def get_expiration_dates(ticker):
    """Fetches option expiration dates for a given ticker using yfinance."""
    if not ticker:
        return None, "Ticker symbol cannot be empty."
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, f"No options expiration dates found for ticker: {ticker}."
        return list(expirations), None
    except Exception as e:
        print(f"Error fetching expirations for {ticker}: {e}")
        return None, f"An error occurred while fetching expiration dates for {ticker}. It might be an invalid ticker or a temporary issue: {e}"

@st.cache_data
def get_option_chain_data(ticker, expiry_date):
    """Fetches the option chain (calls and puts) and current price for a ticker and expiry date."""
    if not ticker or not expiry_date:
        return None, None, None, "Ticker symbol or expiration date is missing."
    try:
        print(f"Fetching option chain for {ticker} expiring {expiry_date}...")
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry_date)
        calls_df = chain.calls
        puts_df = chain.puts
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))

        if current_price is None or not isinstance(current_price, (int, float, np.number)):
            print(f"Warning: Could not determine a valid current price for {ticker}. Price data might be delayed or unavailable.")
            current_price = None
        if calls_df.empty and puts_df.empty:
            return None, None, current_price, f"No options data found for {ticker} on {expiry_date}. The ticker might not have options or data is unavailable for this date."
        print(f"Fetched chain for {ticker} {expiry_date}. Calls: {len(calls_df)}, Puts: {len(puts_df)}, Price: {current_price}")
        return calls_df, puts_df, current_price, None
    except Exception as e:
        print(f"Error fetching option chain for {ticker} on {expiry_date}: {e}")
        print(traceback.format_exc())
        return None, None, None, f"An error occurred fetching the option chain for {ticker} on {expiry_date}: {e}"

@st.cache_data(show_spinner=False)
def get_fred_data(_fred_instance, series_id):
    """Fetches time series data for a given FRED series ID."""
    if not _fred_instance:
        return None, "FRED API client is not initialized. Check API key configuration."
    if not series_id:
        return None, "FRED Series ID cannot be empty."
    try:
        print(f"Fetching FRED data for series ID: {series_id}...")
        data = _fred_instance.get_series(series_id)
        data = data.dropna() # Remove trailing NaNs
        if data.empty:
            print(f"Warning: FRED data for {series_id} is empty after removing NaNs.")
            # return None, f"No valid data found for FRED series '{series_id}'." # Optional: treat as error
        print(f"Successfully fetched FRED data for {series_id}. Shape: {data.shape}")
        return data, None
    except Exception as e:
        error_msg = f"Failed to fetch data for FRED series '{series_id}': {e}"
        print(error_msg)
        print(traceback.format_exc())
        return None, f"Could not fetch data for FRED series '{series_id}'. Verify ID and API connection: {e}"

@st.cache_data(show_spinner=False)
def get_fred_series_info(_fred_instance, series_id):
    """Fetches metadata/information for a specific FRED series ID."""
    if not _fred_instance:
        return None, "FRED API client is not initialized."
    if not series_id:
        return None, "FRED Series ID cannot be empty."
    try:
        print(f"Fetching FRED series info for: {series_id}...")
        info = _fred_instance.get_series_info(series_id)
        print(f"Successfully fetched info for FRED series {series_id}.")
        return info, None
    except Exception as e:
        error_msg = f"Failed to fetch metadata for FRED series '{series_id}': {e}"
        print(error_msg)
        return None, f"Could not fetch metadata for FRED series '{series_id}': {e}"

@st.cache_data(show_spinner=False)
def get_multiple_fred_data(_fred_instance, series_ids, start_date=None, end_date=None):
    """Fetches data for multiple FRED series IDs, merges, and forward fills them."""
    if not _fred_instance:
        return None, "FRED API client is not initialized."
    if not series_ids:
        return None, "No FRED Series IDs provided."

    all_series_data = {}
    errors = {}
    if start_date and end_date:
        print(f"Fetching FRED multi-series data from {start_date.date()} to {end_date.date()}")
    else:
        print("Fetching all available FRED multi-series data (no date range specified).")

    for series_id in series_ids:
        try:
            print(f"Fetching FRED data for series: {series_id}...")
            s_data = _fred_instance.get_series(series_id, observation_start=start_date, observation_end=end_date) if start_date and end_date else _fred_instance.get_series(series_id)
            if not s_data.empty:
                all_series_data[series_id] = s_data
                print(f"Successfully fetched {series_id}, shape: {s_data.shape}")
            else:
                print(f"Warning: No data returned for {series_id} in the specified range/period.")
        except Exception as e:
            error_msg = f"Failed to fetch {series_id}: {e}"
            print(error_msg)
            errors[series_id] = error_msg

    if not all_series_data:
        error_details = f"Errors: {errors}" if errors else "Unknown reason."
        return None, f"Failed to fetch data for *any* of the requested series. {error_details}"
    try:
        combined_df = pd.concat(all_series_data, axis=1, join='outer')
        combined_df.ffill(inplace=True)
        print("Applied forward fill (ffill) to combined FRED data.")
        combined_df.dropna(how='all', inplace=True) # Optional: drop rows if all are NaN after ffill
        print(f"Combined FRED data shape after ffill: {combined_df.shape}")
        if errors: # Return errors as part of a successful fetch of *some* data
            return combined_df, f"Could not fetch data for some series: {errors}"
        return combined_df, None
    except Exception as e:
        print(f"Error combining or filling FRED series: {e}")
        print(traceback.format_exc())
        return None, f"Error occurred while combining or filling FRED data: {e}"

