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
import config 

@st.cache_data
def calculate_rolling_correlation(ticker1, ticker2, window, years=config.YEARS_OF_DATA_CORR):
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2} with window {window}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + window + 50) 
    try:
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)
        if data is None or data.empty: return None, f"Failed to download data for {ticker1} or {ticker2}."
        if 'Close' not in data.columns: return None, "Could not find 'Close' price column."
        close_data = data.get('Close', pd.DataFrame())
        if close_data.empty: return None, f"Could not retrieve 'Close' price data."
        close_data.ffill(inplace=True)
        missing_tickers = [t for t in [ticker1, ticker2] if t not in close_data.columns or close_data[t].isnull().all()]
        if missing_tickers: return None, f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}."
        returns = close_data.pct_change().dropna()
        if not (ticker1 in returns.columns and ticker2 in returns.columns) or returns.empty: return None, f"Could not calculate returns."
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all(): return None, f"Returns data contains only NaNs."
        if len(returns) < window: return pd.DataFrame(), f"Not enough data points ({len(returns)}) for the {window}-day window." 
        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()
        rolling_corr.name = f'{window}d Rolling Corr'
        return rolling_corr, None 
    except Exception as e:
        print(f"Error calculating rolling correlation for {ticker1}/{ticker2}: {e}\n{traceback.format_exc()}")
        return None, f"An error occurred: {e}"

@st.cache_data
def get_expiration_dates(ticker):
    if not ticker: return None, "Ticker symbol cannot be empty."
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        return list(expirations) if expirations else None, None if expirations else f"No options dates found for {ticker}."
    except Exception as e:
        return None, f"Error fetching expirations for {ticker}: {e}"

@st.cache_data
def get_option_chain_data(ticker, expiry_date):
    if not ticker or not expiry_date: return None, None, None, "Ticker or expiration date missing."
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry_date)
        info = stock.info
        price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))
        if not (chain.calls.empty and chain.puts.empty): return chain.calls, chain.puts, price, None
        return None, None, price, f"No options data for {ticker} on {expiry_date}."
    except Exception as e:
        return None, None, None, f"Error fetching option chain for {ticker} on {expiry_date}: {e}"

@st.cache_data(show_spinner=False)
def get_fred_data(_fred_instance, series_id, start_date=None, end_date=None):
    if not _fred_instance: return None, "FRED API client not initialized."
    if not series_id: return None, "FRED Series ID cannot be empty."
    try:
        print(f"Fetching FRED data for {series_id} from {start_date} to {end_date}")
        data = _fred_instance.get_series(series_id, observation_start=start_date, observation_end=end_date)
        data = data.dropna() 
        print(f"Fetched FRED data for {series_id}. Shape: {data.shape}")
        return data, None if not data.empty else f"No data returned for {series_id} in range."
    except Exception as e:
        return None, f"Failed to fetch FRED series '{series_id}': {e}"

@st.cache_data(show_spinner=False)
def get_fred_series_info(_fred_instance, series_id):
    if not _fred_instance: return None, "FRED API client not initialized."
    if not series_id: return None, "FRED Series ID cannot be empty."
    try:
        return _fred_instance.get_series_info(series_id), None
    except Exception as e:
        return None, f"Failed to fetch metadata for FRED series '{series_id}': {e}"

@st.cache_data(show_spinner=False)
def get_multiple_fred_data(_fred_instance, series_ids, start_date=None, end_date=None, include_recession_bands=False):
    if not _fred_instance: return None, "FRED API client not initialized."
    if not series_ids: return None, "No FRED Series IDs provided."
    all_series_data = {}; errors = {}
    current_series_ids = list(series_ids) 
    if include_recession_bands and config.USREC_SERIES_ID not in current_series_ids:
        current_series_ids.append(config.USREC_SERIES_ID)
    for series_id in current_series_ids:
        try:
            print(f"Fetching FRED: {series_id} from {start_date} to {end_date}")
            s_data = _fred_instance.get_series(series_id, observation_start=start_date, observation_end=end_date)
            if not s_data.empty: all_series_data[series_id] = s_data
            else: print(f"Warning: No data for {series_id} in range.")
        except Exception as e: errors[series_id] = str(e); print(f"Error fetching {series_id}: {e}")
    if not all_series_data: return None, f"Failed to fetch data for any requested series. Errors: {errors if errors else 'Unknown'}"
    try:
        combined_df = pd.concat(all_series_data, axis=1, join='outer')
        combined_df.ffill(inplace=True); combined_df.bfill(inplace=True) 
        print(f"Combined FRED data shape after fill: {combined_df.shape}")
        return combined_df, f"Partial success. Errors: {errors}" if errors else None
    except Exception as e: return None, f"Error combining/filling FRED data: {e}"

@st.cache_data(show_spinner=True) # Show spinner for this yfinance call
def get_index_valuation_ratios(ticker_symbol):
    """Fetches current valuation ratios for a given index ticker using yfinance .info."""
    if not ticker_symbol:
        return None, "Ticker symbol cannot be empty."
    try:
        print(f"Fetching .info for index ticker: {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Check if info was successfully fetched
        if not info or info.get('regularMarketPrice') is None: # A basic check if info is populated
             return None, f"Could not retrieve detailed info for {ticker_symbol}. It might be an invalid ticker or data is temporarily unavailable."

        ratios = {
            "Symbol": ticker_symbol,
            "Name": info.get('shortName', info.get('longName', ticker_symbol)),
            "Trailing P/E": info.get('trailingPE'),
            "Forward P/E": info.get('forwardPE'),
            "Price/Book": info.get('priceToBook'),
            "Dividend Yield": info.get('dividendYield')
        }
        # Convert dividend yield to percentage string if it exists
        if ratios["Dividend Yield"] is not None:
            ratios["Dividend Yield"] = f"{ratios['Dividend Yield'] * 100:.2f}%"
            
        print(f"Successfully fetched valuation ratios for {ticker_symbol}.")
        return ratios, None
    except Exception as e:
        print(f"Error fetching valuation ratios for {ticker_symbol}: {e}")
        print(traceback.format_exc())
        return None, f"An error occurred while fetching valuation ratios for {ticker_symbol}: {e}. Please check the ticker symbol."
