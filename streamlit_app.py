import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback # For detailed error printing

# --- Configuration ---
DEFAULT_TICKER_1 = 'EURUSD=X'
DEFAULT_TICKER_2 = 'GLD'
ROLLING_WINDOW = 30
YEARS_OF_DATA = 10

# --- Data fetching and calculation function (Cached) ---
@st.cache_data # Cache results based on input arguments
def calculate_rolling_correlation(ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """
    Fetches data for two tickers and calculates their rolling correlation using 'Close' price.
    Returns the correlation Series or None if data is insufficient or an error occurs
    AFTER successful initial download.
    May raise exceptions if yf.download fails fundamentally (e.g., invalid ticker).
    """
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2} ({years} years, window {window})...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + window + 50)

    # Let yf.download handle potential errors for invalid tickers initially
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)

    if data is None or data.empty or 'Close' not in data.columns:
         print(f"Error: Download failed or 'Close' column missing for {ticker1}/{ticker2}.")
         pass

    close_data = data.get('Close', pd.DataFrame())
    if close_data.empty:
        print(f"Error: 'Close' price data is empty for {ticker1}/{ticker2}.")
        st.warning(f"Could not retrieve 'Close' price data for {ticker1} or {ticker2}.")
        return None

    close_data.ffill(inplace=True)

    missing_tickers = []
    if ticker1 not in close_data.columns or close_data[ticker1].isnull().all():
        missing_tickers.append(ticker1)
    if ticker2 not in close_data.columns or close_data[ticker2].isnull().all():
        missing_tickers.append(ticker2)

    if missing_tickers:
         print(f"Error: Missing or all-null data for ticker(s): {', '.join(missing_tickers)} after ffill.")
         st.error(f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}. Please check symbol on Yahoo Finance.")
         return None

    try:
        returns = close_data.pct_change().dropna()

        if ticker1 not in returns.columns or ticker2 not in returns.columns or returns.empty:
             print(f"Error: Returns calculation failed.")
             st.warning(f"Could not calculate returns for {ticker1} or {ticker2}.")
             return None
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all():
             print(f"Error: All returns are NaN.")
             st.warning(f"Could not calculate returns for {ticker1} or {ticker2}.")
             return None
        if len(returns) < window:
             print(f"Error: Not enough data ({len(returns)}) for window ({window}).")
             st.warning(f"Not enough data points ({len(returns)}) for {window}-day rolling window after processing.")
             return None

        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()

        if rolling_corr.empty:
             print(f"Warning: Rolling correlation is empty after calculation (window={window}).")

        rolling_corr.name = f'{window}d Rolling Corr'
        print(f"Successfully calculated rolling correlation. Shape: {rolling_corr.shape}")
        return rolling_corr

    except Exception as e:
        print(f"An error occurred during calculation: {e}")
        print(traceback.format_exc())
        st.error(f"An unexpected error occurred during data calculation.")
        return None


# --- Function to create the plot ---
def create_corr_plot(rolling_corr, ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Creates the Plotly figure based on the correlation data."""
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"

    if rolling_corr is not None and not rolling_corr.empty:
        pos_corr = rolling_corr.copy()
        pos_corr[pos_corr < 0] = None
        neg_corr = rolling_corr.copy()
        neg_corr[neg_corr >= 0] = None

        fig.add_trace(go.Scatter(
            x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation', line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation', line=dict(color='red')
        ))
        fig.update_layout(
            title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    elif rolling_corr is not None and rolling_corr.empty:
         error_message = f"No correlation data available for {ticker1} / {ticker2} in the selected period."
         fig.update_layout(
            title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
         )
    else: # Handles rolling_corr is None
        error_message = f"Could not load or process data for {ticker1} / {ticker2}. Check symbols or data availability."
        fig.update_layout(
            title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
        )
    return fig

# --- Ticker Examples ---
ticker_examples = {
    "Equity Indices": {
        "S&P 500 (US)": "^GSPC", "Nasdaq Composite (US)": "^IXIC", "Dow Jones Industrial Avg (US)": "^DJI",
        "FTSE 100 (UK)": "^FTSE", "DAX (Germany)": "^GDAXI", "Nikkei 225 (Japan)": "^N225",
        "Straits Times Index (SG)": "^STI",
    },
    "Forex (vs USD)": {
        "Euro": "EURUSD=X", "British Pound": "GBPUSD=X", "Japanese Yen (USD/JPY)": "JPY=X",
        "Australian Dollar": "AUDUSD=X", "Singapore Dollar (USD/SGD)": "SGD=X", "Canadian Dollar (USD/CAD)": "CAD=X",
    },
    "Commodities": {
        "Gold Futures": "GC=F", "Crude Oil Futures (WTI)": "CL=F", "Silver Futures": "SI=F",
        "Copper Futures": "HG=F", "Natural Gas Futures": "NG=F", "SPDR Gold Shares ETF": "GLD",
        "US Oil Fund ETF": "USO",
    },
    "Cryptocurrencies (vs USD)": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD",
    },
     "US Treasury Yields": {
        "10 Year (^TNX)": "^TNX", "5 Year (^FVX)": "^FVX", "30 Year (^TYX)": "^TYX",
    },
    "Example Stocks":{
        "Apple (US)": "AAPL", "Microsoft (US)": "MSFT", "Google (US)": "GOOGL",
        "DBS Group (SG)": "D05.SI", "Tesla (US)": "TSLA",
    }
}

# --- Streamlit App Layout ---
st.set_page_config(page_title="Rolling Correlation", layout="wide")
st.title("Macro Dashboard")

# Initialize Session State
if 'rolling_corr_data' not in st.session_state:
    st.session_state.rolling_corr_data = None
if 'ticker1_calculated' not in st.session_state:
    st.session_state.ticker1_calculated = None
if 'ticker2_calculated' not in st.session_state:
    st.session_state.ticker2_calculated = None

# --- Inputs Area ---
col1, col2 = st.columns(2)
with col1:
    ticker1_input = st.text_input("Enter Ticker 1:",
                                  value=st.session_state.get('ticker1_calculated', DEFAULT_TICKER_1),
                                  key="ticker1_widget")
with col2:
    ticker2_input = st.text_input("Enter Ticker 2:",
                                  value=st.session_state.get('ticker2_calculated', DEFAULT_TICKER_2),
                                  key="ticker2_widget")

calculate_button = st.button("Calculate Correlation", type="primary")

# Placeholder for the plot
plot_placeholder = st.empty()

# --- Calculation and Plotting Logic (Triggered by Button) ---
if calculate_button:
    t1 = ticker1_input.strip().upper()
    t2 = ticker2_input.strip().upper()

    if t1 and t2:
        with st.spinner(f"Calculating {ROLLING_WINDOW}-day rolling correlation for {t1} and {t2}..."):
            try:
                result = calculate_rolling_correlation(t1, t2, window=ROLLING_WINDOW, years=YEARS_OF_DATA)
                st.session_state.rolling_corr_data = result
                st.session_state.ticker1_calculated = t1
                st.session_state.ticker2_calculated = t2
            except Exception as e:
                print(f"Error calling calculate_rolling_correlation for {t1}/{t2}: {e}")
                print(traceback.format_exc())
                st.error(f"Failed to download or process data for {t1} / {t2}. Please verify ticker symbols on Yahoo Finance.")
                st.session_state.rolling_corr_data = None
                st.session_state.ticker1_calculated = t1
                st.session_state.ticker2_calculated = t2
    # Handle cases where one or both inputs are empty after button click
    elif t1 or t2:
        st.warning("Please enter *both* ticker symbols.")
        st.session_state.ticker1_calculated = None # Clear state
        st.session_state.ticker2_calculated = None
        st.session_state.rolling_corr_data = None
    else:
        st.warning("Ticker symbols cannot be empty.")
        st.session_state.ticker1_calculated = None # Clear state
        st.session_state.ticker2_calculated = None
        st.session_state.rolling_corr_data = None

# --- Display Plot Area ---
# Plot based on session state (which is updated only on button press)
if st.session_state.get('ticker1_calculated') and st.session_state.get('ticker2_calculated'):
    # Use the placeholder to display the plot area
    with plot_placeholder.container():
        fig = create_corr_plot(
            st.session_state.rolling_corr_data,
            st.session_state.ticker1_calculated,
            st.session_state.ticker2_calculated,
            window=ROLLING_WINDOW,
            years=YEARS_OF_DATA
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    # Initial message before any calculation is attempted
    with plot_placeholder.container():
         st.info("Enter two ticker symbols above and click 'Calculate Correlation'.")

# --- Ticker Reference Table ---
st.divider() # Add a visual separator before the expander and footer

# Prepare DataFrame for the ticker examples
ticker_data_list = []
for category, tickers in ticker_examples.items():
    for name, symbol in tickers.items():
        ticker_data_list.append({"Asset Class": category, "Description": name, "Yahoo Ticker": symbol})
ticker_df = pd.DataFrame(ticker_data_list)
ticker_df = ticker_df.sort_values(by=["Asset Class", "Description"]).reset_index(drop=True) # Sort for better readability

with st.expander("Show Example Ticker Symbols"):
    st.dataframe(ticker_df, use_container_width=True, hide_index=True)

# --- Footer Section ---
st.divider()

# Replace with your actual LinkedIn profile URL and Name
linkedin_url = "YOUR_LINKEDIN_PROFILE_URL" # e.g. "https://www.linkedin.com/in/yourname/"
your_name = "Your Name"

# Simple Icons SVG path for LinkedIn logo (adjust fill color if needed)
linkedin_svg = """
<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#0077B5">
    <title>LinkedIn</title>
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/>
</svg>
"""

# Construct the footer HTML - Centered text, icon link
# Using flexbox for alignment
footer_html = f"""
<div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 10px 0; color: grey; font-size: small;">
    <span>Created by {"Kenneth Quah"}</span>
    <a href="{"https://www.linkedin.com/in/kennethquah/"}" target="_blank" style="text-decoration: none; color: grey; margin-left: 10px; display: inline-flex; align-items: center;">
        {linkedin_svg} </a>
</div>
"""

# Use st.markdown to display the footer - Place this last
st.markdown(footer_html, unsafe_allow_html=True)

# Final caption about data source (optional placement)
st.caption(f"Data sourced from Yahoo Finance via yfinance library. Calculations based on daily 'Close' prices over the last {YEARS_OF_DATA} years with a {ROLLING_WINDOW}-day window.")