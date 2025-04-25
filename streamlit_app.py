import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback # For detailed error printing
import numpy as np # Needed for checking numeric types

# --- Configuration ---
DEFAULT_TICKER_1_CORR = 'EURUSD=X'
DEFAULT_TICKER_2_CORR = 'GLD'
DEFAULT_TICKER_SKEW = 'AAPL' # Default for skew plot
ROLLING_WINDOW = 30
YEARS_OF_DATA = 10

# --- Data Fetching Functions ---

@st.cache_data # Cache results based on input arguments
def calculate_rolling_correlation(ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """
    Fetches data for two tickers and calculates their rolling correlation using 'Close' price.
    Returns the correlation Series or None if data is insufficient or an error occurs.
    """
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + window + 50)

    try:
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)
        if data is None or data.empty or 'Close' not in data.columns:
             print(f"Error: Download failed or 'Close' column missing for {ticker1}/{ticker2}.")
             return None
        close_data = data.get('Close', pd.DataFrame())
        if close_data.empty:
             print(f"Error: 'Close' price data is empty for {ticker1}/{ticker2}.")
             st.warning(f"Could not retrieve 'Close' price data for {ticker1} or {ticker2}.")
             return None
        close_data.ffill(inplace=True)

        missing_tickers = []
        if ticker1 not in close_data.columns or close_data[ticker1].isnull().all(): missing_tickers.append(ticker1)
        if ticker2 not in close_data.columns or close_data[ticker2].isnull().all(): missing_tickers.append(ticker2)
        if missing_tickers:
             print(f"Error: Missing or all-null data for ticker(s): {', '.join(missing_tickers)}.")
             st.error(f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}.")
             return None

        returns = close_data.pct_change().dropna()
        if ticker1 not in returns.columns or ticker2 not in returns.columns or returns.empty: return None
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all(): return None
        if len(returns) < window: return None
        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()
        rolling_corr.name = f'{window}d Rolling Corr'
        print(f"Successfully calculated rolling correlation. Shape: {rolling_corr.shape}")
        return rolling_corr
    except Exception as e:
        print(f"An error occurred during correlation calculation: {e}")
        print(traceback.format_exc())
        st.error(f"An unexpected error occurred processing correlation for {ticker1}/{ticker2}.")
        return None

@st.cache_data # Cache expiration dates for a ticker for short time
def get_expiration_dates(ticker):
    """Fetches available option expiration dates for a ticker."""
    if not ticker: return None, "Ticker is empty."
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, f"No options expiration dates found for ticker {ticker}."
        return list(expirations), None
    except Exception as e:
        print(f"Error fetching expirations for {ticker}: {e}")
        return None, f"Error fetching expirations for {ticker}. Is it a valid equity/ETF ticker with options?"

@st.cache_data # Cache chain data for ticker+expiry
def get_option_chain_data(ticker, expiry_date):
    """Fetches option chain data and current price for a ticker and expiry date."""
    if not ticker or not expiry_date: return None, None, None, "Ticker or expiry date missing."
    try:
        print(f"Fetching option chain for {ticker} expiring {expiry_date}...")
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry_date)
        calls_df = chain.calls
        puts_df = chain.puts
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))
        if current_price is None or not isinstance(current_price, (int, float)):
             print(f"Warning: Could not determine current price for {ticker}. Plotting without price line.")
             current_price = None
        if calls_df.empty and puts_df.empty:
             return None, None, current_price, f"No options data returned for {ticker} on {expiry_date}."
        print(f"Fetched chain for {ticker} {expiry_date}. Calls: {len(calls_df)}, Puts: {len(puts_df)}, Price: {current_price}")
        return calls_df, puts_df, current_price, None
    except Exception as e:
        print(f"Error fetching option chain for {ticker} {expiry_date}: {e}")
        print(traceback.format_exc())
        return None, None, None, f"Error fetching option chain for {ticker} on {expiry_date}. Check ticker and date."

# --- Plotting Functions ---

def create_corr_plot(rolling_corr, ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Creates the Plotly figure for rolling correlation."""
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"
    if rolling_corr is not None and not rolling_corr.empty:
        pos_corr = rolling_corr.copy(); pos_corr[pos_corr < 0] = None
        neg_corr = rolling_corr.copy(); neg_corr[neg_corr >= 0] = None
        fig.add_trace(go.Scatter(x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation', line=dict(color='red')))
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    elif rolling_corr is not None and rolling_corr.empty:
         error_message = f"No correlation data available for {ticker1} / {ticker2}."
         fig.update_layout(title=plot_title, annotations=[dict(text=error_message, showarrow=False, align='center')])
    else: # rolling_corr is None
        error_message = f"Could not load or process data for {ticker1} / {ticker2}."
        fig.update_layout(title=plot_title, annotations=[dict(text=error_message, showarrow=False, align='center')])
    if rolling_corr is None or rolling_corr.empty: # Common layout for error/empty cases
         fig.update_layout(xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500)
    return fig

def create_iv_skew_plot(calls_df, puts_df, ticker, expiry_date, current_price):
    """Creates the Plotly figure for IV Skew."""
    fig = go.Figure()
    plot_title = f"{ticker} Implied Volatility Skew (Expiry: {expiry_date})"
    data_plotted = False

    if calls_df is not None and not calls_df.empty and 'strike' in calls_df.columns and 'impliedVolatility' in calls_df.columns:
        calls_df = calls_df[pd.to_numeric(calls_df['strike'], errors='coerce').notna()]
        calls_df = calls_df[pd.to_numeric(calls_df['impliedVolatility'], errors='coerce').notna()]
        if not calls_df.empty:
            fig.add_trace(go.Scatter(x=calls_df['strike'], y=calls_df['impliedVolatility'] * 100, mode='markers+lines', name='Calls IV (%)', marker=dict(color='blue'), line=dict(color='blue')))
            data_plotted = True

    if puts_df is not None and not puts_df.empty and 'strike' in puts_df.columns and 'impliedVolatility' in puts_df.columns:
        puts_df = puts_df[pd.to_numeric(puts_df['strike'], errors='coerce').notna()]
        puts_df = puts_df[pd.to_numeric(puts_df['impliedVolatility'], errors='coerce').notna()]
        if not puts_df.empty:
            fig.add_trace(go.Scatter(x=puts_df['strike'], y=puts_df['impliedVolatility'] * 100, mode='markers+lines', name='Puts IV (%)', marker=dict(color='orange'), line=dict(color='orange')))
            data_plotted = True

    fig.update_layout(title=plot_title, xaxis_title='Strike Price', yaxis_title='Implied Volatility (%)', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    if current_price is not None:
        fig.add_vline(x=current_price, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Current Price: {current_price:.2f}", annotation_position="top right")

    if not data_plotted:
         error_message = f"No valid options IV data found for {ticker} expiring {expiry_date}."
         fig.add_annotation(text=error_message, showarrow=False, align='center')
         fig.update_layout(yaxis_range=[0, 100]) # Set default range for empty plot
    return fig

# --- Ticker Examples Data ---
ticker_examples = {
    "Equity Indices": { "S&P 500 (US)": "^GSPC", "Nasdaq Composite (US)": "^IXIC", "Dow Jones Industrial Avg (US)": "^DJI", "FTSE 100 (UK)": "^FTSE", "DAX (Germany)": "^GDAXI", "Nikkei 225 (Japan)": "^N225", "Straits Times Index (SG)": "^STI", },
    "Forex (vs USD)": { "Euro": "EURUSD=X", "British Pound": "GBPUSD=X", "Japanese Yen (USD/JPY)": "JPY=X", "Australian Dollar": "AUDUSD=X", "Singapore Dollar (USD/SGD)": "SGD=X", "Canadian Dollar (USD/CAD)": "CAD=X", },
    "Commodities": { "Gold Futures": "GC=F", "Crude Oil Futures (WTI)": "CL=F", "Silver Futures": "SI=F", "Copper Futures": "HG=F", "Natural Gas Futures": "NG=F", "SPDR Gold Shares ETF": "GLD", "US Oil Fund ETF": "USO", },
    "Cryptocurrencies (vs USD)": { "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD", },
    "US Treasury Yields": { "10 Year (^TNX)": "^TNX", "5 Year (^FVX)": "^FVX", "30 Year (^TYX)": "^TYX", },
    "Example Stocks":{ "Apple (US)": "AAPL", "Microsoft (US)": "MSFT", "Google (US)": "GOOGL", "DBS Group (SG)": "D05.SI", "Tesla (US)": "TSLA", }
}
ticker_data_list = []
for category, tickers in ticker_examples.items():
    for name, symbol in tickers.items():
        ticker_data_list.append({"Asset Class": category, "Description": name, "Yahoo Ticker": symbol})
ticker_df = pd.DataFrame(ticker_data_list)
ticker_df = ticker_df.sort_values(by=["Asset Class", "Description"]).reset_index(drop=True)


# --- Streamlit App ---
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("Financial Dashboard")

# --- Initialize Session State ---
# Correlation State
if 'rolling_corr_data' not in st.session_state: st.session_state.rolling_corr_data = None
if 'ticker1_calculated_corr' not in st.session_state: st.session_state.ticker1_calculated_corr = None
if 'ticker2_calculated_corr' not in st.session_state: st.session_state.ticker2_calculated_corr = None
# Skew State
if 'calls_data_skew' not in st.session_state: st.session_state.calls_data_skew = None
if 'puts_data_skew' not in st.session_state: st.session_state.puts_data_skew = None
if 'price_skew' not in st.session_state: st.session_state.price_skew = None
if 'ticker_calculated_skew' not in st.session_state: st.session_state.ticker_calculated_skew = None
if 'expiry_calculated_skew' not in st.session_state: st.session_state.expiry_calculated_skew = None


# --- Section 1: Rolling Correlation ---
st.header("Rolling Correlation Calculator")
col1_corr, col2_corr = st.columns(2)
with col1_corr:
    ticker1_corr_input = st.text_input("Ticker 1:", value=st.session_state.get('ticker1_calculated_corr', DEFAULT_TICKER_1_CORR), key="ticker1_corr_widget")
with col2_corr:
    ticker2_corr_input = st.text_input("Ticker 2:", value=st.session_state.get('ticker2_calculated_corr', DEFAULT_TICKER_2_CORR), key="ticker2_corr_widget")

calculate_corr_button = st.button("Calculate Correlation", key="corr_button", type="primary")
plot_placeholder_corr = st.empty()

if calculate_corr_button:
    t1_corr = ticker1_corr_input.strip().upper()
    t2_corr = ticker2_corr_input.strip().upper()
    if t1_corr and t2_corr:
        with st.spinner(f"Calculating correlation for {t1_corr} and {t2_corr}..."):
            try:
                result_corr = calculate_rolling_correlation(t1_corr, t2_corr, window=ROLLING_WINDOW, years=YEARS_OF_DATA)
                st.session_state.rolling_corr_data = result_corr
                st.session_state.ticker1_calculated_corr = t1_corr
                st.session_state.ticker2_calculated_corr = t2_corr
            except Exception as e:
                st.error(f"Failed processing correlation for {t1_corr}/{t2_corr}.")
                st.session_state.rolling_corr_data = None
                st.session_state.ticker1_calculated_corr = t1_corr
                st.session_state.ticker2_calculated_corr = t2_corr
    elif t1_corr or t2_corr:
        st.warning("Please enter *both* ticker symbols for correlation.")
        st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None; st.session_state.rolling_corr_data = None
    else:
        st.warning("Ticker symbols cannot be empty.")
        st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None; st.session_state.rolling_corr_data = None

# Display Correlation Plot based on state
if st.session_state.get('ticker1_calculated_corr') and st.session_state.get('ticker2_calculated_corr'):
    with plot_placeholder_corr.container():
        fig_corr = create_corr_plot(st.session_state.rolling_corr_data, st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
else:
     with plot_placeholder_corr.container():
         st.info("Enter two tickers and click 'Calculate Correlation'.")

# --- Section 2: Implied Volatility Skew ---
st.divider()
st.header("Implied Volatility Skew Viewer")

# Inputs for Skew
# Use the DEBUG block for the ticker input to handle potential errors safely
# --- Start DEBUG Block for Skew Ticker Input ---
default_val_skew = st.session_state.get('ticker_calculated_skew', DEFAULT_TICKER_SKEW)
ticker_skew_input_widget_val = st.text_input(
    "Equity/ETF Ticker:",
    value=default_val_skew,
    key="ticker_skew_widget" # Use a specific key
)
if isinstance(ticker_skew_input_widget_val, str):
    ticker_skew_input = ticker_skew_input_widget_val.strip().upper()
else:
    print(f"WARNING: Unexpected type from text_input: {type(ticker_skew_input_widget_val)}. Using default.")
    ticker_skew_input = DEFAULT_TICKER_SKEW
# --- End DEBUG Block ---


expirations, error_msg = get_expiration_dates(ticker_skew_input)
expiry_input = None

if expirations:
    try:
        today = pd.Timestamp.now().normalize()
        exp_dates = pd.to_datetime(expirations)
        target_date = today + pd.Timedelta(days=90)
        future_dates = exp_dates[exp_dates >= today] # Consider today as well

        if not future_dates.empty:
            closest_date = future_dates.iloc[(future_dates - target_date).abs().argsort()[0]]
            default_expiry_str = closest_date.strftime('%Y-%m-%d')
            default_sel_index = expirations.index(default_expiry_str) if default_expiry_str in expirations else 0
        else: # Handle case where all expirations are in the past (unlikely but possible)
            default_sel_index = len(expirations) - 1 if expirations else 0
    except Exception as e:
        print(f"Error finding default expiration index: {e}")
        default_sel_index = 0

    previous_expiry = st.session_state.get('expiry_calculated_skew')
    current_index = default_sel_index # Start with default
    if previous_expiry and previous_expiry in expirations:
         # If previous selection is still valid for the *current* ticker, use it
         if ticker_skew_input == st.session_state.get('ticker_calculated_skew'):
              current_index = expirations.index(previous_expiry)

    expiry_input = st.selectbox("Select Expiration Date:", expirations, index=current_index, key="expiry_select_widget")
else:
    if ticker_skew_input: # Only show warning if user has typed something
        st.warning(error_msg or f"Enter a valid equity ticker with options.")

graph_skew_button = st.button("Graph IV Skew", key="skew_button", type="primary")
plot_placeholder_skew = st.empty()

# Skew Calculation/Plotting Logic
if graph_skew_button:
    if ticker_skew_input and expiry_input:
        with st.spinner(f"Fetching option chain for {ticker_skew_input} expiring {expiry_input}..."):
            # Call the cached data fetch function
            calls, puts, price, error_msg_fetch = get_option_chain_data(ticker_skew_input, expiry_input)
            if error_msg_fetch:
                st.error(error_msg_fetch)
                st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None
                st.session_state.ticker_calculated_skew = ticker_skew_input # Store attempted params
                st.session_state.expiry_calculated_skew = expiry_input
            else:
                st.session_state.calls_data_skew = calls; st.session_state.puts_data_skew = puts; st.session_state.price_skew = price
                st.session_state.ticker_calculated_skew = ticker_skew_input; st.session_state.expiry_calculated_skew = expiry_input
    else:
        st.warning("Please enter a valid ticker and select an expiration date.")
        st.session_state.ticker_calculated_skew = None; st.session_state.expiry_calculated_skew = None # Clear state
        st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None


# Display Skew Plot based on state
if st.session_state.get('ticker_calculated_skew') and st.session_state.get('expiry_calculated_skew'):
    with plot_placeholder_skew.container():
        # Data is already fetched and stored in state by button click
        fig_skew = create_iv_skew_plot(
            st.session_state.calls_data_skew,
            st.session_state.puts_data_skew,
            st.session_state.ticker_calculated_skew,
            st.session_state.expiry_calculated_skew,
            st.session_state.price_skew
        )
        st.plotly_chart(fig_skew, use_container_width=True)
else:
     with plot_placeholder_skew.container():
         st.info("Enter an equity ticker, select an expiration, and click 'Graph IV Skew'.")


# --- Ticker Reference Table & Footer ---
st.divider()
with st.expander("Show Example Ticker Symbols"):
    st.dataframe(ticker_df, use_container_width=True, hide_index=True)

st.divider()
# Replace with your actual LinkedIn profile URL and Name
linkedin_url = "YOUR_LINKEDIN_PROFILE_URL" # e.g. "https://www.linkedin.com/in/yourname/"
your_name = "Your Name"
linkedin_svg = """<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#0077B5"><title>LinkedIn</title><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>"""
footer_html = f"""<div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 10px 0; color: grey; font-size: small;"><span>Created by {your_name}</span><a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: grey; margin-left: 10px; display: inline-flex; align-items: center;">{linkedin_svg}</a></div>"""
st.markdown(footer_html, unsafe_allow_html=True)

st.caption(f"Data sourced from Yahoo Finance via yfinance library.")