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
YEARS_OF_DATA = 10 # For correlation
DEFAULT_FRED_SERIES_NAME = "Effective Federal Funds Rate (Daily)" # Define default FRED series name
# --- Define duration for Fed Jaws chart ---
FED_JAWS_DURATION_DAYS = 90 # Approx 3 months


# --- FRED API Key Configuration ---
# Try to get the FRED API key from Streamlit secrets
# Create a file .streamlit/secrets.toml and add: FRED_API_KEY = "YOUR_API_KEY_HERE"
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except (KeyError, FileNotFoundError): # Handle missing file/key
    FRED_API_KEY = None # Set to None if not found

# Instantiate Fred object (only if key is available)
fred = None
fred_error_message = None # To store any initialization error message
if FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        print("FRED client initialized successfully.") # Debug print
    except Exception as e:
        fred_error_message = f"Failed to initialize FRED API: {e}. Check your API key configuration."
        print(fred_error_message)
        FRED_API_KEY = None # Disable further attempts
else:
     # No key provided at all
     fred_error_message = "FRED API Key not found. Please configure it in Streamlit Secrets (`.streamlit/secrets.toml`) to use the FRED data section."
     print(fred_error_message)


# --- Data Fetching Functions ---

@st.cache_data
def calculate_rolling_correlation(ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Calculates rolling correlation (uses yfinance)."""
    print(f"Executing: calculate_rolling_correlation for {ticker1} vs {ticker2}...")
    # Set date range for data download
    end_date = datetime.now()
    # Add buffer days for rolling window calculation and potential missing data
    start_date = end_date - timedelta(days=years*365 + window + 50)
    try:
        # Download historical data for both tickers
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date, progress=False)

        # Basic validation of downloaded data
        if data is None or data.empty:
            st.error(f"Failed to download data for {ticker1} or {ticker2}.")
            return None
        if 'Close' not in data.columns:
             st.error(f"Could not find 'Close' price column in downloaded data.")
             return None

        # Extract 'Close' prices, handling potential multi-level columns
        close_data = data.get('Close', pd.DataFrame())
        if close_data.empty:
            st.warning(f"Could not retrieve 'Close' price data for {ticker1} or {ticker2}.")
            return None

        # Forward fill missing values to handle non-trading days
        close_data.ffill(inplace=True)

        # Check if data exists for both tickers after potential download issues
        missing_tickers = []
        if ticker1 not in close_data.columns or close_data[ticker1].isnull().all():
            missing_tickers.append(ticker1)
        if ticker2 not in close_data.columns or close_data[ticker2].isnull().all():
            missing_tickers.append(ticker2)
        if missing_tickers:
            st.error(f"Could not find sufficient data for ticker(s): {', '.join(missing_tickers)}.")
            return None

        # Calculate percentage returns
        returns = close_data.pct_change().dropna()

        # Check if returns data is valid for both tickers
        if ticker1 not in returns.columns or ticker2 not in returns.columns or returns.empty:
            st.warning(f"Could not calculate returns for {ticker1} or {ticker2}.")
            return None
        if returns[ticker1].isnull().all() or returns[ticker2].isnull().all():
             st.warning(f"Returns data contains only NaNs for {ticker1} or {ticker2}.")
             return None

        # Ensure enough data points for the rolling window
        if len(returns) < window:
            st.warning(f"Not enough data points ({len(returns)}) for the specified {window}-day rolling window.")
            return None

        # Calculate rolling correlation
        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()
        rolling_corr.name = f'{window}d Rolling Corr' # Assign a name for the series

        print(f"Successfully calculated rolling correlation. Shape: {rolling_corr.shape}")
        return rolling_corr

    except Exception as e:
        print(f"Error calculating rolling correlation for {ticker1}/{ticker2}: {e}")
        print(traceback.format_exc()) # Print detailed traceback for debugging
        st.error(f"An error occurred while calculating correlation for {ticker1}/{ticker2}. Check ticker symbols and data availability.")
        return None

@st.cache_data
def get_expiration_dates(ticker):
    """Fetches option expiration dates for a given ticker using yfinance."""
    if not ticker:
        return None, "Ticker symbol cannot be empty."
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations: # Check if the tuple is empty
            return None, f"No options expiration dates found for ticker: {ticker}."
        return list(expirations), None # Return list of dates and no error
    except Exception as e:
        print(f"Error fetching expirations for {ticker}: {e}")
        # Consider more specific error handling if needed (e.g., network errors)
        return None, f"An error occurred while fetching expiration dates for {ticker}. It might be an invalid ticker or a temporary issue."

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

        # Attempt to get the current stock price robustly
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))

        # Validate the fetched price
        if current_price is None or not isinstance(current_price, (int, float, np.number)):
             print(f"Warning: Could not determine a valid current price for {ticker}. Price data might be delayed or unavailable.")
             current_price = None # Set to None if invalid

        # Check if any options data was returned
        if calls_df.empty and puts_df.empty:
            return None, None, current_price, f"No options data found for {ticker} on {expiry_date}. The ticker might not have options or data is unavailable for this date."

        print(f"Fetched chain for {ticker} {expiry_date}. Calls: {len(calls_df)}, Puts: {len(puts_df)}, Price: {current_price}")
        return calls_df, puts_df, current_price, None # Return data and no error

    except Exception as e:
        print(f"Error fetching option chain for {ticker} on {expiry_date}: {e}")
        print(traceback.format_exc())
        return None, None, None, f"An error occurred fetching the option chain for {ticker} on {expiry_date}. Check the ticker and expiration date."


@st.cache_data(show_spinner=False) # Disable default spinner for FRED calls
def get_fred_data(_fred_instance, series_id):
    """Fetches time series data for a given FRED series ID."""
    if not _fred_instance:
        # This check prevents errors if FRED failed to initialize (e.g., bad API key)
        return None, "FRED API client is not initialized. Check API key configuration."
    if not series_id:
        return None, "FRED Series ID cannot be empty."
    try:
        print(f"Fetching FRED data for series ID: {series_id}...")
        data = _fred_instance.get_series(series_id)
        # Remove trailing NaNs which are common in FRED data
        data = data.dropna()
        if data.empty:
             print(f"Warning: FRED data for {series_id} is empty after removing NaNs.")
             # Optionally return an error message here if empty data is critical
             # return None, f"No valid data found for FRED series '{series_id}'."

        print(f"Successfully fetched FRED data for {series_id}. Shape: {data.shape}")
        return data, None # Return data and no error
    except Exception as e:
        error_msg = f"Failed to fetch data for FRED series '{series_id}': {e}"
        print(error_msg)
        print(traceback.format_exc())
        # Provide a user-friendly error message
        return None, f"Could not fetch data for FRED series '{series_id}'. Please verify the Series ID and your FRED API connection."

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
        # The info object is typically a pandas Series
        print(f"Successfully fetched info for FRED series {series_id}.")
        return info, None
    except Exception as e:
        error_msg = f"Failed to fetch metadata for FRED series '{series_id}': {e}"
        print(error_msg)
        # Don't necessarily need traceback here unless debugging specific info issues
        return None, f"Could not fetch metadata for FRED series '{series_id}'. The series might exist, but metadata retrieval failed."

# --- Function to fetch multiple FRED series ---
@st.cache_data(show_spinner=False)
def get_multiple_fred_data(_fred_instance, series_ids, start_date=None, end_date=None):
    """
    Fetches data for multiple FRED series IDs and merges them.
    If start_date and end_date are provided, fetches data within that range.
    Otherwise, fetches all available historical data for each series.
    """
    if not _fred_instance:
        return None, "FRED API client is not initialized."
    if not series_ids:
        return None, "No FRED Series IDs provided."

    all_series_data = {}
    errors = {}
    successful_ids = []

    # Log the date range being used (if provided)
    if start_date and end_date:
        print(f"Fetching FRED multi-series data from {start_date.date()} to {end_date.date()}")
    else:
        print("Fetching all available FRED multi-series data (no date range specified).")


    for series_id in series_ids:
        try:
            print(f"Fetching FRED data for series: {series_id}...")
            # Pass start/end dates if they exist
            if start_date and end_date:
                 s_data = _fred_instance.get_series(series_id, observation_start=start_date, observation_end=end_date)
            else:
                 s_data = _fred_instance.get_series(series_id) # Fetch all data

            s_data = s_data.dropna() # Drop NaNs
            if not s_data.empty:
                all_series_data[series_id] = s_data
                successful_ids.append(series_id)
                print(f"Successfully fetched {series_id}, shape: {s_data.shape}")
            else:
                print(f"Warning: No data returned for {series_id} in the specified range/period after dropping NaNs.")
                # Don't necessarily mark as error, could be no data in range
                # errors[series_id] = "No data found or returned empty."
        except Exception as e:
            error_msg = f"Failed to fetch {series_id}: {e}"
            print(error_msg)
            errors[series_id] = error_msg

    if not all_series_data:
         # If *no* series could be fetched at all
         error_details = f"Errors: {errors}" if errors else "Unknown reason."
         return None, f"Failed to fetch data for *any* of the requested series. {error_details}"

    # Combine the successfully fetched series into a single DataFrame
    # Use outer join to keep all dates
    try:
        combined_df = pd.concat(all_series_data, axis=1, join='outer')
        # Consider if filling is appropriate for these specific rates
        # combined_df.ffill(inplace=True) # Forward fill
        print(f"Combined FRED data shape: {combined_df.shape}")

        # Report any errors for series that failed but where others succeeded
        if errors:
            st.warning(f"Could not fetch data for some series: {errors}")

        return combined_df, None # Return combined data and no error (even if some series failed)

    except Exception as e:
        print(f"Error combining FRED series: {e}")
        print(traceback.format_exc())
        return None, f"Error occurred while combining FRED data: {e}"


# --- Plotting Functions ---
def create_corr_plot(rolling_corr, ticker1, ticker2, window=ROLLING_WINDOW, years=YEARS_OF_DATA):
    """Creates the Plotly figure for visualizing rolling correlation."""
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"

    # Check if correlation data is valid and not empty
    if rolling_corr is not None and not rolling_corr.empty:
        # Separate positive and negative correlations for different colors
        pos_corr = rolling_corr.copy()
        pos_corr[pos_corr < 0] = None # Set negative values to NaN for the positive trace
        neg_corr = rolling_corr.copy()
        neg_corr[neg_corr >= 0] = None # Set positive values to NaN for the negative trace

        # Add traces for positive and negative correlation segments
        fig.add_trace(go.Scatter(
            x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation',
            line=dict(color='red')
        ))

        # Configure layout for the plot with data
        fig.update_layout(
            title=plot_title,
            xaxis_title='Date',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1], # Ensure y-axis spans the full correlation range
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Place legend above plot
        )
    # Handle cases where correlation calculation returned an empty series (e.g., not enough data)
    elif rolling_corr is not None and rolling_corr.empty:
        error_message = f"No correlation data available for {ticker1} / {ticker2} with the selected parameters."
        fig.update_layout(
            title=plot_title,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
        )
    # Handle cases where correlation calculation failed (returned None)
    else:
        error_message = f"Could not load or process correlation data for {ticker1} / {ticker2}."
        fig.update_layout(
            title=plot_title,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
        )

    # Ensure basic layout is set even if there's no data or an error
    if rolling_corr is None or rolling_corr.empty:
        fig.update_layout(
             xaxis_title='Date',
             yaxis_title='Correlation Coefficient',
             yaxis_range=[-1, 1],
             height=500
         )

    return fig

def create_iv_skew_plot(calls_df, puts_df, ticker, expiry_date, current_price):
    """Creates the Plotly figure for Implied Volatility Skew."""
    fig = go.Figure()
    plot_title = f"{ticker} Implied Volatility Skew (Expiry: {expiry_date})"
    data_plotted = False # Flag to track if any data was actually plotted

    # Plot Calls IV if data is available and valid
    if calls_df is not None and not calls_df.empty and 'strike' in calls_df.columns and 'impliedVolatility' in calls_df.columns:
        # Ensure strike and IV are numeric, coercing errors to NaN and dropping them
        calls_df['strike'] = pd.to_numeric(calls_df['strike'], errors='coerce')
        calls_df['impliedVolatility'] = pd.to_numeric(calls_df['impliedVolatility'], errors='coerce')
        calls_df.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not calls_df.empty:
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['impliedVolatility'] * 100, # Convert IV to percentage
                mode='markers+lines',
                name='Calls IV (%)',
                marker=dict(color='blue'),
                line=dict(color='blue')
            ))
            data_plotted = True

    # Plot Puts IV if data is available and valid
    if puts_df is not None and not puts_df.empty and 'strike' in puts_df.columns and 'impliedVolatility' in puts_df.columns:
        # Ensure strike and IV are numeric, coercing errors to NaN and dropping them
        puts_df['strike'] = pd.to_numeric(puts_df['strike'], errors='coerce')
        puts_df['impliedVolatility'] = pd.to_numeric(puts_df['impliedVolatility'], errors='coerce')
        puts_df.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not puts_df.empty:
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['impliedVolatility'] * 100, # Convert IV to percentage
                mode='markers+lines',
                name='Puts IV (%)',
                marker=dict(color='orange'),
                line=dict(color='orange')
            ))
            data_plotted = True

    # Configure plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legend above plot
    )

    # Add a vertical line for the current stock price if available
    if current_price is not None and isinstance(current_price, (int, float, np.number)):
        fig.add_vline(
            x=current_price,
            line_width=1,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"Current Price: {current_price:.2f}",
            annotation_position="top right"
        )

    # If no valid data was plotted, display a message on the chart
    if not data_plotted:
        error_message = f"No valid options IV data found for {ticker} expiring {expiry_date}."
        fig.add_annotation(text=error_message, showarrow=False, align='center')
        # Set a default y-axis range if no data, otherwise it might be empty
        fig.update_layout(yaxis_range=[0, 100]) # Example range, adjust as needed

    return fig

def create_fred_plot(series_data, series_id, series_info):
    """Creates the Plotly figure for a SINGLE FRED time series and displays metadata."""
    fig = go.Figure()
    plot_title = f"FRED Series: {series_id}" # Default title
    y_axis_label = "Value" # Default y-axis label

    # Safely extract metadata from series_info (which is a pandas Series) if available
    if series_info is not None and not series_info.empty:
        # Use .get() for safe access, providing defaults
        plot_title = series_info.get('title', plot_title)
        units = series_info.get('units_short', 'Value')
        freq = series_info.get('frequency_short', '')
        adj = series_info.get('seasonal_adjustment_short', 'NSA') # NSA = Not Seasonally Adjusted
        y_axis_label = f"{units} ({freq}, {adj})" if freq else f"{units} ({adj})"

    # Plot the data if it exists and is not empty
    if series_data is not None and not series_data.empty:
        fig.add_trace(go.Scatter(
            x=series_data.index,
            y=series_data.values,
            mode='lines',
            name=series_id # Use series_id in the legend trace name
        ))
        # Configure layout for the plot with data
        fig.update_layout(
            title=plot_title,
            xaxis_title='Date',
            yaxis_title=y_axis_label,
            height=500
        )
        # Metadata caption will be handled outside this function in the main script
    else:
        # Handle cases where data is None or empty
        error_message = f"Could not load data for FRED series '{series_id}'."
        fig.update_layout(
            title=plot_title, # Still show title even with error
            xaxis_title='Date',
            yaxis_title=y_axis_label, # Still show axis labels
            height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
        )

    return fig

# --- Plotting function for Fed's Jaws ---
def create_fed_jaws_plot(jaws_data):
    """Creates the Plotly figure for the Fed's Jaws chart."""
    fig = go.Figure()
    plot_title = f"Fed's Jaws: Key Policy Rates (Last {FED_JAWS_DURATION_DAYS} Days)" # Updated title

    if jaws_data is None or jaws_data.empty:
        error_message = f"No data available to plot for the Fed's Jaws chart (last {FED_JAWS_DURATION_DAYS} days)."
        fig.update_layout(
            title=plot_title,
            height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')],
            xaxis_title='Date',
            yaxis_title='Percent (%)' # Default axis label
        )
        return fig

    # Define specific styling for target range limits
    target_range_line_style = dict(color='red', width=2, dash='dot')

    # Define user-friendly names for legend
    series_names = {
        'DFEDTARU': "Target Range - Upper Limit",
        'IORB':     "Interest Rate on Reserve Balances",
        'DPCREDIT': "Discount Window Primary Credit Rate",
        'SOFR':     "Secured Overnight Financing Rate",
        'DFF':      "Effective Federal Funds Rate (EFFR)",
        'OBFR':     "Overnight Bank Funding Rate",
        'DFEDTARL': "Target Range - Lower Limit",
    }

    # Plot each series present in the DataFrame
    for series_id in jaws_data.columns:
        # Check if the column actually exists before plotting (robustness)
        if series_id in jaws_data:
            line_style = None # Default line style
            # Get user-friendly name, fallback to series_id if not defined
            series_name = series_names.get(series_id, series_id)

            # Apply special styling for target range limits
            if series_id == 'DFEDTARU' or series_id == 'DFEDTARL':
                line_style = target_range_line_style

            # Add trace for the series
            fig.add_trace(go.Scatter(
                x=jaws_data.index,
                y=jaws_data[series_id],
                mode='lines',
                name=series_name, # Use user-friendly name
                line=line_style # Apply specific style if defined, otherwise default
            ))
        else:
             print(f"Warning: Series ID '{series_id}' defined but not found in fetched jaws_data.")


    # Configure plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Date',
        yaxis_title='Percent (%)', # Assuming most rates are percentages
        height=600, # Slightly taller for better visibility
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom", y=-0.3, # Adjusted position below the plot slightly more space
            xanchor="center", x=0.5
            )
    )

    return fig


# --- Ticker Examples Data ---
# (Keep the existing ticker_examples dictionary and DataFrame creation)
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
        "Copper Futures": "HG=F", "Natural Gas Futures": "NG=F",
        "SPDR Gold Shares ETF": "GLD", "US Oil Fund ETF": "USO",
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
ticker_data_list = []
for category, tickers in ticker_examples.items():
    for name, symbol in tickers.items():
        ticker_data_list.append({"Asset Class": category, "Description": name, "Yahoo Ticker": symbol})
ticker_df = pd.DataFrame(ticker_data_list)
ticker_df = ticker_df.sort_values(by=["Asset Class", "Description"]).reset_index(drop=True)


# --- FRED Series Examples ---
# Dictionary mapping user-friendly names to FRED Series IDs for the single viewer
FRED_SERIES_EXAMPLES = {
    "Effective Federal Funds Rate (Daily)": "DFF", # Moved default to top for clarity
    "Nominal GDP (Quarterly)": "GDP",
    "Real GDP (Quarterly)": "GDPC1",
    "CPI - All Urban Consumers (Monthly)": "CPIAUCSL",
    "Core CPI (Less Food & Energy, Monthly)": "CPILFESL",
    "PCE Price Index (Monthly)": "PCEPI",
    "Core PCE Price Index (Monthly)": "PCEPILFE",
    "Unemployment Rate (Monthly)": "UNRATE",
    "Initial Claims (Weekly)": "ICSA",
    "10-Year Treasury Constant Maturity Rate (Daily)": "DGS10",
    "M1 Money Stock (Weekly)": "WM1NS",
    "M2 Money Stock (Weekly)": "WM2NS",
    "Industrial Production Index (Monthly)": "INDPRO",
    "Retail Sales - Total (Monthly)": "RSAFS",
    "Gold Price (London Bullion, Daily)": "GOLDAMGBD228NLBM",
    "VIX (Volatility Index, Daily)": "VIXCLS",
    # Add other single series examples here if desired
}
# Create a list of the user-friendly names for the selectbox options
fred_series_options = list(FRED_SERIES_EXAMPLES.keys())

# --- FRED Series IDs for Fed's Jaws Chart ---
FED_JAWS_SERIES_IDS = [
    'DFEDTARU', # Federal Funds Target Range - Upper Limit
    'IORB',     # Interest Rate on Reserve Balances
    'DPCREDIT', # Discount Window Primary Credit Rate
    'SOFR',     # Secured Overnight Financing Rate
    'DFF',      # Effective Federal Funds Rate (EFFR)
    'OBFR',     # Overnight Bank Funding Rate
    'DFEDTARL', # Federal Funds Target Range - Lower Limit
]


# --- Streamlit App ---
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📈 Financial Dashboard") # Added an emoji for flair

# --- Initialize Session State ---
# Use functions to avoid repeating keys and provide defaults cleanly
def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

# Correlation State
init_state('rolling_corr_data', None)
init_state('ticker1_calculated_corr', None)
init_state('ticker2_calculated_corr', None)
init_state('ticker1_input_corr', DEFAULT_TICKER_1_CORR)
init_state('ticker2_input_corr', DEFAULT_TICKER_2_CORR)

# Skew State
init_state('calls_data_skew', None)
init_state('puts_data_skew', None)
init_state('price_skew', None)
init_state('ticker_calculated_skew', None)
init_state('expiry_calculated_skew', None)
init_state('ticker_input_skew', DEFAULT_TICKER_SKEW)
init_state('expiry_input_skew', None)

# FRED (Single Series) State
init_state('fred_data', None)
init_state('fred_series_info', None)
init_state('fred_series_id_calculated', None)
init_state('fred_series_name_calculated', None)
init_state('fred_series_name_input', DEFAULT_FRED_SERIES_NAME)

# --- Fed's Jaws State ---
init_state('fed_jaws_data', None)
init_state('fed_jaws_calculated', False) # Flag to know if calculation was attempted


# --- Section 1: Rolling Correlation ---
# (Keep existing code for Section 1)
st.header("📊 Rolling Correlation Calculator")
st.write("Calculates the rolling correlation between the daily returns of two assets.")
col1_corr, col2_corr = st.columns(2)
with col1_corr:
    # Use session state to preserve input field value across reruns
    ticker1_input_val = st.text_input(
        "Ticker 1:",
        value=st.session_state.ticker1_input_corr, # Use the input state variable
        key="ticker1_corr_widget",
        help="Enter a Yahoo Finance ticker symbol (e.g., AAPL, ^GSPC, EURUSD=X)."
    )
    # Safely process input
    st.session_state.ticker1_input_corr = ticker1_input_val.strip().upper() if isinstance(ticker1_input_val, str) else DEFAULT_TICKER_1_CORR

with col2_corr:
    ticker2_input_val = st.text_input(
        "Ticker 2:",
        value=st.session_state.ticker2_input_corr, # Use the input state variable
        key="ticker2_corr_widget",
        help="Enter another Yahoo Finance ticker symbol (e.g., GLD, ^IXIC, BTC-USD)."
    )
    # Safely process input
    st.session_state.ticker2_input_corr = ticker2_input_val.strip().upper() if isinstance(ticker2_input_val, str) else DEFAULT_TICKER_2_CORR

calculate_corr_button = st.button("Calculate Correlation", key="corr_button", type="primary")
plot_placeholder_corr = st.empty()
# Correlation Logic (remains the same)
if calculate_corr_button:
    t1_corr = st.session_state.ticker1_input_corr; t2_corr = st.session_state.ticker2_input_corr
    if t1_corr and t2_corr:
        if t1_corr == t2_corr:
             st.warning("Please enter two different ticker symbols.")
             st.session_state.rolling_corr_data = None; st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None
        else:
            with st.spinner(f"Calculating {ROLLING_WINDOW}-day rolling correlation for {t1_corr} vs {t2_corr}..."):
                try:
                    result_corr = calculate_rolling_correlation(t1_corr, t2_corr, window=ROLLING_WINDOW, years=YEARS_OF_DATA)
                    st.session_state.rolling_corr_data = result_corr
                    st.session_state.ticker1_calculated_corr = t1_corr; st.session_state.ticker2_calculated_corr = t2_corr
                except Exception as e:
                    st.error(f"An unexpected error occurred during correlation calculation: {e}")
                    st.session_state.rolling_corr_data = None
                    st.session_state.ticker1_calculated_corr = t1_corr; st.session_state.ticker2_calculated_corr = t2_corr
    elif not t1_corr and not t2_corr: st.warning("Please enter ticker symbols in both fields."); st.session_state.rolling_corr_data = None; st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None
    else: st.warning("Please enter a ticker symbol in the missing field."); st.session_state.rolling_corr_data = None; st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None
# Display Correlation Plot (remains the same)
if st.session_state.get('ticker1_calculated_corr') and st.session_state.get('ticker2_calculated_corr'):
    with plot_placeholder_corr.container():
        fig_corr = create_corr_plot(st.session_state.rolling_corr_data, st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr, window=ROLLING_WINDOW, years=YEARS_OF_DATA)
        st.plotly_chart(fig_corr, use_container_width=True)
else:
     with plot_placeholder_corr.container(): st.info("Enter two ticker symbols and click 'Calculate Correlation' to view the rolling correlation plot.")


# --- Section 2: Implied Volatility Skew ---
st.divider()
st.header("📉 Implied Volatility Skew Viewer")
st.write("Visualizes the Implied Volatility (IV) smile/skew for options of a selected equity or ETF.")
ticker_input_value_skew = st.text_input(
    "Equity/ETF Ticker:", value=st.session_state.ticker_input_skew, key="ticker_skew_widget",
    help="Enter a Yahoo Finance ticker for a stock or ETF with options (e.g., AAPL, SPY, TSLA)."
)
if isinstance(ticker_input_value_skew, str): st.session_state.ticker_input_skew = ticker_input_value_skew.strip().upper()
else: print(f"Warning: st.text_input for skew ticker returned non-string: {type(ticker_input_value_skew)}. Using default."); st.session_state.ticker_input_skew = DEFAULT_TICKER_SKEW
expirations, error_msg_exp = get_expiration_dates(st.session_state.ticker_input_skew)
expiry_input = None
if expirations:
    try:
        today = pd.Timestamp.now().normalize()
        exp_dates = pd.to_datetime(expirations) # exp_dates is a DatetimeIndex
        future_dates = exp_dates[exp_dates >= today] # future_dates is also a DatetimeIndex

        default_sel_index = 0 # Default to first index if logic fails or no future dates
        if not future_dates.empty:
            # Find date closest to 90 days from now
            target_date = today + pd.Timedelta(days=90)
            # --- FIX 2: Use NumPy directly on the TimedeltaIndex values ---
            time_deltas = future_dates - target_date # This results in a TimedeltaIndex
            # Use np.argmin on the absolute values of the underlying NumPy array
            closest_date_pos_in_future = np.abs(time_deltas.values).argmin()
            # --- END FIX 2 ---
            closest_date_ts = future_dates[closest_date_pos_in_future] # Get the actual Timestamp
            default_expiry_str = closest_date_ts.strftime('%Y-%m-%d') # Format as string
            # Find the index of this string in the original list
            if default_expiry_str in expirations:
                default_sel_index = expirations.index(default_expiry_str)
            else:
                 # Fallback if formatted string not found (shouldn't happen often)
                 print(f"Warning: Closest date string '{default_expiry_str}' not found in original expirations list.")
                 default_sel_index = 0 # Fallback to first
        elif expirations: # If only past dates exist, default to the last one in the list
             default_sel_index = len(expirations) - 1

        # Preserve selection if ticker hasn't changed and expiry is valid
        current_index = default_sel_index
        if st.session_state.ticker_input_skew == st.session_state.get('ticker_calculated_skew') and st.session_state.get('expiry_calculated_skew') in expirations:
             current_index = expirations.index(st.session_state.expiry_calculated_skew)

        # Use session state for the selectbox value
        st.session_state.expiry_input_skew = st.selectbox(
            "Select Expiration Date:", expirations, index=current_index, key="expiry_select_widget", help="Choose the options contract expiration date."
            )
        expiry_input = st.session_state.expiry_input_skew # Assign to local variable for logic below

    except Exception as e:
        st.error(f"Error finding default expiration date: {e}") # Show error to user
        print(f"Error finding default expiration date: {e}") # Print for debugging
        print(traceback.format_exc())
        # Fallback: Simple selectbox without smart default index
        st.session_state.expiry_input_skew = st.selectbox(
            "Select Expiration Date:", expirations, key="expiry_select_widget_fallback"
            )
        expiry_input = st.session_state.expiry_input_skew
else:
    if st.session_state.ticker_input_skew: st.warning(error_msg_exp or f"Could not find options data for '{st.session_state.ticker_input_skew}'. Enter a valid ticker with options.")

graph_skew_button = st.button("Graph IV Skew", key="skew_button", type="primary", disabled=(not expiry_input))
plot_placeholder_skew = st.empty()
# Skew Logic (remains the same)
if graph_skew_button:
    ticker_to_graph = st.session_state.ticker_input_skew; expiry_to_graph = st.session_state.expiry_input_skew
    if ticker_to_graph and expiry_to_graph:
        with st.spinner(f"Fetching option chain data for {ticker_to_graph} (Expiry: {expiry_to_graph})..."):
            calls, puts, price, error_msg_fetch = get_option_chain_data(ticker_to_graph, expiry_to_graph)
            if error_msg_fetch:
                st.error(error_msg_fetch); st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None
                st.session_state.ticker_calculated_skew = ticker_to_graph; st.session_state.expiry_calculated_skew = expiry_to_graph
            else:
                st.session_state.calls_data_skew = calls; st.session_state.puts_data_skew = puts; st.session_state.price_skew = price
                st.session_state.ticker_calculated_skew = ticker_to_graph; st.session_state.expiry_calculated_skew = expiry_to_graph
    else:
        st.warning("Please enter a ticker and select an expiration date.")
        st.session_state.ticker_calculated_skew = None; st.session_state.expiry_calculated_skew = None
        st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None
# Display Skew Plot (remains the same)
if st.session_state.get('ticker_calculated_skew') and st.session_state.get('expiry_calculated_skew'):
    with plot_placeholder_skew.container():
        fig_skew = create_iv_skew_plot(st.session_state.calls_data_skew, st.session_state.puts_data_skew, st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew, st.session_state.price_skew)
        st.plotly_chart(fig_skew, use_container_width=True)
else:
     with plot_placeholder_skew.container(): st.info("Enter an equity/ETF ticker, select an expiration date, and click 'Graph IV Skew'.")


# --- Section 3: FRED Economic Data (Single Series) ---
# (Keep existing code for Section 3)
st.divider()
st.header("🏛️ FRED Economic Data Viewer (Single Series)")
st.write("Fetches and displays a single time series from the FRED database.")
plot_placeholder_fred = st.empty()
caption_placeholder_fred = st.empty()
if fred is None:
     with plot_placeholder_fred.container(): st.error(fred_error_message or "FRED API client could not be initialized. Please configure your FRED_API_KEY in Streamlit Secrets.")
else:
    st.session_state.fred_series_name_input = st.selectbox(
        "Select FRED Series:", options=fred_series_options,
        index=fred_series_options.index(st.session_state.fred_series_name_input) if st.session_state.fred_series_name_input in fred_series_options else fred_series_options.index(DEFAULT_FRED_SERIES_NAME),
        key="fred_series_select_widget", help="Select an economic data series from FRED."
    )
    selected_series_id = FRED_SERIES_EXAMPLES.get(st.session_state.fred_series_name_input)
    fetch_fred_button = st.button("Fetch & Plot FRED Data", key="fred_button", type="primary")
    if fetch_fred_button:
        series_name_to_fetch = st.session_state.fred_series_name_input; series_id_to_fetch = selected_series_id
        if series_id_to_fetch:
            with st.spinner(f"Fetching data for {series_id_to_fetch} ({series_name_to_fetch})..."):
                 series_data, error_msg_data = get_fred_data(fred, series_id_to_fetch)
                 series_info, error_msg_info = get_fred_series_info(fred, series_id_to_fetch)
                 if error_msg_data:
                     st.error(error_msg_data); st.session_state.fred_data = None; st.session_state.fred_series_info = None
                     st.session_state.fred_series_id_calculated = series_id_to_fetch; st.session_state.fred_series_name_calculated = series_name_to_fetch
                 else:
                     st.session_state.fred_data = series_data; st.session_state.fred_series_info = series_info
                     st.session_state.fred_series_id_calculated = series_id_to_fetch; st.session_state.fred_series_name_calculated = series_name_to_fetch
                     if error_msg_info: st.warning(f"Successfully fetched data for {series_id_to_fetch}, but could not fetch metadata: {error_msg_info}")
        else:
            st.warning("Invalid series selection."); st.session_state.fred_series_id_calculated = None; st.session_state.fred_series_name_calculated = None; st.session_state.fred_data = None; st.session_state.fred_series_info = None
    if st.session_state.get('fred_series_id_calculated'):
        with plot_placeholder_fred.container():
            fig_fred = create_fred_plot(st.session_state.fred_data, st.session_state.fred_series_id_calculated, st.session_state.fred_series_info)
            st.plotly_chart(fig_fred, use_container_width=True)
        with caption_placeholder_fred.container():
             current_info = st.session_state.get('fred_series_info')
             if current_info is not None and not current_info.empty:
                 last_updated = current_info.get('last_updated', 'N/A'); notes = current_info.get('notes', 'N/A')
                 notes_display = (notes[:200] + '...') if notes and len(notes) > 200 else notes
                 st.caption(f"Last Updated: {last_updated}. Notes: {notes_display if notes_display else 'N/A'}")
             elif st.session_state.fred_data is not None: st.caption("Metadata not available for this series.")
    else:
        with plot_placeholder_fred.container(): st.info("Select a FRED economic data series and click 'Fetch & Plot FRED Data'.")


# --- Section 4: Fed's Jaws Chart ---
# (Keep existing code for Section 4)
st.divider()
st.header("🦅 Fed's Jaws: Key Policy Rates")
st.write(f"Visualizes key Federal Reserve interest rates over the last **{FED_JAWS_DURATION_DAYS} days**, including the target range (upper/lower bounds shown as dotted red lines).") # Updated description
plot_placeholder_jaws = st.empty()
if fred is None:
    with plot_placeholder_jaws.container():
        st.error(fred_error_message or "FRED API client is not initialized. Cannot display Fed's Jaws chart.")
else:
    fetch_jaws_button = st.button("Fetch/Refresh Fed's Jaws Data", key="jaws_button", type="primary")
    if fetch_jaws_button:
        with st.spinner(f"Fetching last {FED_JAWS_DURATION_DAYS} days of Fed's Jaws data from FRED..."): # Updated spinner message
            end_date_jaws = datetime.now()
            start_date_jaws = end_date_jaws - timedelta(days=FED_JAWS_DURATION_DAYS)
            jaws_data_result, error_msg_jaws = get_multiple_fred_data(
                _fred_instance=fred, series_ids=FED_JAWS_SERIES_IDS,
                start_date=start_date_jaws, end_date=end_date_jaws
                )
            if error_msg_jaws:
                st.error(f"Failed to fetch Fed's Jaws data: {error_msg_jaws}")
                st.session_state.fed_jaws_data = None
                st.session_state.fed_jaws_calculated = True
            else:
                st.session_state.fed_jaws_data = jaws_data_result
                st.session_state.fed_jaws_calculated = True
    if st.session_state.get('fed_jaws_calculated'):
         with plot_placeholder_jaws.container():
             fig_jaws = create_fed_jaws_plot(st.session_state.fed_jaws_data)
             st.plotly_chart(fig_jaws, use_container_width=True)
             st.caption(f"Data includes: {', '.join(FED_JAWS_SERIES_IDS)}. Target range limits (DFEDTARU, DFEDTARL) shown as dotted red lines.")
    else:
         with plot_placeholder_jaws.container():
             st.info(f"Click 'Fetch/Refresh Fed's Jaws Data' to load and display the chart for the last {FED_JAWS_DURATION_DAYS} days.")


# --- Ticker Reference Table & Footer ---
# (Keep existing code for Ticker Reference and Footer)
st.divider()
with st.expander("Show Example Ticker Symbols (Yahoo Finance)"):
    st.dataframe(
        ticker_df, use_container_width=True, hide_index=True,
        column_config={
             "Asset Class": st.column_config.TextColumn("Asset Class"),
             "Description": st.column_config.TextColumn("Description"),
             "Yahoo Ticker": st.column_config.TextColumn("Yahoo Ticker"),
        }
    )
st.divider()
linkedin_url = "https://www.linkedin.com/in/kennethquah/"
your_name = "Kenneth Quah"
linkedin_svg = """<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#0077B5" style="vertical-align: middle;"><title>LinkedIn</title><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>"""
footer_html = f"""
<div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 15px 0; color: grey; font-size: 0.875rem;">
    <span style="margin-right: 10px;">Created by {your_name}</span>
    <a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: grey; display: inline-flex; align-items: center;" title="LinkedIn Profile">
        {linkedin_svg}
    </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
st.caption(f"Market data sourced from Yahoo Finance via yfinance library. Economic data sourced from FRED® (Federal Reserve Economic Data) via fredapi library. Data may be delayed.")
