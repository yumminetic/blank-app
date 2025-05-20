# app.py
"""
Main Streamlit application file for the Financial Dashboard.
Handles UI, state management, and orchestrates calls to other modules.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback # For detailed error printing if needed directly in UI logic

# Import from custom modules
import config
import data_fetchers
import plotters
import utils

# --- Streamlit App ---
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📈 Financial Dashboard")

# --- Initialize Session State using utility function ---
# Correlation State
utils.init_state('rolling_corr_data', None)
utils.init_state('rolling_corr_error', None) # To store error messages from data fetcher
utils.init_state('ticker1_calculated_corr', None)
utils.init_state('ticker2_calculated_corr', None)
utils.init_state('ticker1_input_corr', config.DEFAULT_TICKER_1_CORR)
utils.init_state('ticker2_input_corr', config.DEFAULT_TICKER_2_CORR)

# Skew State
utils.init_state('calls_data_skew', None)
utils.init_state('puts_data_skew', None)
utils.init_state('price_skew', None)
utils.init_state('skew_error', None) # To store error messages
utils.init_state('ticker_calculated_skew', None)
utils.init_state('expiry_calculated_skew', None)
utils.init_state('ticker_input_skew', config.DEFAULT_TICKER_SKEW)
utils.init_state('expiry_input_skew', None) # Will be populated by selectbox

# FRED (Single Series) State
utils.init_state('fred_data', None)
utils.init_state('fred_series_info', None)
utils.init_state('fred_data_error', None)
utils.init_state('fred_info_error', None)
utils.init_state('fred_series_id_calculated', None)
utils.init_state('fred_series_name_calculated', None)
utils.init_state('fred_series_name_input', config.DEFAULT_FRED_SERIES_NAME)

# Fed's Jaws State
utils.init_state('fed_jaws_data', None)
utils.init_state('fed_jaws_error', None)
utils.init_state('fed_jaws_calculated', False)


# --- Section 1: Rolling Correlation ---
st.header("📊 Rolling Correlation Calculator")
st.write("Calculates the rolling correlation between the daily returns of two assets.")
col1_corr, col2_corr = st.columns(2)
with col1_corr:
    ticker1_input_val_corr = st.text_input(
        "Ticker 1:", value=st.session_state.ticker1_input_corr, key="ticker1_corr_widget",
        help="Enter a Yahoo Finance ticker symbol (e.g., AAPL, ^GSPC, EURUSD=X)."
    )
    st.session_state.ticker1_input_corr = ticker1_input_val_corr.strip().upper() if isinstance(ticker1_input_val_corr, str) else config.DEFAULT_TICKER_1_CORR
with col2_corr:
    ticker2_input_val_corr = st.text_input(
        "Ticker 2:", value=st.session_state.ticker2_input_corr, key="ticker2_corr_widget",
        help="Enter another Yahoo Finance ticker symbol (e.g., GLD, ^IXIC, BTC-USD)."
    )
    st.session_state.ticker2_input_corr = ticker2_input_val_corr.strip().upper() if isinstance(ticker2_input_val_corr, str) else config.DEFAULT_TICKER_2_CORR

calculate_corr_button = st.button("Calculate Correlation", key="corr_button", type="primary")
plot_placeholder_corr = st.empty()

if calculate_corr_button:
    t1_corr = st.session_state.ticker1_input_corr
    t2_corr = st.session_state.ticker2_input_corr
    st.session_state.rolling_corr_data = None # Reset previous data
    st.session_state.rolling_corr_error = None # Reset previous error

    if t1_corr and t2_corr:
        if t1_corr == t2_corr:
            st.warning("Please enter two different ticker symbols.")
            st.session_state.ticker1_calculated_corr = None # Clear calculated state
            st.session_state.ticker2_calculated_corr = None
        else:
            with st.spinner(f"Calculating {config.ROLLING_WINDOW}-day rolling correlation for {t1_corr} vs {t2_corr}..."):
                result_corr, error_msg_corr = data_fetchers.calculate_rolling_correlation(
                    t1_corr, t2_corr, window=config.ROLLING_WINDOW, years=config.YEARS_OF_DATA
                )
                st.session_state.rolling_corr_data = result_corr
                st.session_state.rolling_corr_error = error_msg_corr
                st.session_state.ticker1_calculated_corr = t1_corr # Set calculated tickers regardless of error for plot title
                st.session_state.ticker2_calculated_corr = t2_corr
                if error_msg_corr:
                    st.error(f"Correlation Error: {error_msg_corr}") # Display error from fetcher
    elif not t1_corr and not t2_corr:
        st.warning("Please enter ticker symbols in both fields.")
    else:
        st.warning("Please enter a ticker symbol in the missing field.")

# Display Correlation Plot
# This block always runs to show either the plot, an error message from the data fetcher, or the initial info message.
with plot_placeholder_corr.container():
    if st.session_state.get('ticker1_calculated_corr') and st.session_state.get('ticker2_calculated_corr'):
        # If an error occurred during data fetching, it's stored in rolling_corr_error
        # The plotter function is designed to handle None or empty data and display a message on the chart
        fig_corr = plotters.create_corr_plot(
            st.session_state.rolling_corr_data,
            st.session_state.ticker1_calculated_corr,
            st.session_state.ticker2_calculated_corr,
            window=config.ROLLING_WINDOW, # Pass from config
            years=config.YEARS_OF_DATA   # Pass from config
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        if st.session_state.rolling_corr_error and st.session_state.rolling_corr_data is None:
            # This is an additional textual error if the plot itself doesn't convey enough
            # st.error(f"Details: {st.session_state.rolling_corr_error}") # Already shown above
            pass # Error is shown above when button is clicked
    else:
        st.info("Enter two ticker symbols and click 'Calculate Correlation' to view the rolling correlation plot.")


# --- Section 2: Implied Volatility Skew ---
st.divider()
st.header("📉 Implied Volatility Skew Viewer")
st.write("Visualizes the Implied Volatility (IV) smile/skew for options of a selected equity or ETF.")

ticker_input_val_skew = st.text_input(
    "Equity/ETF Ticker:", value=st.session_state.ticker_input_skew, key="ticker_skew_widget",
    help="Enter a Yahoo Finance ticker for a stock or ETF with options (e.g., AAPL, SPY, TSLA)."
)
st.session_state.ticker_input_skew = ticker_input_val_skew.strip().upper() if isinstance(ticker_input_val_skew, str) else config.DEFAULT_TICKER_SKEW

expirations, error_msg_exp = data_fetchers.get_expiration_dates(st.session_state.ticker_input_skew)
selected_expiry_for_selectbox = st.session_state.expiry_input_skew # Retain previous selection if valid

if expirations:
    try:
        today = pd.Timestamp.now().normalize()
        exp_dates_ts = pd.to_datetime(expirations)
        future_dates_ts = exp_dates_ts[exp_dates_ts >= today]
        default_sel_index = 0

        if not future_dates_ts.empty:
            target_date = today + pd.Timedelta(days=90)
            closest_date_ts = future_dates_ts[np.abs((future_dates_ts - target_date).total_seconds()).argmin()]
            default_expiry_str = closest_date_ts.strftime('%Y-%m-%d')
            if default_expiry_str in expirations:
                default_sel_index = expirations.index(default_expiry_str)
        elif expirations: # Only past dates
            default_sel_index = len(expirations) -1

        # Try to keep the selected expiry if the ticker hasn't changed and the expiry is still valid
        current_selection_index = default_sel_index
        if st.session_state.ticker_input_skew == st.session_state.get('ticker_calculated_skew') and \
           st.session_state.get('expiry_calculated_skew') in expirations:
            current_selection_index = expirations.index(st.session_state.expiry_calculated_skew)
        
        # Update session state for selectbox from its own output
        st.session_state.expiry_input_skew = st.selectbox(
            "Select Expiration Date:", expirations, index=current_selection_index, key="expiry_select_widget",
            help="Choose the options contract expiration date."
        )
        selected_expiry_for_selectbox = st.session_state.expiry_input_skew

    except Exception as e:
        st.error(f"Error processing expiration dates: {e}")
        print(f"Error processing expiration dates: {e}\n{traceback.format_exc()}")
        # Fallback selectbox if smart defaulting fails
        st.session_state.expiry_input_skew = st.selectbox(
            "Select Expiration Date (fallback):", expirations, key="expiry_select_widget_fallback"
        )
        selected_expiry_for_selectbox = st.session_state.expiry_input_skew
elif st.session_state.ticker_input_skew: # Only show warning if a ticker has been input
    st.warning(error_msg_exp or f"Could not find options data or expirations for '{st.session_state.ticker_input_skew}'.")

graph_skew_button = st.button("Graph IV Skew", key="skew_button", type="primary", disabled=(not selected_expiry_for_selectbox))
plot_placeholder_skew = st.empty()

if graph_skew_button:
    ticker_to_graph_skew = st.session_state.ticker_input_skew
    expiry_to_graph_skew = st.session_state.expiry_input_skew # Use the value from selectbox's state
    st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None
    st.session_state.skew_error = None

    if ticker_to_graph_skew and expiry_to_graph_skew:
        with st.spinner(f"Fetching option chain data for {ticker_to_graph_skew} (Expiry: {expiry_to_graph_skew})..."):
            calls, puts, price, error_msg_fetch_skew = data_fetchers.get_option_chain_data(ticker_to_graph_skew, expiry_to_graph_skew)
            st.session_state.calls_data_skew = calls
            st.session_state.puts_data_skew = puts
            st.session_state.price_skew = price
            st.session_state.skew_error = error_msg_fetch_skew
            st.session_state.ticker_calculated_skew = ticker_to_graph_skew
            st.session_state.expiry_calculated_skew = expiry_to_graph_skew
            if error_msg_fetch_skew:
                st.error(f"Option Chain Error: {error_msg_fetch_skew}")
    else:
        st.warning("Please enter a ticker and select an expiration date.")
        st.session_state.ticker_calculated_skew = None; st.session_state.expiry_calculated_skew = None


with plot_placeholder_skew.container():
    if st.session_state.get('ticker_calculated_skew') and st.session_state.get('expiry_calculated_skew'):
        fig_skew = plotters.create_iv_skew_plot(
            st.session_state.calls_data_skew, st.session_state.puts_data_skew,
            st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew,
            st.session_state.price_skew
        )
        st.plotly_chart(fig_skew, use_container_width=True)
        # Error message is handled by the plotter or shown above on button click
    else:
        st.info("Enter an equity/ETF ticker, select an expiration date, and click 'Graph IV Skew'.")


# --- Section 3: FRED Economic Data (Single Series) ---
st.divider()
st.header("🏛️ FRED Economic Data Viewer (Single Series)")
st.write("Fetches and displays a single time series from the FRED database.")
plot_placeholder_fred = st.empty()
caption_placeholder_fred = st.empty()

if config.fred is None: # Check the fred instance from config
    with plot_placeholder_fred.container():
        st.error(config.fred_error_message or "FRED API client could not be initialized. Please configure your FRED_API_KEY in Streamlit Secrets.")
else:
    # Use the selectbox value directly from session_state to ensure persistence
    st.session_state.fred_series_name_input = st.selectbox(
        "Select FRED Series:", options=config.fred_series_options,
        index=config.fred_series_options.index(st.session_state.fred_series_name_input)
            if st.session_state.fred_series_name_input in config.fred_series_options
            else config.fred_series_options.index(config.DEFAULT_FRED_SERIES_NAME), # Fallback to default
        key="fred_series_select_widget", help="Select an economic data series from FRED."
    )
    selected_series_id_fred = config.FRED_SERIES_EXAMPLES.get(st.session_state.fred_series_name_input)
    fetch_fred_button = st.button("Fetch & Plot FRED Data", key="fred_button", type="primary")

    if fetch_fred_button:
        series_name_to_fetch_fred = st.session_state.fred_series_name_input
        series_id_to_fetch_fred = selected_series_id_fred
        st.session_state.fred_data = None; st.session_state.fred_series_info = None # Reset
        st.session_state.fred_data_error = None; st.session_state.fred_info_error = None

        if series_id_to_fetch_fred:
            with st.spinner(f"Fetching data for {series_id_to_fetch_fred} ({series_name_to_fetch_fred})..."):
                s_data, err_data = data_fetchers.get_fred_data(config.fred, series_id_to_fetch_fred)
                s_info, err_info = data_fetchers.get_fred_series_info(config.fred, series_id_to_fetch_fred)

                st.session_state.fred_data = s_data
                st.session_state.fred_series_info = s_info
                st.session_state.fred_data_error = err_data
                st.session_state.fred_info_error = err_info
                st.session_state.fred_series_id_calculated = series_id_to_fetch_fred
                st.session_state.fred_series_name_calculated = series_name_to_fetch_fred

                if err_data: st.error(f"FRED Data Error: {err_data}")
                if err_info: st.warning(f"FRED Metadata Error: {err_info}") # Warning for metadata
        else:
            st.warning("Invalid FRED series selection.")
            st.session_state.fred_series_id_calculated = None

    with plot_placeholder_fred.container():
        if st.session_state.get('fred_series_id_calculated'):
            fig_fred = plotters.create_fred_plot(
                st.session_state.fred_data,
                st.session_state.fred_series_id_calculated,
                st.session_state.fred_series_info
            )
            st.plotly_chart(fig_fred, use_container_width=True)
        else:
            st.info("Select a FRED economic data series and click 'Fetch & Plot FRED Data'.")

    with caption_placeholder_fred.container():
        if st.session_state.get('fred_series_id_calculated') and st.session_state.fred_series_info is not None:
            current_info_fred = st.session_state.fred_series_info
            if not current_info_fred.empty:
                last_updated = current_info_fred.get('last_updated', 'N/A')
                notes = current_info_fred.get('notes', 'N/A')
                notes_display = (notes[:200] + '...') if notes and len(notes) > 200 else notes
                st.caption(f"Last Updated: {last_updated}. Notes: {notes_display if notes_display else 'N/A'}")
            elif st.session_state.fred_data is not None: # Data exists but info might be empty/failed
                st.caption("Metadata might not be fully available for this series.")
        # No caption if nothing calculated yet


# --- Section 4: Fed's Jaws Chart ---
st.divider()
st.header("🦅 Fed's Jaws: Key Policy Rates")
st.write(f"Visualizes key Federal Reserve interest rates over the last **{config.FED_JAWS_DURATION_DAYS} days**.")
plot_placeholder_jaws = st.empty()

if config.fred is None:
    with plot_placeholder_jaws.container():
        st.error(config.fred_error_message or "FRED API client is not initialized. Cannot display Fed's Jaws chart.")
else:
    fetch_jaws_button = st.button("Fetch/Refresh Fed's Jaws Data", key="jaws_button", type="primary")

    if fetch_jaws_button:
        st.session_state.fed_jaws_data = None # Reset
        st.session_state.fed_jaws_error = None
        with st.spinner(f"Fetching last {config.FED_JAWS_DURATION_DAYS} days of Fed's Jaws data from FRED..."):
            end_date_jaws = datetime.now()
            start_date_jaws = end_date_jaws - timedelta(days=config.FED_JAWS_DURATION_DAYS)
            jaws_data_result, error_msg_jaws = data_fetchers.get_multiple_fred_data(
                _fred_instance=config.fred, series_ids=config.FED_JAWS_SERIES_IDS,
                start_date=start_date_jaws, end_date=end_date_jaws
            )
            st.session_state.fed_jaws_data = jaws_data_result
            st.session_state.fed_jaws_error = error_msg_jaws
            st.session_state.fed_jaws_calculated = True # Mark as attempted
            if error_msg_jaws:
                # If it's a partial error (some series failed but others fetched), data might still exist
                if jaws_data_result is not None:
                     st.warning(f"Fed's Jaws Data Warning: {error_msg_jaws}")
                else: # Total failure
                     st.error(f"Fed's Jaws Data Error: {error_msg_jaws}")


    with plot_placeholder_jaws.container():
        if st.session_state.get('fed_jaws_calculated'): # If button was clicked
            fig_jaws = plotters.create_fed_jaws_plot(st.session_state.fed_jaws_data) # Plotter handles None data
            st.plotly_chart(fig_jaws, use_container_width=True)
            if st.session_state.fed_jaws_data is not None and not st.session_state.fed_jaws_data.empty:
                st.caption(f"Data includes: {', '.join(config.FED_JAWS_SERIES_IDS)}. Target range limits (DFEDTARU, DFEDTARL) shown as dotted red lines.")
            elif st.session_state.fed_jaws_error and st.session_state.fed_jaws_data is None:
                 # Error already shown above, plot placeholder will show "no data"
                 pass
        else: # Before button is clicked
            st.info(f"Click 'Fetch/Refresh Fed's Jaws Data' to load and display the chart for the last {config.FED_JAWS_DURATION_DAYS} days.")


# --- Ticker Reference Table & Footer ---
st.divider()
with st.expander("Show Example Ticker Symbols (Yahoo Finance)"):
    st.dataframe(
        config.ticker_df, use_container_width=True, hide_index=True,
        column_config={
            "Asset Class": st.column_config.TextColumn("Asset Class"),
            "Description": st.column_config.TextColumn("Description"),
            "Yahoo Ticker": st.column_config.TextColumn("Yahoo Ticker"),
        }
    )
st.divider()
footer_html_content = utils.generate_footer_html(config.YOUR_NAME, config.LINKEDIN_URL, config.LINKEDIN_SVG)
st.markdown(footer_html_content, unsafe_allow_html=True)
st.caption("Market data sourced from Yahoo Finance via yfinance library. Economic data sourced from FRED® (Federal Reserve Economic Data) via fredapi library. Data may be delayed.")

