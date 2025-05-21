# streamlit_app.py
"""
Main Streamlit application file for the Macro/Quantamental Dashboard.
Handles UI, state management, and orchestrates calls to other modules.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback 

import config # Assuming config.py is in the same directory
import data_fetchers # Assuming data_fetchers.py is in the same directory
import plotters # Assuming plotters.py is in the same directory
import utils # Assuming utils.py is in the same directory

# --- Streamlit App ---
st.set_page_config(page_title="Macro/Quantamental Dashboard", layout="wide") # Updated page title
st.title("📊 Macro/Quantamental Dashboard") # Updated main title

# --- Initialize Session State ---
# Correlation State
utils.init_state('rolling_corr_data', None)
utils.init_state('rolling_corr_error', None) 
utils.init_state('ticker1_calculated_corr', None)
utils.init_state('ticker2_calculated_corr', None)
utils.init_state('ticker1_input_corr', config.DEFAULT_TICKER_1_CORR)
utils.init_state('ticker2_input_corr', config.DEFAULT_TICKER_2_CORR)
utils.init_state('rolling_window_corr', config.DEFAULT_ROLLING_WINDOW_CORR) # For the slider
utils.init_state('corr_window_calculated', config.DEFAULT_ROLLING_WINDOW_CORR) # To store the window used for the last plot

# Skew State
utils.init_state('calls_data_skew', None)
utils.init_state('puts_data_skew', None)
utils.init_state('price_skew', None)
utils.init_state('skew_error', None) 
utils.init_state('ticker_calculated_skew', None)
utils.init_state('expiry_calculated_skew', None)
utils.init_state('ticker_input_skew', config.DEFAULT_TICKER_SKEW)
utils.init_state('expiry_input_skew', None) 

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

# Fed Funds vs Core PCE State
utils.init_state('ffr_pce_data', None) 
utils.init_state('ffr_pce_error', None)
utils.init_state('ffr_pce_calculated', False)
utils.init_state('current_ffr_pce_diff', None)


# --- Section 1: Rolling Correlation ---
st.header("📊 Rolling Correlation Calculator")
st.write("Calculates the rolling correlation between the daily returns of two assets.")

# Input for rolling window
# The value from the slider is automatically stored in st.session_state.rolling_window_corr due to the key
st.slider(
    "Select Rolling Window (Days):",
    min_value=config.MIN_ROLLING_WINDOW_CORR,
    max_value=config.MAX_ROLLING_WINDOW_CORR,
    value=st.session_state.rolling_window_corr, 
    step=1,
    key="rolling_window_corr" # This key links the slider to st.session_state.rolling_window_corr
)

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
    selected_rolling_window = st.session_state.rolling_window_corr # Get current window from slider state

    st.session_state.rolling_corr_data = None 
    st.session_state.rolling_corr_error = None 
    st.session_state.ticker1_calculated_corr = t1_corr 
    st.session_state.ticker2_calculated_corr = t2_corr
    st.session_state.corr_window_calculated = selected_rolling_window # Store the window used for this specific calculation


    if t1_corr and t2_corr:
        if t1_corr == t2_corr:
            st.warning("Please enter two different ticker symbols.")
            st.session_state.ticker1_calculated_corr = None # Clear if invalid for plot
            st.session_state.ticker2_calculated_corr = None
        else:
            with st.spinner(f"Calculating {selected_rolling_window}-day rolling correlation for {t1_corr} vs {t2_corr}..."):
                result_corr, error_msg_corr = data_fetchers.calculate_rolling_correlation(
                    t1_corr, t2_corr, window=selected_rolling_window, years=config.YEARS_OF_DATA_CORR
                )
                st.session_state.rolling_corr_data = result_corr
                st.session_state.rolling_corr_error = error_msg_corr
                if error_msg_corr:
                    plot_placeholder_corr.error(f"Correlation Error: {error_msg_corr}")
    elif not t1_corr and not t2_corr:
        st.warning("Please enter ticker symbols in both fields.")
        st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None
    else:
        st.warning("Please enter a ticker symbol in the missing field.")
        st.session_state.ticker1_calculated_corr = None; st.session_state.ticker2_calculated_corr = None

with plot_placeholder_corr.container(): 
    if st.session_state.get('ticker1_calculated_corr') and st.session_state.get('ticker2_calculated_corr'):
        # Plot if no error during fetch, or if data exists despite a warning (partial fetch)
        if not st.session_state.rolling_corr_error or st.session_state.rolling_corr_data is not None: 
            fig_corr = plotters.create_corr_plot(
                st.session_state.rolling_corr_data,
                st.session_state.ticker1_calculated_corr,
                st.session_state.ticker2_calculated_corr,
                window=st.session_state.corr_window_calculated, # Use the window that was actually used for calculation
                years=config.YEARS_OF_DATA_CORR
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        # If there was an error and no data, the error message is already shown by the button logic above.
    else: # Initial state or after invalid input
        st.info("Select a rolling window, enter two ticker symbols, and click 'Calculate Correlation'.")


# --- Section 2: Implied Volatility Skew ---
st.divider()
st.header("📉 Implied Volatility Skew Viewer")
st.write("Visualizes the Implied Volatility (IV) smile/skew for options of a selected equity or ETF. Zero IV values are interpolated for smoother visualization.")

ticker_input_val_skew = st.text_input(
    "Equity/ETF Ticker:", value=st.session_state.ticker_input_skew, key="ticker_skew_widget",
    help="Enter a Yahoo Finance ticker for a stock or ETF with options (e.g., AAPL, SPY, TSLA)."
)
st.session_state.ticker_input_skew = ticker_input_val_skew.strip().upper() if isinstance(ticker_input_val_skew, str) else config.DEFAULT_TICKER_SKEW

expirations, error_msg_exp = data_fetchers.get_expiration_dates(st.session_state.ticker_input_skew)
selected_expiry_for_selectbox = None 

if error_msg_exp: 
    if st.session_state.ticker_input_skew: 
        st.warning(error_msg_exp)

if expirations: 
    try:
        today = pd.Timestamp.now().normalize()
        exp_dates_ts = pd.to_datetime(expirations)
        future_dates_ts = exp_dates_ts[exp_dates_ts >= today]
        default_sel_index = 0

        if not future_dates_ts.empty:
            target_date = today + pd.Timedelta(days=90)
            closest_date_idx_in_future = np.abs((future_dates_ts - target_date).total_seconds()).argmin()
            closest_date_ts = future_dates_ts[closest_date_idx_in_future]
            default_expiry_str = closest_date_ts.strftime('%Y-%m-%d')
            if default_expiry_str in expirations:
                default_sel_index = expirations.index(default_expiry_str)
        elif expirations: 
            default_sel_index = len(expirations) - 1
        
        current_selection_index = default_sel_index 
        if st.session_state.ticker_input_skew == st.session_state.get('ticker_calculated_skew') and \
           st.session_state.get('expiry_calculated_skew') in expirations:
            current_selection_index = expirations.index(st.session_state.expiry_calculated_skew)
        
        st.session_state.expiry_input_skew = st.selectbox(
            "Select Expiration Date:", expirations, index=current_selection_index, key="expiry_select_widget_skew", 
            help="Choose the options contract expiration date."
        )
        selected_expiry_for_selectbox = st.session_state.expiry_input_skew

    except Exception as e:
        st.error(f"Error processing expiration dates: {e}")
        print(f"Error processing expiration dates: {e}\n{traceback.format_exc()}")
        if expirations: 
             st.session_state.expiry_input_skew = st.selectbox(
                "Select Expiration Date (fallback):", expirations, key="expiry_select_widget_fallback_skew" 
            )
             selected_expiry_for_selectbox = st.session_state.expiry_input_skew
elif st.session_state.ticker_input_skew and not error_msg_exp: 
     st.info(f"No option expiration dates found for '{st.session_state.ticker_input_skew}'. It might not have options or data is temporarily unavailable.")


graph_skew_button = st.button("Graph IV Skew", key="skew_button", type="primary", disabled=(not selected_expiry_for_selectbox))
plot_placeholder_skew = st.empty()

if graph_skew_button:
    ticker_to_graph_skew = st.session_state.ticker_input_skew
    expiry_to_graph_skew = selected_expiry_for_selectbox

    st.session_state.calls_data_skew = None; st.session_state.puts_data_skew = None; st.session_state.price_skew = None
    st.session_state.skew_error = None
    st.session_state.ticker_calculated_skew = ticker_to_graph_skew 
    st.session_state.expiry_calculated_skew = expiry_to_graph_skew 

    if ticker_to_graph_skew and expiry_to_graph_skew:
        with st.spinner(f"Fetching option chain data for {ticker_to_graph_skew} (Expiry: {expiry_to_graph_skew})..."):
            calls, puts, price, error_msg_fetch_skew = data_fetchers.get_option_chain_data(ticker_to_graph_skew, expiry_to_graph_skew)
            st.session_state.calls_data_skew = calls
            st.session_state.puts_data_skew = puts
            st.session_state.price_skew = price
            st.session_state.skew_error = error_msg_fetch_skew
            if error_msg_fetch_skew:
                plot_placeholder_skew.error(f"Option Chain Error: {error_msg_fetch_skew}")
    else: 
        st.warning("Please enter a ticker and select a valid expiration date.")
        st.session_state.ticker_calculated_skew = None; st.session_state.expiry_calculated_skew = None


with plot_placeholder_skew.container():
    if st.session_state.get('ticker_calculated_skew') and st.session_state.get('expiry_calculated_skew'):
        if not st.session_state.skew_error or \
           (st.session_state.calls_data_skew is not None or st.session_state.puts_data_skew is not None):
            # The create_iv_skew_plot function now handles interpolation internally
            fig_skew = plotters.create_iv_skew_plot(
                st.session_state.calls_data_skew, st.session_state.puts_data_skew,
                st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew,
                st.session_state.price_skew
            )
            st.plotly_chart(fig_skew, use_container_width=True)
    else:
        st.info("Enter an equity/ETF ticker, select an expiration date, and click 'Graph IV Skew'.")


# --- Section 3: FRED Economic Data (Single Series) ---
st.divider()
st.header("🏛️ FRED Economic Data Viewer (Single Series)")
st.write("Fetches and displays a single time series from the FRED database.")
plot_placeholder_fred = st.empty()
caption_placeholder_fred = st.empty()

if config.fred is None: 
    with plot_placeholder_fred.container(): 
        st.error(config.fred_error_message or "FRED API client could not be initialized. Please configure your FRED_API_KEY in Streamlit Secrets.")
else:
    try:
        current_fred_series_name_index = config.fred_series_options.index(st.session_state.fred_series_name_input)
    except ValueError:
        current_fred_series_name_index = config.fred_series_options.index(config.DEFAULT_FRED_SERIES_NAME) 
        st.session_state.fred_series_name_input = config.DEFAULT_FRED_SERIES_NAME 

    st.session_state.fred_series_name_input = st.selectbox(
        "Select FRED Series:", options=config.fred_series_options,
        index=current_fred_series_name_index,
        key="fred_series_select_widget", help="Select an economic data series from FRED."
    )
    selected_series_id_fred = config.FRED_SERIES_EXAMPLES.get(st.session_state.fred_series_name_input)
    fetch_fred_button = st.button("Fetch & Plot FRED Data", key="fred_button", type="primary")

    if fetch_fred_button:
        series_name_to_fetch_fred = st.session_state.fred_series_name_input
        series_id_to_fetch_fred = selected_series_id_fred
        st.session_state.fred_data = None; st.session_state.fred_series_info = None
        st.session_state.fred_data_error = None; st.session_state.fred_info_error = None
        st.session_state.fred_series_id_calculated = series_id_to_fetch_fred
        st.session_state.fred_series_name_calculated = series_name_to_fetch_fred

        if series_id_to_fetch_fred:
            with st.spinner(f"Fetching data for {series_id_to_fetch_fred} ({series_name_to_fetch_fred})..."):
                s_data, err_data = data_fetchers.get_fred_data(config.fred, series_id_to_fetch_fred)
                s_info, err_info = data_fetchers.get_fred_series_info(config.fred, series_id_to_fetch_fred)

                st.session_state.fred_data = s_data
                st.session_state.fred_series_info = s_info
                st.session_state.fred_data_error = err_data
                st.session_state.fred_info_error = err_info

                if err_data: 
                    plot_placeholder_fred.error(f"FRED Data Error: {err_data}")
                if err_info: 
                    caption_placeholder_fred.warning(f"FRED Metadata Warning: {err_info}")
        else:
            st.warning("Invalid FRED series selection.")
            st.session_state.fred_series_id_calculated = None 

    with plot_placeholder_fred.container():
        if st.session_state.get('fred_series_id_calculated'):
            if not st.session_state.fred_data_error or st.session_state.fred_data is not None:
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
                notes_display = (notes[:250] + '...') if notes and len(notes) > 250 else notes 
                st.caption(f"Last Updated: {last_updated}. Notes: {notes_display if notes_display else 'N/A'}")
            elif st.session_state.fred_data is not None and not st.session_state.fred_info_error : 
                st.caption("Metadata is available but some fields might be empty for this series.")

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
        st.session_state.fed_jaws_data = None 
        st.session_state.fed_jaws_error = None
        st.session_state.fed_jaws_calculated = True 

        with st.spinner(f"Fetching last {config.FED_JAWS_DURATION_DAYS} days of Fed's Jaws data from FRED..."):
            end_date_jaws = datetime.now()
            start_date_jaws = end_date_jaws - timedelta(days=config.FED_JAWS_DURATION_DAYS)
            jaws_data_result, error_msg_jaws = data_fetchers.get_multiple_fred_data(
                _fred_instance=config.fred, series_ids=config.FED_JAWS_SERIES_IDS,
                start_date=start_date_jaws, end_date=end_date_jaws
            )
            st.session_state.fed_jaws_data = jaws_data_result
            st.session_state.fed_jaws_error = error_msg_jaws
            if error_msg_jaws:
                if jaws_data_result is not None and not jaws_data_result.empty: 
                     plot_placeholder_jaws.warning(f"Fed's Jaws Data Warning: {error_msg_jaws}")
                else: 
                     plot_placeholder_jaws.error(f"Fed's Jaws Data Error: {error_msg_jaws}")

    with plot_placeholder_jaws.container():
        if st.session_state.get('fed_jaws_calculated'): 
            if not (st.session_state.fed_jaws_error and (st.session_state.fed_jaws_data is None or st.session_state.fed_jaws_data.empty)) :
                fig_jaws = plotters.create_fed_jaws_plot(st.session_state.fed_jaws_data) 
                st.plotly_chart(fig_jaws, use_container_width=True)
                if st.session_state.fed_jaws_data is not None and not st.session_state.fed_jaws_data.empty:
                    st.caption(f"Data includes: {', '.join(config.FED_JAWS_SERIES_IDS)}. Target range limits (DFEDTARU, DFEDTARL) shown as dotted red lines.")
        else: 
            st.info(f"Click 'Fetch/Refresh Fed's Jaws Data' to load and display the chart for the last {config.FED_JAWS_DURATION_DAYS} days.")


# --- Section 5: Fed Funds Rate vs Core PCE ---
st.divider()
st.header("💰 Fed Funds Rate vs. Core PCE Inflation")
st.write(f"Visualizes the monthly Effective Federal Funds Rate against monthly Core PCE year-over-year inflation (calculated from PCEPILFE index). The gap between the two series is colored red if FFR - Core PCE YoY > {config.FFR_PCE_THRESHOLD}%, potentially indicating heightened recession risk.")

plot_placeholder_ffr_pce = st.empty()
current_diff_placeholder_ffr_pce = st.empty() 

if config.fred is None:
    with plot_placeholder_ffr_pce.container(): 
        st.error(config.fred_error_message or "FRED API client is not initialized. Cannot display this chart.")
else:
    fetch_ffr_pce_button = st.button("Fetch/Refresh Fed Funds vs Core PCE Data", key="ffr_pce_button", type="primary")

    if fetch_ffr_pce_button:
        st.session_state.ffr_pce_data = None 
        st.session_state.ffr_pce_error = None 
        st.session_state.ffr_pce_calculated = True 
        st.session_state.current_ffr_pce_diff = None 

        series_to_fetch_ffr_pce = [
            config.FFR_VS_PCE_SERIES_IDS["ffr"], 
            config.FFR_VS_PCE_SERIES_IDS["core_pce_index"] 
        ]
        with st.spinner("Fetching Fed Funds Rate and Core PCE Index data from FRED..."):
            ffr_pce_data_result, error_msg_ffr_pce = data_fetchers.get_multiple_fred_data(
                _fred_instance=config.fred,
                series_ids=series_to_fetch_ffr_pce
            )
            st.session_state.ffr_pce_data = ffr_pce_data_result 
            st.session_state.ffr_pce_error = error_msg_ffr_pce

            if error_msg_ffr_pce:
                if ffr_pce_data_result is not None and not ffr_pce_data_result.empty:
                    plot_placeholder_ffr_pce.warning(f"FFR vs PCE Data Warning: {error_msg_ffr_pce}")
                else: 
                    plot_placeholder_ffr_pce.error(f"FFR vs PCE Data Error: {error_msg_ffr_pce}")
            
            if ffr_pce_data_result is not None and not ffr_pce_data_result.empty:
                temp_df_ffr_pce = ffr_pce_data_result.copy()
                ffr_col_name = config.FFR_VS_PCE_SERIES_IDS["ffr"]
                pce_index_col_name = config.FFR_VS_PCE_SERIES_IDS["core_pce_index"]
                
                if ffr_col_name in temp_df_ffr_pce.columns and pce_index_col_name in temp_df_ffr_pce.columns:
                    temp_df_ffr_pce.sort_index(inplace=True) 
                    pce_yoy_calculated_col_name = 'PCE_YoY_Calculated' 
                    temp_df_ffr_pce[pce_yoy_calculated_col_name] = temp_df_ffr_pce[pce_index_col_name].pct_change(periods=12) * 100
                    
                    temp_df_ffr_pce.dropna(subset=[ffr_col_name, pce_yoy_calculated_col_name], inplace=True)
                    
                    if not temp_df_ffr_pce.empty:
                        latest_ffr_val = temp_df_ffr_pce[ffr_col_name].iloc[-1]
                        latest_pce_yoy_val = temp_df_ffr_pce[pce_yoy_calculated_col_name].iloc[-1]
                        st.session_state.current_ffr_pce_diff = latest_ffr_val - latest_pce_yoy_val
                    else:
                        st.session_state.current_ffr_pce_diff = "N/A (No overlapping data after YoY calc)"
                else:
                    missing_cols_str = []
                    if ffr_col_name not in temp_df_ffr_pce.columns: missing_cols_str.append(ffr_col_name)
                    if pce_index_col_name not in temp_df_ffr_pce.columns: missing_cols_str.append(pce_index_col_name) 
                    st.session_state.current_ffr_pce_diff = f"N/A (Missing source columns: {', '.join(missing_cols_str)})" 
            elif not error_msg_ffr_pce: 
                 st.session_state.current_ffr_pce_diff = "N/A (Source data series empty)"

    with plot_placeholder_ffr_pce.container():
        if st.session_state.get('ffr_pce_calculated'): 
            if not (st.session_state.ffr_pce_error and (st.session_state.ffr_pce_data is None or st.session_state.ffr_pce_data.empty)):
                fig_ffr_pce = plotters.create_ffr_pce_comparison_plot(
                    st.session_state.ffr_pce_data, 
                    ffr_series_id=config.FFR_VS_PCE_SERIES_IDS["ffr"],
                    pce_index_series_id=config.FFR_VS_PCE_SERIES_IDS["core_pce_index"], 
                    threshold=config.FFR_PCE_THRESHOLD
                )
                st.plotly_chart(fig_ffr_pce, use_container_width=True)
        else: 
            st.info("Click 'Fetch/Refresh Fed Funds vs Core PCE Data' to load and display the chart.")

    with current_diff_placeholder_ffr_pce.container():
        if st.session_state.get('ffr_pce_calculated') and st.session_state.current_ffr_pce_diff is not None:
            if isinstance(st.session_state.current_ffr_pce_diff, (float, int, np.number)): 
                st.metric(
                    label=f"Latest Difference ({config.FFR_VS_PCE_NAMES['ffr']} - Core PCE YoY Calc.)", 
                    value=f"{st.session_state.current_ffr_pce_diff:.2f}%"
                )
                if st.session_state.current_ffr_pce_diff > config.FFR_PCE_THRESHOLD:
                    st.caption(f"⚠️ Difference is above the {config.FFR_PCE_THRESHOLD}% threshold.")
                else:
                    st.caption(f"✅ Difference is at or below the {config.FFR_PCE_THRESHOLD}% threshold.")
            else: 
                st.write(f"**Latest Difference:** {st.session_state.current_ffr_pce_diff}")


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

