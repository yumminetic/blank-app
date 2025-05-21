# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date 
import traceback 
import yfinance as yf # Import yfinance
import config, data_fetchers, plotters, utils

st.set_page_config(page_title="Macro/Quantamental Dashboard", layout="wide")
st.title("📊 Macro/Quantamental Dashboard")

# --- Initialize Session State ---
default_fred_start = date(2000, 1, 1) 
default_fred_end = date.today()

# Correlation
utils.init_state('rolling_corr_data', None); utils.init_state('rolling_corr_error', None) 
utils.init_state('ticker1_calculated_corr', None); utils.init_state('ticker2_calculated_corr', None)
utils.init_state('ticker1_input_corr', config.DEFAULT_TICKER_1_CORR); utils.init_state('ticker2_input_corr', config.DEFAULT_TICKER_2_CORR)
utils.init_state('rolling_window_corr', config.DEFAULT_ROLLING_WINDOW_CORR)
utils.init_state('corr_window_calculated', config.DEFAULT_ROLLING_WINDOW_CORR)

# Skew
utils.init_state('calls_data_skew', None); utils.init_state('puts_data_skew', None); utils.init_state('price_skew', None)
utils.init_state('skew_error', None); utils.init_state('ticker_calculated_skew', None); utils.init_state('expiry_calculated_skew', None)
utils.init_state('ticker_input_skew', config.DEFAULT_TICKER_SKEW); utils.init_state('expiry_input_skew', None) 

# FRED Single
utils.init_state('fred_single_data', None); utils.init_state('fred_single_error', None); utils.init_state('fred_single_calculated', False) 
utils.init_state('fred_series_info', None); utils.init_state('fred_info_error', None); 
utils.init_state('fred_series_id_calculated', None); utils.init_state('fred_series_name_calculated', None) 
utils.init_state('fred_series_name_input', config.DEFAULT_FRED_SERIES_NAME) 
utils.init_state('fred_single_start_date', default_fred_start) 
utils.init_state('fred_single_end_date', default_fred_end)     
utils.init_state('fred_single_show_recession', True)        

# Fed Jaws
utils.init_state('fed_jaws_data', None); utils.init_state('fed_jaws_error', None); utils.init_state('fed_jaws_calculated', False)
utils.init_state('fed_jaws_show_recession', True) 

# FFR vs PCE
utils.init_state('ffr_pce_data', None); utils.init_state('ffr_pce_error', None); utils.init_state('ffr_pce_calculated', False)
utils.init_state('current_ffr_pce_diff', None)
utils.init_state('ffr_pce_start_date', default_fred_start) 
utils.init_state('ffr_pce_end_date', default_fred_end)     
utils.init_state('ffr_pce_show_recession', True)        

# Gold vs Real Yield
utils.init_state('gold_ry_data', None); utils.init_state('gold_ry_error', None); utils.init_state('gold_ry_calculated', False)
# latest_gold and latest_real_yield will be derived when displaying metrics
utils.init_state('gold_ry_start_date', default_fred_start) 
utils.init_state('gold_ry_end_date', default_fred_end)     
utils.init_state('gold_ry_show_recession', True)        
utils.init_state('yfinance_gold_col_name_plotter', None) # To pass the correct gold column name to plotter

# --- Helper for Date Inputs ---
def date_input_cols(start_date_key, end_date_key, section_key_suffix):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state[start_date_key] = st.date_input("Start Date", value=st.session_state[start_date_key], key=f"date_widget_start_{section_key_suffix}")
    with col2:
        st.session_state[end_date_key] = st.date_input("End Date", value=st.session_state[end_date_key], key=f"date_widget_end_{section_key_suffix}")
    return st.session_state[start_date_key], st.session_state[end_date_key]

# --- Section 1: Rolling Correlation ---
st.header("📊 Rolling Correlation Calculator")
st.write("Calculates the rolling correlation between the daily returns of two assets.")
st.slider("Select Rolling Window (Days):", min_value=config.MIN_ROLLING_WINDOW_CORR, max_value=config.MAX_ROLLING_WINDOW_CORR, value=st.session_state.rolling_window_corr, step=1, key="rolling_window_corr")
col1_corr, col2_corr = st.columns(2)
with col1_corr: st.session_state.ticker1_input_corr = st.text_input("Ticker 1:", value=st.session_state.ticker1_input_corr, key="ticker1_corr_widget").strip().upper()
with col2_corr: st.session_state.ticker2_input_corr = st.text_input("Ticker 2:", value=st.session_state.ticker2_input_corr, key="ticker2_corr_widget").strip().upper()
plot_placeholder_corr = st.empty()
if st.button("Calculate Correlation", key="corr_button", type="primary"):
    t1, t2, win = st.session_state.ticker1_input_corr, st.session_state.ticker2_input_corr, st.session_state.rolling_window_corr
    st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr, st.session_state.corr_window_calculated = t1, t2, win
    if t1 and t2 and t1 != t2:
        with st.spinner(f"Calculating {win}-day correlation for {t1} vs {t2}..."):
            st.session_state.rolling_corr_data, st.session_state.rolling_corr_error = data_fetchers.calculate_rolling_correlation(t1, t2, win)
            if st.session_state.rolling_corr_error: plot_placeholder_corr.error(f"Corr Error: {st.session_state.rolling_corr_error}")
    else: st.warning("Enter two different tickers.")
with plot_placeholder_corr.container(): 
    if st.session_state.ticker1_calculated_corr and st.session_state.rolling_corr_data is not None:
        fig_corr = plotters.create_corr_plot(st.session_state.rolling_corr_data, st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr, st.session_state.corr_window_calculated)
        st.plotly_chart(fig_corr, use_container_width=True)
        df_to_download_corr = st.session_state.rolling_corr_data
        if isinstance(df_to_download_corr, pd.Series): df_to_download_corr = df_to_download_corr.to_frame()
        if df_to_download_corr is not None and not df_to_download_corr.empty:
            csv_corr = utils.convert_df_to_csv(df_to_download_corr) 
            st.download_button(label="Download Correlation Data (CSV)", data=csv_corr, file_name=f"corr_{st.session_state.ticker1_calculated_corr}_{st.session_state.ticker2_calculated_corr}_{st.session_state.corr_window_calculated}d.csv", mime='text/csv', key="download_corr_csv")
    elif st.session_state.rolling_corr_error and st.session_state.ticker1_calculated_corr : pass 
    else: st.info("Select window, enter tickers, and click 'Calculate Correlation'.")

# --- Section 2: Implied Volatility Skew ---
st.divider(); st.header("📉 Implied Volatility Skew Viewer")
st.write("Visualizes IV smile/skew. Zero IVs are interpolated.")
st.session_state.ticker_input_skew = st.text_input("Equity/ETF Ticker:", value=st.session_state.ticker_input_skew, key="ticker_skew_widget").strip().upper()
expirations, err_exp = data_fetchers.get_expiration_dates(st.session_state.ticker_input_skew)
sel_exp = None
if err_exp: st.warning(err_exp)
if expirations:
    try: 
        today = pd.Timestamp.now().normalize(); future_dates = pd.to_datetime(expirations)[pd.to_datetime(expirations) >= today]
        def_idx = np.abs((future_dates - (today + pd.Timedelta(days=90))).total_seconds()).argmin() if not future_dates.empty else (len(expirations) -1 if expirations else 0)
        curr_idx = expirations.index(st.session_state.expiry_calculated_skew) if st.session_state.ticker_input_skew == st.session_state.ticker_calculated_skew and st.session_state.expiry_calculated_skew in expirations else def_idx
        sel_exp = st.selectbox("Select Expiration Date:", expirations, index=curr_idx, key="expiry_select_widget_skew")
    except Exception as e: sel_exp = st.selectbox("Select Expiration Date (fallback):", expirations, key="expiry_select_fb_skew") if expirations else None
elif st.session_state.ticker_input_skew and not err_exp: st.info(f"No options for '{st.session_state.ticker_input_skew}'.")
plot_placeholder_skew = st.empty()
if st.button("Graph IV Skew", key="skew_button", type="primary", disabled=(not sel_exp)):
    st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew = st.session_state.ticker_input_skew, sel_exp
    with st.spinner(f"Fetching options for {st.session_state.ticker_calculated_skew} ({st.session_state.expiry_calculated_skew})..."):
        calls, puts, price, err = data_fetchers.get_option_chain_data(st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew)
        st.session_state.calls_data_skew, st.session_state.puts_data_skew, st.session_state.price_skew, st.session_state.skew_error = calls, puts, price, err
        if err: plot_placeholder_skew.error(f"Options Error: {err}")
with plot_placeholder_skew.container():
    if st.session_state.ticker_calculated_skew and (st.session_state.calls_data_skew is not None or st.session_state.puts_data_skew is not None):
        fig_skew = plotters.create_iv_skew_plot(st.session_state.calls_data_skew, st.session_state.puts_data_skew, st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew, st.session_state.price_skew)
        st.plotly_chart(fig_skew, use_container_width=True)
        if st.session_state.calls_data_skew is not None and not st.session_state.calls_data_skew.empty:
            csv_calls = utils.convert_df_to_csv(st.session_state.calls_data_skew)
            st.download_button(label="Download Calls Data (CSV)", data=csv_calls, file_name=f"calls_{st.session_state.ticker_calculated_skew}_{st.session_state.expiry_calculated_skew}.csv", mime='text/csv', key="download_calls_csv")
        if st.session_state.puts_data_skew is not None and not st.session_state.puts_data_skew.empty:
            csv_puts = utils.convert_df_to_csv(st.session_state.puts_data_skew)
            st.download_button(label="Download Puts Data (CSV)", data=csv_puts, file_name=f"puts_{st.session_state.ticker_calculated_skew}_{st.session_state.expiry_calculated_skew}.csv", mime='text/csv', key="download_puts_csv")
    elif st.session_state.skew_error and st.session_state.ticker_calculated_skew: pass
    else: st.info("Enter ticker, select expiry, and click 'Graph IV Skew'.")

# --- FRED Sections Common UI Elements ---
def fred_section_ui(section_key_suffix, title, description, series_fetch_logic, plot_function, display_metrics_logic=None, series_select_options=None, default_series_name_key=None, series_id_map=None):
    st.divider(); st.header(title); st.write(description)
    start_date_key, end_date_key = f"{section_key_suffix}_start_date", f"{section_key_suffix}_end_date"
    show_recession_key = f"{section_key_suffix}_show_recession"
    data_key, error_key, calculated_key = f"{section_key_suffix}_data", f"{section_key_suffix}_error", f"{section_key_suffix}_calculated"
    
    date_input_cols(start_date_key, end_date_key, section_key_suffix) 
    st.checkbox("Show NBER Recession Bands", value=st.session_state[show_recession_key], key=f"recession_cb_{section_key_suffix}")

    if series_select_options: 
        try: idx = series_select_options.index(st.session_state[default_series_name_key])
        except ValueError: idx = 0; st.session_state[default_series_name_key] = series_select_options[0]
        st.session_state[default_series_name_key] = st.selectbox("Select FRED Series:", options=series_select_options, index=idx, key=f"selectbox_widget_{section_key_suffix}")
    
    plot_placeholder = st.empty(); metric_placeholder = st.empty()

    if st.button(f"Fetch/Refresh {title} Data", key=f"fetch_button_{section_key_suffix}", type="primary"):
        st.session_state[data_key], st.session_state[error_key] = None, None 
        st.session_state[calculated_key] = True 
        series_fetch_logic(start_date_key, end_date_key, show_recession_key, data_key, error_key, section_key_suffix, series_id_map, default_series_name_key)
        if st.session_state[error_key]: plot_placeholder.error(f"Data Error: {st.session_state[error_key]}")

    with plot_placeholder.container():
        if st.session_state[calculated_key] and st.session_state.get(data_key) is not None:
            # For Gold vs RY, we need to pass the specific gold column name derived from yfinance
            yfinance_gold_col_name_for_plotter = st.session_state.get('yfinance_gold_col_name_plotter') if section_key_suffix == "gold_ry" else None
            
            fig = plot_function(
                st.session_state[data_key], 
                st.session_state[show_recession_key], 
                section_key_suffix, series_id_map, default_series_name_key,
                yfinance_gold_col_name_for_plotter # Pass it here
            )
            if fig: st.plotly_chart(fig, use_container_width=True)
            df_to_download = st.session_state.get(data_key)
            if df_to_download is not None and not df_to_download.empty:
                csv_data = utils.convert_df_to_csv(df_to_download)
                st.download_button(label=f"Download {title} Data (CSV)", data=csv_data, file_name=f"{section_key_suffix}_data.csv", mime='text/csv', key=f"download_csv_{section_key_suffix}")
        elif st.session_state.get(error_key) and st.session_state[calculated_key]: pass 
        else: st.info(f"Select date range and click 'Fetch/Refresh {title} Data'.")
    
    with metric_placeholder.container():
        if st.session_state[calculated_key] and display_metrics_logic:
            display_metrics_logic(st.session_state.get(data_key), section_key_suffix) 

# --- Section 3: FRED Economic Data Viewer (Single Series) ---
def fetch_single_fred_logic(start_date_key, end_date_key, show_recession_key, data_key, error_key, _, series_id_map, default_series_name_key):
    series_name = st.session_state[default_series_name_key] 
    series_id = series_id_map.get(series_name)
    st.session_state.fred_series_id_calculated = series_id 
    st.session_state.fred_series_name_calculated = series_name 
    if not series_id: st.session_state[error_key] = "Invalid FRED series selection."; return
    with st.spinner(f"Fetching {series_name} ({series_id})..."):
        s_start, s_end, show_rec = st.session_state[start_date_key], st.session_state[end_date_key], st.session_state[show_recession_key]
        df = pd.DataFrame()
        s_data, err = data_fetchers.get_fred_data(config.fred, series_id, s_start, s_end)
        if s_data is not None: df[series_id] = s_data
        st.session_state.fred_series_info, st.session_state.fred_info_error = data_fetchers.get_fred_series_info(config.fred, series_id)
        if show_rec:
            rec_data, rec_err = data_fetchers.get_fred_data(config.fred, config.USREC_SERIES_ID, s_start, s_end)
            if rec_data is not None: df[config.USREC_SERIES_ID] = rec_data
            if rec_err: err = f"{err if err else ''} Recession Bands Error: {rec_err}"
        st.session_state[data_key] = df if not df.empty else None
        st.session_state[error_key] = err

def plot_single_fred(data_df, show_recession_value, _, __, ___, ____): # Added placeholder for yfinance_gold_col_name
    series_id = st.session_state.fred_series_id_calculated 
    series_data_col = data_df.get(series_id) if data_df is not None and series_id in data_df.columns else None
    recession_series = data_df.get(config.USREC_SERIES_ID) if data_df is not None and show_recession_value else None
    return plotters.create_fred_plot(series_data_col, series_id, st.session_state.fred_series_info, recession_series, show_recession_value)

def display_single_fred_metrics(data_df, _):
    if st.session_state.fred_series_info is not None and not st.session_state.fred_series_info.empty:
        info = st.session_state.fred_series_info
        st.caption(f"Title: {info.get('title', 'N/A')}. Last Updated: {info.get('last_updated', 'N/A')}. Units: {info.get('units_short', 'N/A')}. Frequency: {info.get('frequency_short', 'N/A')}.")
        notes = info.get('notes', '')
        if notes and isinstance(notes, str): st.expander("Series Notes").caption(notes)
    elif st.session_state.fred_info_error: st.caption(f"Metadata Error: {st.session_state.fred_info_error}")

if config.fred: fred_section_ui("fred_single", "FRED Economic Data Viewer", "Fetches and displays a single time series from FRED.", fetch_single_fred_logic, plot_single_fred, display_single_fred_metrics, config.fred_series_options, "fred_series_name_input", config.FRED_SERIES_EXAMPLES)
else: st.divider(); st.header("🏛️ FRED Economic Data Viewer"); st.error(config.fred_error_message)

# --- Section 4: Fed's Jaws Chart ---
st.divider(); st.header("🦅 Fed's Jaws: Key Policy Rates")
st.write(f"Visualizes key Federal Reserve interest rates over the last **{config.FED_JAWS_DURATION_DAYS} days**.")
plot_placeholder_jaws = st.empty()
if config.fred:
    st.checkbox("Show NBER Recession Bands", value=st.session_state.fed_jaws_show_recession, key="fed_jaws_show_recession")
    if st.button("Fetch/Refresh Fed's Jaws Data", key="jaws_button", type="primary"):
        st.session_state.fed_jaws_calculated = True; end_j = datetime.now(); start_j = end_j - timedelta(days=config.FED_JAWS_DURATION_DAYS)
        with st.spinner(f"Fetching Fed's Jaws data..."):
            st.session_state.fed_jaws_data, st.session_state.fed_jaws_error = data_fetchers.get_multiple_fred_data(config.fred, list(config.FED_JAWS_SERIES_IDS), start_j, end_j, include_recession_bands=st.session_state.fed_jaws_show_recession)
            if st.session_state.fed_jaws_error: plot_placeholder_jaws.warning(f"Jaws Data Warning: {st.session_state.fed_jaws_error}")
    with plot_placeholder_jaws.container():
        if st.session_state.fed_jaws_calculated and st.session_state.fed_jaws_data is not None:
            recession_series_jaws = st.session_state.fed_jaws_data.get(config.USREC_SERIES_ID) if st.session_state.fed_jaws_show_recession else None
            fig_jaws = plotters.create_fed_jaws_plot(st.session_state.fed_jaws_data, recession_series_jaws, st.session_state.fed_jaws_show_recession)
            st.plotly_chart(fig_jaws, use_container_width=True)
            if st.session_state.fed_jaws_data is not None and not st.session_state.fed_jaws_data.empty:
                st.download_button("Download Jaws Data (CSV)", utils.convert_df_to_csv(st.session_state.fed_jaws_data), "fed_jaws_data.csv", "text/csv", key="dl_jaws_csv")
        elif st.session_state.fed_jaws_error and st.session_state.fed_jaws_calculated: pass 
        else: st.info(f"Click 'Fetch/Refresh' for Fed's Jaws chart.")
else: st.divider(); st.header("🦅 Fed's Jaws: Key Policy Rates"); st.error(config.fred_error_message)

# --- Section 5: Fed Funds Rate vs Core PCE ---
def fetch_ffr_pce_logic(start_date_key, end_date_key, show_recession_key, data_key, error_key, *_):
    s_ids = [config.FFR_VS_PCE_SERIES_IDS["ffr"], config.FFR_VS_PCE_SERIES_IDS["core_pce_index"]]
    with st.spinner("Fetching FFR & Core PCE data..."):
        st.session_state[data_key], st.session_state[error_key] = data_fetchers.get_multiple_fred_data(config.fred, s_ids, st.session_state[start_date_key], st.session_state[end_date_key], include_recession_bands=st.session_state[show_recession_key])
def plot_ffr_pce(data_df, show_recession_value, _, __, ___, ____): # Added placeholder for yfinance_gold_col_name
    recession_series = data_df.get(config.USREC_SERIES_ID) if data_df is not None and show_recession_value else None
    return plotters.create_ffr_pce_comparison_plot(data_df, config.FFR_VS_PCE_SERIES_IDS["ffr"], config.FFR_VS_PCE_SERIES_IDS["core_pce_index"], config.FFR_PCE_THRESHOLD, recession_series, show_recession_value)
def display_ffr_pce_metrics(data_df, _): 
    if data_df is not None and config.FFR_VS_PCE_SERIES_IDS["ffr"] in data_df.columns and config.FFR_VS_PCE_SERIES_IDS["core_pce_index"] in data_df.columns:
        tdf = data_df.copy(); tdf.sort_index(inplace=True)
        tdf['PCE_YoY_Calc'] = tdf[config.FFR_VS_PCE_SERIES_IDS["core_pce_index"]].pct_change(periods=12) * 100
        tdf.dropna(subset=[config.FFR_VS_PCE_SERIES_IDS["ffr"], 'PCE_YoY_Calc'], inplace=True)
        if not tdf.empty:
            diff = tdf[config.FFR_VS_PCE_SERIES_IDS["ffr"]].iloc[-1] - tdf['PCE_YoY_Calc'].iloc[-1]
            st.metric(f"Latest Difference (FFR - Core PCE YoY Calc.)", f"{diff:.2f}%", delta_color=("inverse" if diff < config.FFR_PCE_THRESHOLD else "normal"))
            st.caption(f"Threshold: {config.FFR_PCE_THRESHOLD}%. Current diff is {'above' if diff > config.FFR_PCE_THRESHOLD else 'at or below'} threshold.")
        else: st.caption("N/A (No overlapping data for metric).")
    else: st.caption("N/A (Data missing for metric).")
if config.fred: fred_section_ui("ffr_pce", "💰 Fed Funds Rate vs. Core PCE Inflation", f"Visualizes FFR vs Core PCE YoY. Gap colored if FFR - PCE YoY > {config.FFR_PCE_THRESHOLD}%.", fetch_ffr_pce_logic, plot_ffr_pce, display_ffr_pce_metrics)
else: st.divider(); st.header("💰 Fed Funds Rate vs. Core PCE Inflation"); st.error(config.fred_error_message)

# --- Section 6: Gold vs. 10Y Real Yield ---
def fetch_gold_ry_logic(start_date_key, end_date_key, show_recession_key, data_key, error_key, section_key_suffix, *_): # Added section_key_suffix
    s_start, s_end, show_rec = st.session_state[start_date_key], st.session_state[end_date_key], st.session_state[show_recession_key]
    combined_data = pd.DataFrame()
    current_error = None

    # Fetch Gold from yfinance
    gold_ticker = config.GOLD_YFINANCE_TICKER
    yfinance_gold_col_name = f"{gold_ticker}_Close" # Define the column name we'll use
    st.session_state['yfinance_gold_col_name_plotter'] = yfinance_gold_col_name # Store for plotter

    try:
        print(f"Fetching yfinance Gold ({gold_ticker}) from {s_start} to {s_end}")
        gold_data_yf = yf.download(gold_ticker, start=s_start, end=s_end, progress=False)
        if not gold_data_yf.empty and 'Close' in gold_data_yf.columns:
            combined_data[yfinance_gold_col_name] = gold_data_yf['Close'] # Use the defined column name
        else:
            current_error = f"No Gold data from yfinance for {gold_ticker}. "
    except Exception as e:
        current_error = f"Error fetching Gold ({gold_ticker}) from yfinance: {e}. "
        print(f"yfinance gold fetch error: {e}")

    # Fetch Real Yield from FRED
    real_yield_id = config.GOLD_VS_REAL_YIELD_SERIES_IDS["real_yield_10y"]
    ry_data, ry_err = data_fetchers.get_fred_data(config.fred, real_yield_id, s_start, s_end)
    if ry_data is not None:
        combined_data[real_yield_id] = ry_data
    if ry_err:
        current_error = (current_error or "") + f"Real Yield Error: {ry_err}. "

    # Fetch Recession Data if requested
    if show_rec:
        rec_data, rec_err = data_fetchers.get_fred_data(config.fred, config.USREC_SERIES_ID, s_start, s_end)
        if rec_data is not None:
            combined_data[config.USREC_SERIES_ID] = rec_data
        if rec_err:
            current_error = (current_error or "") + f"Recession Data Error: {rec_err}. "
    
    # Combine and ffill
    if not combined_data.empty:
        # Ensure index is DatetimeIndex for proper alignment if sources differ slightly
        combined_data.index = pd.to_datetime(combined_data.index)
        # Reindex with a full date range and then ffill, then bfill
        # This ensures all series are on the same daily frequency for plotting if one is sparser
        if not combined_data.index.empty:
            full_date_range = pd.date_range(start=combined_data.index.min(), end=combined_data.index.max(), freq='B') # Business day frequency
            combined_data = combined_data.reindex(full_date_range)
            combined_data.ffill(inplace=True)
            combined_data.bfill(inplace=True) # Backfill for leading NaNs
            st.session_state[data_key] = combined_data.dropna(how='all') # Drop rows if all are NaN after ffill/bfill
        else:
            st.session_state[data_key] = None # No data to process
            current_error = (current_error or "") + "Combined data index was empty."
    else:
        st.session_state[data_key] = None
        current_error = (current_error or "") + "No data fetched for Gold vs Real Yield."

    st.session_state[error_key] = current_error.strip() if current_error else None


def plot_gold_ry(data_df, show_recession_value, _, __, ___, yfinance_gold_col_name_for_plotter): # Added yfinance_gold_col_name
    recession_series = data_df.get(config.USREC_SERIES_ID) if data_df is not None and show_recession_value else None
    # Use the yfinance_gold_col_name passed from session state via fred_section_ui
    actual_gold_col_name = yfinance_gold_col_name_for_plotter if yfinance_gold_col_name_for_plotter else config.GOLD_YFINANCE_TICKER + "_Close" # Fallback
    
    return plotters.create_gold_vs_real_yield_plot(
        data_df, 
        yfinance_gold_col_name=actual_gold_col_name, # Pass the correct gold column name
        fred_real_yield_col_name=config.GOLD_VS_REAL_YIELD_SERIES_IDS["real_yield_10y"], 
        recession_data_series=recession_series, 
        show_recession_bands=show_recession_value
    )

def display_gold_ry_metrics(data_df, _): 
    if data_df is not None:
        # Gold column name is now dynamic based on yfinance ticker + "_Close"
        gold_col_name = st.session_state.get('yfinance_gold_col_name_plotter', config.GOLD_YFINANCE_TICKER + "_Close")
        ry_id = config.GOLD_VS_REAL_YIELD_SERIES_IDS["real_yield_10y"]
        
        latest_g = data_df[gold_col_name].dropna().iloc[-1] if gold_col_name in data_df.columns and not data_df[gold_col_name].dropna().empty else "N/A"
        latest_ry = data_df[ry_id].dropna().iloc[-1] if ry_id in data_df.columns and not data_df[ry_id].dropna().empty else "N/A"
        c1, c2 = st.columns(2)
        c1.metric(f"Latest {config.GOLD_YFINANCE_TICKER} Price", f"${latest_g:,.2f}" if isinstance(latest_g, (float,int)) else latest_g)
        c2.metric("Latest 10Y Real Yield", f"{latest_ry:.2f}%" if isinstance(latest_ry, (float,int)) else latest_ry)
    else: st.caption("N/A (Data missing for metric).")

if config.fred: 
    fred_section_ui(
        "gold_ry", "🪙 Gold vs. 10Y Real Yield", 
        f"Plots Gold Price ({config.GOLD_YFINANCE_TICKER} from yfinance) against the 10-Year Treasury Inflation-Indexed Security yield (FRED).", 
        fetch_gold_ry_logic, plot_gold_ry, display_gold_ry_metrics
    )
else: st.divider(); st.header("🪙 Gold vs. 10Y Real Yield"); st.error(config.fred_error_message)


# --- Ticker Reference Table & Footer ---
st.divider()
with st.expander("Show Example Ticker Symbols (Yahoo Finance)"): st.dataframe(config.ticker_df, use_container_width=True, hide_index=True, column_config={"Asset Class": st.column_config.TextColumn("Asset Class"), "Description": st.column_config.TextColumn("Description"), "Yahoo Ticker": st.column_config.TextColumn("Yahoo Ticker")})
st.divider(); st.markdown(utils.generate_footer_html(config.YOUR_NAME, config.LINKEDIN_URL, config.LINKEDIN_SVG), unsafe_allow_html=True)
st.caption("Market data from Yahoo Finance (yfinance). Economic data from FRED® (fredapi). Data may be delayed.")

