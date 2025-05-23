# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta 
import traceback 
import yfinance as yf 
import config, data_fetchers, plotters, utils

st.set_page_config(page_title="Macro/Quantamental Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>📊 Macro/Quantamental Dashboard</h1>", unsafe_allow_html=True) 

# --- Tab Navigation (remains the same) ---
sections = {
    "Correlation Calculator": "correlation_calculator",
    "IV Skew Viewer": "iv_skew_viewer",
    "Index Valuation": "index_valuation_ratios",
    "FRED Single Series": "fred_single_viewer",
    "Fed's Jaws": "fed_jaws_chart", 
    "FFR vs Core PCE": "ffr_pce_chart",
    "Gold vs Real Yield": "gold_real_yield_chart",
    "Ticker Reference": "ticker_reference"
}
tab_links = []
for title, anchor_id in sections.items():
    tab_links.append(f"<a href='#{anchor_id}' style='margin: 0 10px 8px 10px; padding: 8px 12px; text-decoration: none; color: #4F8BF9; font-weight: 500; border-radius: 5px; background-color: #f0f2f6; display: inline-block;' onmouseover='this.style.backgroundColor=\"#e0e2e6\"' onmouseout='this.style.backgroundColor=\"#f0f2f6\"'>{title}</a>")
tabs_html = "<div style='text-align: center; margin-bottom: 25px; margin-top: 5px; line-height: 1.8;'>" + "".join(tab_links) + "</div>" 
st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<hr style='margin-bottom: 30px;'/>", unsafe_allow_html=True)


# --- Initialize Session State (and Global Controls) ---
utils.init_state('global_start_date', date.today() - timedelta(days=10*365)) # Default to last 10 years
utils.init_state('global_end_date', date.today())
utils.init_state('dashboard_calculated_once', False) # Flag to see if main button was ever pressed

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

# Index Valuation
utils.init_state('index_valuation_ticker_input', config.DEFAULT_INDEX_VALUATION_TICKER)
utils.init_state('index_valuation_data', None); utils.init_state('index_valuation_error', None)
utils.init_state('index_valuation_calculated_ticker', None)

# FRED Single
utils.init_state('fred_single_data', None); utils.init_state('fred_single_error', None)
utils.init_state('fred_series_info', None); utils.init_state('fred_info_error', None); 
utils.init_state('fred_series_id_calculated', None); utils.init_state('fred_series_name_calculated', None) 
utils.init_state('fred_series_name_input', config.DEFAULT_FRED_SERIES_NAME) 
utils.init_state('fred_single_show_recession', True)        

# Fed Jaws
utils.init_state('fed_jaws_data', None); utils.init_state('fed_jaws_error', None)
utils.init_state('fed_jaws_show_recession', True) 

# FFR vs PCE
utils.init_state('ffr_pce_data', None); utils.init_state('ffr_pce_error', None)
utils.init_state('current_ffr_pce_diff', None)
utils.init_state('ffr_pce_show_recession', True)        

# Gold vs Real Yield
utils.init_state('gold_ry_data', None); utils.init_state('gold_ry_error', None)
utils.init_state('yfinance_gold_col_name_plotter', None) 
utils.init_state('gold_ry_show_recession', True)   

# --- Global Controls ---
st.sidebar.header("Global Controls")
st.session_state.global_start_date = st.sidebar.date_input(
    "Global Start Date", 
    value=st.session_state.global_start_date, 
    key="global_start_date_widget"
)
st.session_state.global_end_date = st.sidebar.date_input(
    "Global End Date", 
    value=st.session_state.global_end_date, 
    min_value=st.session_state.global_start_date, # Ensure end date is after start date
    key="global_end_date_widget"
)

if st.sidebar.button("🚀 Calculate & Refresh All Dashboard Data", type="primary", use_container_width=True):
    st.session_state.dashboard_calculated_once = True
    # --- Trigger all data fetching ---
    # Correlation
    t1_c, t2_c, win_c = st.session_state.ticker1_input_corr, st.session_state.ticker2_input_corr, st.session_state.rolling_window_corr
    st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr, st.session_state.corr_window_calculated = t1_c, t2_c, win_c
    if t1_c and t2_c and t1_c != t2_c:
        with st.spinner(f"Calculating Correlation ({t1_c} vs {t2_c}, {win_c}d)..."):
            st.session_state.rolling_corr_data, st.session_state.rolling_corr_error = data_fetchers.calculate_rolling_correlation(t1_c, t2_c, win_c)
    
    # IV Skew
    ticker_s, expiry_s = st.session_state.ticker_input_skew, st.session_state.expiry_input_skew
    st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew = ticker_s, expiry_s
    if ticker_s and expiry_s:
        with st.spinner(f"Fetching IV Skew ({ticker_s}, {expiry_s})..."):
            calls, puts, price, err = data_fetchers.get_option_chain_data(ticker_s, expiry_s)
            st.session_state.calls_data_skew, st.session_state.puts_data_skew, st.session_state.price_skew, st.session_state.skew_error = calls, puts, price, err

    # Index Valuation
    ticker_iv = st.session_state.index_valuation_ticker_input
    st.session_state.index_valuation_calculated_ticker = ticker_iv
    if ticker_iv:
        with st.spinner(f"Fetching Index Valuation ({ticker_iv})..."):
            st.session_state.index_valuation_data, st.session_state.index_valuation_error = data_fetchers.get_index_valuation_ratios(ticker_iv)

    # FRED Single Series
    series_name_fs = st.session_state.fred_series_name_input
    series_id_fs = config.FRED_SERIES_EXAMPLES.get(series_name_fs)
    st.session_state.fred_series_id_calculated, st.session_state.fred_series_name_calculated = series_id_fs, series_name_fs
    if series_id_fs and config.fred:
        with st.spinner(f"Fetching FRED: {series_name_fs}..."):
            df_fs = pd.DataFrame()
            s_data_fs, err_fs = data_fetchers.get_fred_data(config.fred, series_id_fs, st.session_state.global_start_date, st.session_state.global_end_date)
            if s_data_fs is not None: df_fs[series_id_fs] = s_data_fs
            st.session_state.fred_series_info, st.session_state.fred_info_error = data_fetchers.get_fred_series_info(config.fred, series_id_fs)
            if st.session_state.fred_single_show_recession:
                rec_data_fs, rec_err_fs = data_fetchers.get_fred_data(config.fred, config.USREC_SERIES_ID, st.session_state.global_start_date, st.session_state.global_end_date)
                if rec_data_fs is not None: df_fs[config.USREC_SERIES_ID] = rec_data_fs
                if rec_err_fs: err_fs = f"{err_fs if err_fs else ''} Recession Bands Error: {rec_err_fs}"
            st.session_state.fred_single_data = df_fs if not df_fs.empty else None
            st.session_state.fred_single_error = err_fs
            
    # Fed Jaws (uses its own duration, but refresh triggered globally)
    if config.fred:
        with st.spinner("Fetching Fed's Jaws data..."):
            end_j = datetime.combine(st.session_state.global_end_date, datetime.min.time()) # Use global end date
            start_j = end_j - timedelta(days=config.FED_JAWS_DURATION_DAYS)
            st.session_state.fed_jaws_data, st.session_state.fed_jaws_error = data_fetchers.get_multiple_fred_data(
                config.fred, list(config.FED_JAWS_SERIES_IDS), start_j, end_j, 
                include_recession_bands=st.session_state.fed_jaws_show_recession
            )

    # FFR vs PCE
    if config.fred:
        with st.spinner("Fetching FFR vs Core PCE data..."):
            s_ids_ffr = [config.FFR_VS_PCE_SERIES_IDS["ffr"], config.FFR_VS_PCE_SERIES_IDS["core_pce_index"]]
            st.session_state.ffr_pce_data, st.session_state.ffr_pce_error = data_fetchers.get_multiple_fred_data(
                config.fred, s_ids_ffr, st.session_state.global_start_date, st.session_state.global_end_date, 
                include_recession_bands=st.session_state.ffr_pce_show_recession
            )
            # Calculate current difference
            if st.session_state.ffr_pce_data is not None and not st.session_state.ffr_pce_data.empty:
                tdf_ffr = st.session_state.ffr_pce_data.copy(); tdf_ffr.sort_index(inplace=True)
                ffr_col, pce_idx_col = config.FFR_VS_PCE_SERIES_IDS["ffr"], config.FFR_VS_PCE_SERIES_IDS["core_pce_index"]
                if ffr_col in tdf_ffr.columns and pce_idx_col in tdf_ffr.columns:
                    tdf_ffr['PCE_YoY_Calc'] = tdf_ffr[pce_idx_col].pct_change(periods=12) * 100
                    tdf_ffr.dropna(subset=[ffr_col, 'PCE_YoY_Calc'], inplace=True)
                    if not tdf_ffr.empty: st.session_state.current_ffr_pce_diff = tdf_ffr[ffr_col].iloc[-1] - tdf_ffr['PCE_YoY_Calc'].iloc[-1]
                    else: st.session_state.current_ffr_pce_diff = "N/A (No overlap)"
                else: st.session_state.current_ffr_pce_diff = "N/A (Cols missing)"
            else: st.session_state.current_ffr_pce_diff = "N/A (Data empty)"


    # Gold vs Real Yield
    if config.fred:
        with st.spinner("Fetching Gold vs Real Yield data..."):
            s_start_gry, s_end_gry, show_rec_gry = st.session_state.global_start_date, st.session_state.global_end_date, st.session_state.gold_ry_show_recession
            combined_data_gry = pd.DataFrame(); current_error_gry = None
            gold_ticker_gry = config.GOLD_YFINANCE_TICKER; yf_gold_col_gry = f"{gold_ticker_gry}_Close"
            st.session_state['yfinance_gold_col_name_plotter'] = yf_gold_col_gry
            try:
                gold_data_yf_gry = yf.download(gold_ticker_gry, start=s_start_gry, end=s_end_gry, progress=False)
                if not gold_data_yf_gry.empty and 'Close' in gold_data_yf_gry.columns: combined_data_gry[yf_gold_col_gry] = gold_data_yf_gry['Close']
                else: current_error_gry = f"No Gold data from yfinance for {gold_ticker_gry}. "
            except Exception as e: current_error_gry = f"Error fetching Gold ({gold_ticker_gry}) from yfinance: {e}. "
            ry_id_gry = config.GOLD_VS_REAL_YIELD_SERIES_IDS["real_yield_10y"]
            ry_data_gry, ry_err_gry = data_fetchers.get_fred_data(config.fred, ry_id_gry, s_start_gry, s_end_gry)
            if ry_data_gry is not None: combined_data_gry[ry_id_gry] = ry_data_gry
            if ry_err_gry: current_error_gry = (current_error_gry or "") + f"RY Error: {ry_err_gry}. "
            if show_rec_gry:
                rec_data_gry, rec_err_gry = data_fetchers.get_fred_data(config.fred, config.USREC_SERIES_ID, s_start_gry, s_end_gry)
                if rec_data_gry is not None: combined_data_gry[config.USREC_SERIES_ID] = rec_data_gry
                if rec_err_gry: current_error_gry = (current_error_gry or "") + f"Recession Error: {rec_err_gry}. "
            if not combined_data_gry.empty and not combined_data_gry.index.empty:
                full_range_gry = pd.date_range(start=combined_data_gry.index.min(), end=combined_data_gry.index.max(), freq='B')
                combined_data_gry = combined_data_gry.reindex(full_range_gry).ffill().bfill()
                st.session_state.gold_ry_data = combined_data_gry.dropna(how='all')
            else: st.session_state.gold_ry_data = None; current_error_gry = (current_error_gry or "") + "No combined data for Gold/RY."
            st.session_state.gold_ry_error = current_error_gry.strip() if current_error_gry else None
    st.success("Dashboard data refreshed!")


# --- Section 1: Rolling Correlation ---
# ... (Display logic remains similar, but no individual fetch button) ...
correlation_anchor_id = sections["Correlation Calculator"]
st.markdown(f"<a name='{correlation_anchor_id}'></a>", unsafe_allow_html=True) 
st.header("📊 Rolling Correlation Calculator")
st.write("Uses daily returns. Adjust tickers and window, then click 'Refresh Dashboard Data' in the sidebar.")
st.slider("Select Rolling Window (Days):", min_value=config.MIN_ROLLING_WINDOW_CORR, max_value=config.MAX_ROLLING_WINDOW_CORR, value=st.session_state.rolling_window_corr, step=1, key="rolling_window_corr")
col1_corr, col2_corr = st.columns(2)
with col1_corr: st.session_state.ticker1_input_corr = st.text_input("Ticker 1:", value=st.session_state.ticker1_input_corr, key="ticker1_corr_widget").strip().upper()
with col2_corr: st.session_state.ticker2_input_corr = st.text_input("Ticker 2:", value=st.session_state.ticker2_input_corr, key="ticker2_corr_widget").strip().upper()
plot_placeholder_corr = st.empty()
with plot_placeholder_corr.container():
    if st.session_state.dashboard_calculated_once:
        if st.session_state.rolling_corr_error: st.error(f"Correlation Error: {st.session_state.rolling_corr_error}")
        elif st.session_state.rolling_corr_data is not None:
            fig_corr = plotters.create_corr_plot(st.session_state.rolling_corr_data, st.session_state.ticker1_calculated_corr, st.session_state.ticker2_calculated_corr, st.session_state.corr_window_calculated)
            st.plotly_chart(fig_corr, use_container_width=True)
            df_to_download_corr = st.session_state.rolling_corr_data
            if isinstance(df_to_download_corr, pd.Series): df_to_download_corr = df_to_download_corr.to_frame()
            if df_to_download_corr is not None and not df_to_download_corr.empty:
                csv_corr = utils.convert_df_to_csv(df_to_download_corr) 
                st.download_button(label="Download Correlation Data (CSV)", data=csv_corr, file_name=f"corr_{st.session_state.ticker1_calculated_corr}_{st.session_state.ticker2_calculated_corr}_{st.session_state.corr_window_calculated}d.csv", mime='text/csv', key="download_corr_csv_global")
        elif st.session_state.ticker1_calculated_corr: # Attempted but no data and no specific error
             st.info("No correlation data to display for the selected tickers/window after refresh.")
    else: st.info("Configure tickers/window and click 'Refresh Dashboard Data' in the sidebar.")


# --- Section 2: Implied Volatility Skew ---
# ... (Display logic similar, no individual fetch button) ...
iv_skew_anchor_id = sections["IV Skew Viewer"]
st.markdown(f"<a name='{iv_skew_anchor_id}'></a>", unsafe_allow_html=True) 
st.divider(); st.header("📉 Implied Volatility Skew Viewer")
st.write("Visualizes IV smile/skew. Zero IVs are interpolated. Configure ticker/expiry, then click 'Refresh Dashboard Data' in sidebar.")
st.session_state.ticker_input_skew = st.text_input("Equity/ETF Ticker:", value=st.session_state.ticker_input_skew, key="ticker_skew_widget_global").strip().upper()
expirations_g, err_exp_g = data_fetchers.get_expiration_dates(st.session_state.ticker_input_skew) # Use _g for global context
sel_exp_g = None
if err_exp_g: st.warning(err_exp_g)
if expirations_g:
    try: 
        today_g = pd.Timestamp.now().normalize(); future_dates_g = pd.to_datetime(expirations_g)[pd.to_datetime(expirations_g) >= today_g]
        def_idx_g = np.abs((future_dates_g - (today_g + pd.Timedelta(days=90))).total_seconds()).argmin() if not future_dates_g.empty else (len(expirations_g) -1 if expirations_g else 0)
        curr_idx_g = expirations_g.index(st.session_state.expiry_calculated_skew) if st.session_state.ticker_input_skew == st.session_state.ticker_calculated_skew and st.session_state.expiry_calculated_skew in expirations_g else def_idx_g
        # Store selected expiry directly in session state for the global refresh to pick up
        st.session_state.expiry_input_skew = st.selectbox("Select Expiration Date:", expirations_g, index=curr_idx_g, key="expiry_select_widget_skew_global")
        sel_exp_g = st.session_state.expiry_input_skew 
    except Exception as e: st.session_state.expiry_input_skew = st.selectbox("Select Expiration Date (fallback):", expirations_g, key="expiry_select_fb_skew_global") if expirations_g else None; sel_exp_g = st.session_state.expiry_input_skew
elif st.session_state.ticker_input_skew and not err_exp_g: st.info(f"No options for '{st.session_state.ticker_input_skew}'.")
plot_placeholder_skew = st.empty()
with plot_placeholder_skew.container():
    if st.session_state.dashboard_calculated_once:
        if st.session_state.skew_error: st.error(f"IV Skew Error: {st.session_state.skew_error}")
        elif st.session_state.ticker_calculated_skew and (st.session_state.calls_data_skew is not None or st.session_state.puts_data_skew is not None):
            fig_skew = plotters.create_iv_skew_plot(st.session_state.calls_data_skew, st.session_state.puts_data_skew, st.session_state.ticker_calculated_skew, st.session_state.expiry_calculated_skew, st.session_state.price_skew)
            st.plotly_chart(fig_skew, use_container_width=True)
            # Download buttons
            if st.session_state.calls_data_skew is not None and not st.session_state.calls_data_skew.empty:
                st.download_button(label="Download Calls Data (CSV)", data=utils.convert_df_to_csv(st.session_state.calls_data_skew), file_name=f"calls_{st.session_state.ticker_calculated_skew}_{st.session_state.expiry_calculated_skew}.csv", mime='text/csv', key="download_calls_csv_global")
            if st.session_state.puts_data_skew is not None and not st.session_state.puts_data_skew.empty:
                st.download_button(label="Download Puts Data (CSV)", data=utils.convert_df_to_csv(st.session_state.puts_data_skew), file_name=f"puts_{st.session_state.ticker_calculated_skew}_{st.session_state.expiry_calculated_skew}.csv", mime='text/csv', key="download_puts_csv_global")
        elif st.session_state.ticker_calculated_skew: # Attempted but no data
             st.info("No IV skew data to display for the selected ticker/expiry after refresh.")
    else: st.info("Configure ticker/expiry and click 'Refresh Dashboard Data' in the sidebar.")


# --- Section: Index Valuation Ratios ---
# ... (Display logic similar, no individual fetch button) ...
index_val_anchor_id = sections["Index Valuation"]
st.markdown(f"<a name='{index_val_anchor_id}'></a>", unsafe_allow_html=True)
st.divider(); st.header("밸류에이션 Index Valuation Ratios") 
st.write("Displays current valuation ratios for major stock market indices. Configure ticker, then click 'Refresh Dashboard Data' in sidebar.")
st.session_state.index_valuation_ticker_input = st.text_input("Enter Index Ticker (e.g., ^GSPC, ^IXIC):", value=st.session_state.index_valuation_ticker_input, key="index_valuation_ticker_widget_global").strip().upper()
valuation_placeholder = st.empty()
with valuation_placeholder.container():
    if st.session_state.dashboard_calculated_once:
        if st.session_state.index_valuation_error: st.error(f"Valuation Data Error: {st.session_state.index_valuation_error}")
        elif st.session_state.index_valuation_calculated_ticker and st.session_state.index_valuation_data:
            data_val = st.session_state.index_valuation_data
            st.subheader(f"Current Valuation for: {data_val.get('Name', st.session_state.index_valuation_calculated_ticker)}")
            col1_val, col2_val = st.columns(2)
            with col1_val:
                st.metric(label="Trailing P/E", value=f"{data_val.get('Trailing P/E'):.2f}" if isinstance(data_val.get('Trailing P/E'), (float, int)) else "N/A")
                st.metric(label="Price/Book", value=f"{data_val.get('Price/Book'):.2f}" if isinstance(data_val.get('Price/Book'), (float, int)) else "N/A")
            with col2_val:
                st.metric(label="Forward P/E", value=f"{data_val.get('Forward P/E'):.2f}" if isinstance(data_val.get('Forward P/E'), (float, int)) else "N/A")
                st.metric(label="Dividend Yield", value=data_val.get('Dividend Yield', "N/A"))
            if data_val:
                st.download_button(label=f"Download Valuation Data (CSV)", data=utils.convert_df_to_csv(pd.DataFrame([data_val])), file_name=f"valuation_{st.session_state.index_valuation_calculated_ticker}.csv", mime='text/csv', key=f"dl_val_{st.session_state.index_valuation_calculated_ticker}_csv")
        elif st.session_state.index_valuation_calculated_ticker:
            st.info("No valuation data to display after refresh.")
    else: st.info("Configure index ticker and click 'Refresh Dashboard Data' in the sidebar.")


# --- FRED Sections (Single, FFR vs PCE, Gold vs RY, Fed Jaws) ---
# These will now use global date pickers and refresh via the global button.
# The structure will be: Anchor, Header, Controls (like recession toggle or series select), Display Area

# --- Section: FRED Single Series ---
fred_single_anchor_id = sections["FRED Single Series"]
st.markdown(f"<a name='{fred_single_anchor_id}'></a>", unsafe_allow_html=True)
st.divider(); st.header("🏛️ FRED Economic Data Viewer")
st.write("Fetches and displays a single time series from FRED. Uses global date range.")
if config.fred:
    try: idx_fs = config.fred_series_options.index(st.session_state.fred_series_name_input)
    except ValueError: idx_fs = 0; st.session_state.fred_series_name_input = config.fred_series_options[0]
    st.session_state.fred_series_name_input = st.selectbox("Select FRED Series:", options=config.fred_series_options, index=idx_fs, key="fred_series_select_widget_global")
    st.checkbox("Show NBER Recession Bands", value=st.session_state.fred_single_show_recession, key="fred_single_show_recession")
    
    plot_placeholder_fs = st.empty(); caption_placeholder_fs = st.empty()
    with plot_placeholder_fs.container():
        if st.session_state.dashboard_calculated_once:
            if st.session_state.fred_single_error: st.error(f"FRED Single Series Error: {st.session_state.fred_single_error}")
            elif st.session_state.fred_single_data is not None and st.session_state.fred_series_id_calculated:
                recession_data_fs = st.session_state.fred_single_data.get(config.USREC_SERIES_ID) if st.session_state.fred_single_show_recession else None
                fig_fs = plotters.create_fred_plot(st.session_state.fred_single_data.get(st.session_state.fred_series_id_calculated), st.session_state.fred_series_id_calculated, st.session_state.fred_series_info, recession_data_fs, st.session_state.fred_single_show_recession)
                st.plotly_chart(fig_fs, use_container_width=True)
                if st.session_state.fred_single_data is not None and not st.session_state.fred_single_data.empty:
                    st.download_button("Download FRED Series Data (CSV)", utils.convert_df_to_csv(st.session_state.fred_single_data), f"fred_{st.session_state.fred_series_id_calculated}.csv", "text/csv", key="dl_fred_single_csv")
            elif st.session_state.fred_series_id_calculated: st.info("No data for selected FRED series in the given range.")
    with caption_placeholder_fs.container():
        if st.session_state.dashboard_calculated_once and st.session_state.fred_series_info: display_single_fred_metrics(None, None) # Metrics are from session_state.fred_series_info
else: st.markdown(f"<a name='{fred_single_anchor_id}'></a>", unsafe_allow_html=True); st.divider(); st.header("🏛️ FRED Economic Data Viewer"); st.error(config.fred_error_message)


# --- Section: Fed's Jaws Chart ---
# ... (Similar display logic, uses its own date calculation but triggered by global refresh) ...
key_feds_jaws = "Fed's Jaws"; anchor_id_feds_jaws = sections[key_feds_jaws]
st.markdown(f"<a name='{anchor_id_feds_jaws}'></a>", unsafe_allow_html=True) 
st.divider(); st.header("🦅 Fed's Jaws: Key Policy Rates") 
st.write(f"Visualizes key Federal Reserve interest rates over the last **{config.FED_JAWS_DURATION_DAYS} days** (ending on global end date).")
plot_placeholder_jaws = st.empty()
if config.fred:
    st.checkbox("Show NBER Recession Bands", value=st.session_state.fed_jaws_show_recession, key="fed_jaws_show_recession_global")
    with plot_placeholder_jaws.container():
        if st.session_state.dashboard_calculated_once:
            if st.session_state.fed_jaws_error: st.warning(f"Jaws Data Warning: {st.session_state.fed_jaws_error}")
            elif st.session_state.fed_jaws_data is not None:
                recession_series_jaws = st.session_state.fed_jaws_data.get(config.USREC_SERIES_ID) if st.session_state.fed_jaws_show_recession else None
                fig_jaws = plotters.create_fed_jaws_plot(st.session_state.fed_jaws_data, recession_series_jaws, st.session_state.fed_jaws_show_recession)
                st.plotly_chart(fig_jaws, use_container_width=True)
                if st.session_state.fed_jaws_data is not None and not st.session_state.fed_jaws_data.empty:
                    st.download_button("Download Jaws Data (CSV)", utils.convert_df_to_csv(st.session_state.fed_jaws_data), "fed_jaws_data.csv", "text/csv", key="dl_jaws_csv_global")
            else: st.info("No Fed Jaws data to display after refresh.")
    # No initial info message here as it's part of global refresh
else: st.markdown(f"<a name='{anchor_id_feds_jaws}'></a>", unsafe_allow_html=True); st.divider(); st.header("🦅 Fed's Jaws: Key Policy Rates"); st.error(config.fred_error_message)


# --- Section: FFR vs Core PCE ---
ffr_pce_anchor_id = sections["FFR vs Core PCE"]
st.markdown(f"<a name='{ffr_pce_anchor_id}'></a>", unsafe_allow_html=True)
st.divider(); st.header("💰 Fed Funds Rate vs. Core PCE Inflation")
st.write(f"Visualizes FFR vs Core PCE YoY. Uses global date range. Gap colored if FFR - PCE YoY > {config.FFR_PCE_THRESHOLD}%.")
plot_placeholder_ffr_pce = st.empty(); metric_placeholder_ffr_pce = st.empty()
if config.fred:
    st.checkbox("Show NBER Recession Bands", value=st.session_state.ffr_pce_show_recession, key="ffr_pce_show_recession_global")
    with plot_placeholder_ffr_pce.container():
        if st.session_state.dashboard_calculated_once:
            if st.session_state.ffr_pce_error: st.error(f"FFR vs PCE Error: {st.session_state.ffr_pce_error}")
            elif st.session_state.ffr_pce_data is not None:
                recession_series_ffr = st.session_state.ffr_pce_data.get(config.USREC_SERIES_ID) if st.session_state.ffr_pce_show_recession else None
                fig_ffr = plotters.create_ffr_pce_comparison_plot(st.session_state.ffr_pce_data, config.FFR_VS_PCE_SERIES_IDS["ffr"], config.FFR_VS_PCE_SERIES_IDS["core_pce_index"], config.FFR_PCE_THRESHOLD, recession_series_ffr, st.session_state.ffr_pce_show_recession)
                st.plotly_chart(fig_ffr, use_container_width=True)
                if st.session_state.ffr_pce_data is not None and not st.session_state.ffr_pce_data.empty:
                    st.download_button("Download FFR vs PCE Data (CSV)", utils.convert_df_to_csv(st.session_state.ffr_pce_data), "ffr_pce_data.csv", "text/csv", key="dl_ffr_pce_csv")
            else: st.info("No FFR vs PCE data to display after refresh.")
    with metric_placeholder_ffr_pce.container():
        if st.session_state.dashboard_calculated_once and st.session_state.current_ffr_pce_diff is not None:
            display_ffr_pce_metrics(st.session_state.ffr_pce_data, None) # Call the metric display function
else: st.markdown(f"<a name='{ffr_pce_anchor_id}'></a>", unsafe_allow_html=True); st.divider(); st.header("💰 Fed Funds Rate vs. Core PCE Inflation"); st.error(config.fred_error_message)


# --- Section: Gold vs. 10Y Real Yield ---
gold_ry_anchor_id = sections["Gold vs Real Yield"]
st.markdown(f"<a name='{gold_ry_anchor_id}'></a>", unsafe_allow_html=True)
st.divider(); st.header("🪙 Gold vs. 10Y Real Yield")
st.write(f"Plots Gold Price ({config.GOLD_YFINANCE_TICKER}) vs 10Y Real Yield. Uses global date range.")
plot_placeholder_gold_ry = st.empty(); metric_placeholder_gold_ry = st.empty()
if config.fred: # Real yield is from FRED, so check this
    st.checkbox("Show NBER Recession Bands", value=st.session_state.gold_ry_show_recession, key="gold_ry_show_recession_global")
    with plot_placeholder_gold_ry.container():
        if st.session_state.dashboard_calculated_once:
            if st.session_state.gold_ry_error: st.error(f"Gold/RY Error: {st.session_state.gold_ry_error}")
            elif st.session_state.gold_ry_data is not None:
                recession_series_gry = st.session_state.gold_ry_data.get(config.USREC_SERIES_ID) if st.session_state.gold_ry_show_recession else None
                fig_gry = plotters.create_gold_vs_real_yield_plot(st.session_state.gold_ry_data, st.session_state.yfinance_gold_col_name_plotter, config.GOLD_VS_REAL_YIELD_SERIES_IDS["real_yield_10y"], recession_series_gry, st.session_state.gold_ry_show_recession)
                st.plotly_chart(fig_gry, use_container_width=True)
                if st.session_state.gold_ry_data is not None and not st.session_state.gold_ry_data.empty:
                    st.download_button("Download Gold/RY Data (CSV)", utils.convert_df_to_csv(st.session_state.gold_ry_data), "gold_ry_data.csv", "text/csv", key="dl_gold_ry_csv")
            else: st.info("No Gold vs Real Yield data to display after refresh.")
    with metric_placeholder_gold_ry.container():
        if st.session_state.dashboard_calculated_once and st.session_state.gold_ry_data is not None:
             display_gold_ry_metrics(st.session_state.gold_ry_data, None) # Call the metric display function
else: st.markdown(f"<a name='{gold_ry_anchor_id}'></a>", unsafe_allow_html=True); st.divider(); st.header("🪙 Gold vs. 10Y Real Yield"); st.error(config.fred_error_message)


# --- Initial Message if not calculated ---
if not st.session_state.dashboard_calculated_once:
    st.info("ℹ️ Welcome! Please adjust global date range and other inputs as needed, then click 'Calculate & Refresh All Dashboard Data' in the sidebar to load all charts and data.")


# --- Ticker Reference Table & Footer ---
# ... (remains the same) ...
ticker_ref_anchor_id = sections["Ticker Reference"]
st.markdown(f"<a name='{ticker_ref_anchor_id}'></a>", unsafe_allow_html=True) 
st.divider()
with st.expander("Show Example Ticker Symbols (Yahoo Finance)"): 
    st.dataframe(config.ticker_df, use_container_width=True, hide_index=True, column_config={"Asset Class": st.column_config.TextColumn("Asset Class"), "Description": st.column_config.TextColumn("Description"), "Yahoo Ticker": st.column_config.TextColumn("Yahoo Ticker")})
st.divider(); st.markdown(utils.generate_footer_html(config.YOUR_NAME, config.LINKEDIN_URL, config.LINKEDIN_SVG), unsafe_allow_html=True)
st.caption("Market data from Yahoo Finance (yfinance). Economic data from FRED® (fredapi). Data may be delayed.")

