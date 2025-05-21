# plotters.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
import config 

def add_recession_bands_to_fig(fig, recession_data_series): 
    """Adds recession bands to a Plotly figure if recession_data_series is provided and valid."""
    if recession_data_series is None or recession_data_series.empty or not isinstance(recession_data_series, pd.Series):
        # print("Debug: add_recession_bands_to_fig received invalid or empty recession_data_series.")
        return

    recession_data_series.index = pd.to_datetime(recession_data_series.index)
    in_recession_periods = recession_data_series[recession_data_series == 1]
    
    if in_recession_periods.empty:
        return

    start_date = None
    processed_end_dates = set() 

    for date_index, _ in in_recession_periods.items():
        if start_date is None: start_date = date_index
        current_loc = recession_data_series.index.get_loc(date_index)
        is_last_point = (current_loc == len(recession_data_series) - 1)
        next_is_not_recession = False
        if not is_last_point:
            next_date_val = recession_data_series.iloc[current_loc + 1]
            if pd.isna(next_date_val) or next_date_val == 0: next_is_not_recession = True
        if is_last_point or next_is_not_recession:
            end_date = date_index 
            if end_date not in processed_end_dates:
                actual_end_date = end_date + pd.DateOffset(days=1) if end_date == start_date else end_date
                fig.add_vrect(x0=start_date, x1=actual_end_date, fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0, name="NBER Recession" if not any(trace.name == "NBER Recession" for trace in fig.data) else None, showlegend=not any(trace.name == "NBER Recession" for trace in fig.data))
                processed_end_dates.add(end_date)
            start_date = None 
    if start_date is not None and start_date not in processed_end_dates and not in_recession_periods.empty: 
        end_date = in_recession_periods.index[-1]
        if end_date not in processed_end_dates:
            actual_end_date = end_date + pd.DateOffset(days=1) if end_date == start_date else end_date
            fig.add_vrect(x0=start_date, x1=actual_end_date, fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0, name="NBER Recession" if not any(trace.name == "NBER Recession" for trace in fig.data) else None, showlegend=not any(trace.name == "NBER Recession" for trace in fig.data))

def create_corr_plot(rolling_corr_data, ticker1, ticker2, window, years=config.YEARS_OF_DATA_CORR):
    fig = go.Figure(); plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"
    if rolling_corr_data is not None and not rolling_corr_data.empty:
        pos_corr = rolling_corr_data.copy(); pos_corr[pos_corr < 0] = None 
        neg_corr = rolling_corr_data.copy(); neg_corr[neg_corr >= 0] = None 
        fig.add_trace(go.Scatter(x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation', line=dict(color='red')))
    elif rolling_corr_data is not None and rolling_corr_data.empty: 
        plot_title += " (No data for window)" 
        fig.add_annotation(text=f"No correlation data for {ticker1}/{ticker2} ({window}-day window).", showarrow=False, align='center')
    else: 
        plot_title += " (Error)" 
        fig.add_annotation(text=f"Could not load correlation data for {ticker1}/{ticker2}.", showarrow=False, align='center')
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def interpolate_iv(df_options):
    if df_options is None or df_options.empty or 'strike' not in df_options.columns or 'impliedVolatility' not in df_options.columns: return df_options
    df_interpolated = df_options.copy()
    df_interpolated['strike'] = pd.to_numeric(df_interpolated['strike'], errors='coerce')
    df_interpolated['impliedVolatility'] = pd.to_numeric(df_interpolated['impliedVolatility'], errors='coerce')
    df_interpolated.dropna(subset=['strike'], inplace=True)
    if df_interpolated.empty: return df_interpolated
    df_interpolated.sort_values(by='strike', inplace=True)
    df_interpolated['iv_for_interpolation'] = df_interpolated['impliedVolatility'].copy()
    df_interpolated.loc[df_interpolated['iv_for_interpolation'] == 0, 'iv_for_interpolation'] = np.nan
    df_interpolated['iv_interpolated_values'] = df_interpolated['iv_for_interpolation'].interpolate(method='linear', limit_direction='both', limit_area='inside')
    df_interpolated['impliedVolatility'] = np.where((df_options['impliedVolatility'] == 0) & (df_interpolated['iv_interpolated_values'].notna()), df_interpolated['iv_interpolated_values'], df_interpolated['impliedVolatility'])
    df_interpolated.drop(columns=['iv_for_interpolation', 'iv_interpolated_values'], inplace=True, errors='ignore')
    df_interpolated.dropna(subset=['strike', 'impliedVolatility'], inplace=True)
    return df_interpolated

def create_iv_skew_plot(calls_df, puts_df, ticker, expiry_date, current_price):
    fig = go.Figure(); plot_title = f"{ticker} Implied Volatility Skew (Expiry: {expiry_date})"; data_plotted = False 
    calls_df_processed = interpolate_iv(calls_df); puts_df_processed = interpolate_iv(puts_df)
    if calls_df_processed is not None and not calls_df_processed.empty:
        fig.add_trace(go.Scatter(x=calls_df_processed['strike'],y=calls_df_processed['impliedVolatility'] * 100, mode='markers+lines', name='Calls IV (%)', marker=dict(color='blue'), line=dict(color='blue'))); data_plotted = True
    if puts_df_processed is not None and not puts_df_processed.empty:
        fig.add_trace(go.Scatter(x=puts_df_processed['strike'], y=puts_df_processed['impliedVolatility'] * 100, mode='markers+lines', name='Puts IV (%)', marker=dict(color='orange'), line=dict(color='orange'))); data_plotted = True
    fig.update_layout(title=plot_title, xaxis_title='Strike Price', yaxis_title='Implied Volatility (%)', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if current_price is not None and isinstance(current_price, (int, float, np.number)): 
        fig.add_vline(x=current_price, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Current Price: {current_price:.2f}", annotation_position="top right")
    if not data_plotted:
        fig.add_annotation(text=f"No valid options IV data for {ticker} expiring {expiry_date} after processing.", showarrow=False, align='center'); fig.update_layout(yaxis_range=[0, 100]) 
    return fig

def create_fred_plot(series_data, series_id, series_info, recession_data_series=None, show_recession_bands=False): 
    fig = go.Figure(); plot_title = f"FRED Series: {series_id}"; y_axis_label = "Value" 
    if series_info is not None and not series_info.empty:
        plot_title = series_info.get('title', plot_title); units = series_info.get('units_short', 'Value')
        freq = series_info.get('frequency_short', ''); adj = series_info.get('seasonal_adjustment_short', 'NSA') 
        y_axis_label = f"{units} ({freq}, {adj})" if freq else f"{units} ({adj})"
    if series_data is not None and not series_data.empty:
        fig.add_trace(go.Scatter(x=series_data.index, y=series_data.values, mode='lines', name=series_id ))
        if show_recession_bands and recession_data_series is not None: add_recession_bands_to_fig(fig, recession_data_series)
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title=y_axis_label, height=500)
    else:
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title=y_axis_label, height=500, annotations=[dict(text=f"Could not load data for FRED series '{series_id}'.", showarrow=False, align='center')])
    return fig

def create_fed_jaws_plot(jaws_data_df, recession_data_series=None, show_recession_bands=False): 
    fig = go.Figure(); plot_title = f"Fed's Jaws: Key Policy Rates (Last {config.FED_JAWS_DURATION_DAYS} Days)"
    if jaws_data_df is None or jaws_data_df.empty:
        fig.update_layout(title=plot_title, height=500, annotations=[dict(text=f"No data for Fed's Jaws chart.", showarrow=False, align='center')], xaxis_title='Date', yaxis_title='Percent (%)'); return fig
    series_names = {'DFEDTARU': "Target Range - Upper", 'IORB': "IORB", 'DPCREDIT': "Discount Window", 'SOFR': "SOFR", 'DFF': "EFFR", 'OBFR': "OBFR", 'DFEDTARL': "Target Range - Lower"}
    for series_id in config.FED_JAWS_SERIES_IDS:
        if series_id in jaws_data_df.columns: 
            fig.add_trace(go.Scatter(x=jaws_data_df.index, y=jaws_data_df[series_id], mode='lines', name=series_names.get(series_id, series_id), line=dict(dash='dot' if series_id in ['DFEDTARU', 'DFEDTARL'] else None)))
        else: print(f"Warning: Series {series_id} not found in Jaws data for plotting.")
    if show_recession_bands and recession_data_series is not None: add_recession_bands_to_fig(fig, recession_data_series)
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Percent (%)', height=600, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    return fig

def create_ffr_pce_comparison_plot(data_df, ffr_series_id, pce_index_series_id, threshold=2.0, recession_data_series=None, show_recession_bands=False): 
    fig = go.Figure(); plot_title = "Federal Funds Rate vs. Core PCE Inflation (YoY)"
    if data_df is None or data_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No data available for FFR vs PCE plot.", showarrow=False, align='center')]); return fig
    plot_df = data_df.copy()
    if ffr_series_id not in plot_df.columns:
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing FFR column: '{ffr_series_id}'. Available: {list(plot_df.columns)}", showarrow=False, align='center')]); return fig
    if pce_index_series_id not in plot_df.columns:
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing PCE Index column: '{pce_index_series_id}'. Available: {list(plot_df.columns)}", showarrow=False, align='center')]); return fig
    plot_df.sort_index(inplace=True)
    pce_yoy_col = 'PCE_YoY_Calculated'; plot_df[pce_yoy_col] = plot_df[pce_index_series_id].pct_change(periods=12) * 100
    plot_df.dropna(subset=[ffr_series_id, pce_yoy_col], inplace=True) 
    if plot_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="Not enough overlapping data after calculating PCE YoY.", showarrow=False, align='center')]); return fig
    ffr = plot_df[ffr_series_id]; pce_yoy = plot_df[pce_yoy_col]; diff = ffr - pce_yoy
    fig.add_trace(go.Scatter(x=plot_df.index, y=ffr, mode='lines', name=config.FFR_VS_PCE_NAMES.get("ffr"), line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=pce_yoy, mode='lines', name=config.FFR_VS_PCE_NAMES.get("core_pce_yoy_calculated"), line=dict(color='red', width=2)))
    if show_recession_bands and recession_data_series is not None: add_recession_bands_to_fig(fig, recession_data_series)
    is_risk = diff > threshold; group_ids = is_risk.ne(is_risk.shift()).cumsum(); leg_risk, leg_norm = False, False
    for _, seg in plot_df.groupby(group_ids):
        if seg.empty or len(seg) < 2: continue
        seg_is_risk = (seg[ffr_series_id].iloc[0] - seg[pce_yoy_col].iloc[0]) > threshold 
        x_coords, y_coords = list(seg.index) + list(seg.index[::-1]), list(seg[ffr_series_id]) + list(seg[pce_yoy_col][::-1])
        fc = 'rgba(231,76,60,0.3)' if seg_is_risk else 'rgba(52,152,219,0.3)'; ln, sl = None, False
        if seg_is_risk and not leg_risk: ln, sl, leg_risk = f'Gap (FFR-PCE > {threshold}%)', True, True
        elif not seg_is_risk and not leg_norm: ln, sl, leg_norm = f'Gap (FFR-PCE <= {threshold}%)', True, True
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, fill='toself', fillcolor=fc, line_width=0, name=ln, showlegend=sl, hoverinfo='skip'))
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Percent (%)', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_gold_vs_real_yield_plot(data_df, yfinance_gold_col_name, fred_real_yield_col_name, recession_data_series=None, show_recession_bands=False): 
    fig = go.Figure(); plot_title = "Gold Price vs. 10-Year Real Yield"
    
    # print(f"PLOTTER DEBUG: create_gold_vs_real_yield_plot received yfinance_gold_col_name = '{yfinance_gold_col_name}'")
    # print(f"PLOTTER DEBUG: create_gold_vs_real_yield_plot received fred_real_yield_col_name = '{fred_real_yield_col_name}'")
    # if data_df is not None: print(f"PLOTTER DEBUG: data_df columns received by plotter: {list(data_df.columns)}")
    # else: print("PLOTTER DEBUG: data_df is None when received by plotter.")

    if data_df is None or data_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No data for Gold vs Real Yield plot.", showarrow=False, align='center')]); return fig
    
    plot_df = data_df.copy()
    
    if yfinance_gold_col_name not in plot_df.columns:
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing Gold data column: '{yfinance_gold_col_name}'. Available: {list(plot_df.columns)}", showarrow=False, align='center')]); return fig
    if fred_real_yield_col_name not in plot_df.columns:
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing Real Yield data column: '{fred_real_yield_col_name}'. Available: {list(plot_df.columns)}", showarrow=False, align='center')]); return fig
        
    plot_df.dropna(subset=[yfinance_gold_col_name, fred_real_yield_col_name], how='any', inplace=True) 
    if plot_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No overlapping data for Gold and Real Yield.", showarrow=False, align='center')]); return fig
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[yfinance_gold_col_name], name=config.GOLD_VS_REAL_YIELD_NAMES.get("gold_yfinance"), line=dict(color='gold'), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[fred_real_yield_col_name], name=config.GOLD_VS_REAL_YIELD_NAMES.get("real_yield_10y"), line=dict(color='green'), yaxis="y2"))
    
    if show_recession_bands and recession_data_series is not None: 
        add_recession_bands_to_fig(fig, recession_data_series)
        
    fig.update_layout(
        title=plot_title, 
        xaxis_title='Date', 
        height=600,
        yaxis=dict(
            title=dict(
                text=config.GOLD_VS_REAL_YIELD_NAMES.get("gold_yfinance"),
                font=dict(color="gold") # Font styling for the title
            ),
            tickfont=dict(color="gold") # Font styling for tick labels
        ),
        yaxis2=dict(
            title=dict(
                text=config.GOLD_VS_REAL_YIELD_NAMES.get("real_yield_10y"),
                font=dict(color="green") # Font styling for the title
            ),
            tickfont=dict(color="green"), # Font styling for tick labels
            overlaying="y", 
            side="right"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
