# plotters.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
import config 

def add_recession_bands_to_fig(fig, recession_data_series, y_range=None):
    """Adds recession bands to a Plotly figure if recession_data_series is provided."""
    if recession_data_series is None or recession_data_series.empty:
        return

    # Ensure dates are in datetime format for comparison
    recession_data_series.index = pd.to_datetime(recession_data_series.index)
    
    # Find periods where recession indicator is 1
    in_recession = recession_data_series[recession_data_series == 1]
    
    start_date = None
    for date, value in in_recession.items():
        if start_date is None:
            start_date = date
        # Check if next entry exists and if it's not a continuation or if it's the last entry
        current_idx = recession_data_series.index.get_loc(date)
        if current_idx + 1 < len(recession_data_series.index):
            next_date = recession_data_series.index[current_idx + 1]
            # If there's a gap (more than typical monthly interval + buffer) or next is not recession
            if (next_date - date).days > 45 or recession_data_series.get(next_date, 0) == 0:
                end_date = date 
                fig.add_vrect(x0=start_date, x1=end_date, 
                              fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0,
                              name="NBER Recession") # Name for potential legend filtering
                start_date = None 
        elif start_date: # Last point and in recession
            end_date = date
            fig.add_vrect(x0=start_date, x1=end_date, 
                          fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0, name="NBER Recession")
            start_date = None # Reset
    # Ensure unique legend items for recession bands if multiple traces are added this way
    # This simple approach might add multiple "NBER Recession" legends if not handled by Plotly
    # A more robust way is to add one trace for all bands or manage legendgroup. For now, this is ok.

def create_corr_plot(rolling_corr_data, ticker1, ticker2, window, years=config.YEARS_OF_DATA_CORR):
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"
    if rolling_corr_data is not None and not rolling_corr_data.empty:
        pos_corr = rolling_corr_data.copy(); pos_corr[pos_corr < 0] = None 
        neg_corr = rolling_corr_data.copy(); neg_corr[neg_corr >= 0] = None 
        fig.add_trace(go.Scatter(x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation', line=dict(color='red')))
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    elif rolling_corr_data is not None and rolling_corr_data.empty: 
        fig.update_layout(annotations=[dict(text=f"No correlation data for {ticker1}/{ticker2} ({window}-day window).", showarrow=False)])
    else: 
        fig.update_layout(annotations=[dict(text=f"Could not load correlation data for {ticker1}/{ticker2}.", showarrow=False)])
    if rolling_corr_data is None or rolling_corr_data.empty:
         fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Correlation Coefficient', yaxis_range=[-1, 1], height=500)
    return fig

def interpolate_iv(df_options):
    if df_options is None or df_options.empty or 'strike' not in df_options.columns or 'impliedVolatility' not in df_options.columns:
        return df_options
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
    fig = go.Figure()
    plot_title = f"{ticker} Implied Volatility Skew (Expiry: {expiry_date})"
    data_plotted = False 
    calls_df_processed = interpolate_iv(calls_df)
    puts_df_processed = interpolate_iv(puts_df)
    if calls_df_processed is not None and not calls_df_processed.empty:
        fig.add_trace(go.Scatter(x=calls_df_processed['strike'],y=calls_df_processed['impliedVolatility'] * 100, mode='markers+lines', name='Calls IV (%)', marker=dict(color='blue'), line=dict(color='blue')))
        data_plotted = True
    if puts_df_processed is not None and not puts_df_processed.empty:
        fig.add_trace(go.Scatter(x=puts_df_processed['strike'], y=puts_df_processed['impliedVolatility'] * 100, mode='markers+lines', name='Puts IV (%)', marker=dict(color='orange'), line=dict(color='orange')))
        data_plotted = True
    fig.update_layout(title=plot_title, xaxis_title='Strike Price', yaxis_title='Implied Volatility (%)', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if current_price is not None and isinstance(current_price, (int, float, np.number)): 
        fig.add_vline(x=current_price, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Current Price: {current_price:.2f}", annotation_position="top right")
    if not data_plotted:
        fig.add_annotation(text=f"No valid options IV data for {ticker} expiring {expiry_date} after processing.", showarrow=False)
        fig.update_layout(yaxis_range=[0, 100]) 
    return fig

def create_fred_plot(series_data, series_id, series_info, recession_data=None, show_recession_bands=False):
    fig = go.Figure()
    plot_title = f"FRED Series: {series_id}" 
    y_axis_label = "Value" 
    if series_info is not None and not series_info.empty:
        plot_title = series_info.get('title', plot_title); units = series_info.get('units_short', 'Value')
        freq = series_info.get('frequency_short', ''); adj = series_info.get('seasonal_adjustment_short', 'NSA') 
        y_axis_label = f"{units} ({freq}, {adj})" if freq else f"{units} ({adj})"
    if series_data is not None and not series_data.empty:
        fig.add_trace(go.Scatter(x=series_data.index, y=series_data.values, mode='lines', name=series_id ))
        if show_recession_bands and recession_data is not None: add_recession_bands_to_fig(fig, recession_data.get(config.USREC_SERIES_ID))
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title=y_axis_label, height=500)
    else:
        fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title=y_axis_label, height=500, annotations=[dict(text=f"Could not load data for FRED series '{series_id}'.", showarrow=False)])
    return fig

def create_fed_jaws_plot(jaws_data, recession_data=None, show_recession_bands=False):
    fig = go.Figure(); plot_title = f"Fed's Jaws: Key Policy Rates (Last {config.FED_JAWS_DURATION_DAYS} Days)"
    if jaws_data is None or jaws_data.empty:
        fig.update_layout(title=plot_title, height=500, annotations=[dict(text=f"No data for Fed's Jaws chart (last {config.FED_JAWS_DURATION_DAYS} days).", showarrow=False)], xaxis_title='Date', yaxis_title='Percent (%)')
        return fig
    series_names = {'DFEDTARU': "Target Range - Upper", 'IORB': "IORB", 'DPCREDIT': "Discount Window", 'SOFR': "SOFR", 'DFF': "EFFR", 'OBFR': "OBFR", 'DFEDTARL': "Target Range - Lower"}
    for series_id in config.FED_JAWS_SERIES_IDS:
        if series_id in jaws_data.columns:
            fig.add_trace(go.Scatter(x=jaws_data.index, y=jaws_data[series_id], mode='lines', name=series_names.get(series_id, series_id), line=dict(dash='dot' if series_id in ['DFEDTARU', 'DFEDTARL'] else None)))
    if show_recession_bands and recession_data is not None: add_recession_bands_to_fig(fig, recession_data.get(config.USREC_SERIES_ID))
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Percent (%)', height=600, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    return fig

def create_ffr_pce_comparison_plot(data_df, ffr_series_id, pce_index_series_id, threshold=2.0, recession_data_series=None, show_recession_bands=False):
    fig = go.Figure(); plot_title = "Federal Funds Rate vs. Core PCE Inflation (YoY)"
    if data_df is None or data_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No data for FFR vs PCE plot.", showarrow=False)]); return fig
    plot_df = data_df.copy()
    if not (ffr_series_id in plot_df.columns and pce_index_series_id in plot_df.columns):
        missing = [s for s in [ffr_series_id, pce_index_series_id] if s not in plot_df.columns]
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing source columns: {', '.join(missing)}", showarrow=False)]); return fig
    plot_df.sort_index(inplace=True)
    pce_yoy_col = 'PCE_YoY_Calc'; plot_df[pce_yoy_col] = plot_df[pce_index_series_id].pct_change(periods=12) * 100
    plot_df.dropna(subset=[ffr_series_id, pce_yoy_col], inplace=True)
    if plot_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="Not enough data after PCE YoY calc.", showarrow=False)]); return fig
    ffr = plot_df[ffr_series_id]; pce_yoy = plot_df[pce_yoy_col]; diff = ffr - pce_yoy
    fig.add_trace(go.Scatter(x=plot_df.index, y=ffr, mode='lines', name=config.FFR_VS_PCE_NAMES.get("ffr"), line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=pce_yoy, mode='lines', name=config.FFR_VS_PCE_NAMES.get("core_pce_yoy_calculated"), line=dict(color='red', width=2)))
    if show_recession_bands and recession_data_series is not None: add_recession_bands_to_fig(fig, recession_data_series)
    is_risk = diff > threshold; group_ids = is_risk.ne(is_risk.shift()).cumsum()
    leg_risk, leg_norm = False, False
    for _, seg in plot_df.groupby(group_ids):
        if seg.empty or len(seg) < 2: continue
        seg_is_risk = (seg[ffr_series_id].iloc[0] - seg[pce_yoy_col].iloc[0]) > threshold
        x, y = list(seg.index) + list(seg.index[::-1]), list(seg[ffr_series_id]) + list(seg[pce_yoy_col][::-1])
        fc = 'rgba(231,76,60,0.3)' if seg_is_risk else 'rgba(52,152,219,0.3)'; ln, sl = None, False
        if seg_is_risk and not leg_risk: ln, sl, leg_risk = f'Gap (FFR-PCE > {threshold}%)', True, True
        elif not seg_is_risk and not leg_norm: ln, sl, leg_norm = f'Gap (FFR-PCE <= {threshold}%)', True, True
        fig.add_trace(go.Scatter(x=x, y=y, fill='toself', fillcolor=fc, line_width=0, name=ln, showlegend=sl, hoverinfo='skip'))
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Percent (%)', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_gold_vs_real_yield_plot(data_df, gold_series_id, real_yield_series_id, recession_data_series=None, show_recession_bands=False):
    fig = go.Figure(); plot_title = "Gold Price vs. 10-Year Real Yield"
    if data_df is None or data_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No data for Gold vs Real Yield plot.", showarrow=False)]); return fig
    plot_df = data_df.copy()
    if not (gold_series_id in plot_df.columns and real_yield_series_id in plot_df.columns):
        missing = [s for s in [gold_series_id, real_yield_series_id] if s not in plot_df.columns]
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing source columns: {', '.join(missing)}", showarrow=False)]); return fig
    plot_df.dropna(subset=[gold_series_id, real_yield_series_id], how='any', inplace=True) # Ensure both have data for a point
    if plot_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No overlapping data for Gold and Real Yield.", showarrow=False)]); return fig
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[gold_series_id], name=config.GOLD_VS_REAL_YIELD_NAMES.get("gold"), line=dict(color='gold'), yaxis="y1"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[real_yield_series_id], name=config.GOLD_VS_REAL_YIELD_NAMES.get("real_yield_10y"), line=dict(color='green'), yaxis="y2"))
    
    if show_recession_bands and recession_data_series is not None: add_recession_bands_to_fig(fig, recession_data_series)
        
    fig.update_layout(
        title=plot_title, xaxis_title='Date', height=600,
        yaxis=dict(title=config.GOLD_VS_REAL_YIELD_NAMES.get("gold"), titlefont=dict(color="gold"), tickfont=dict(color="gold")),
        yaxis2=dict(title=config.GOLD_VS_REAL_YIELD_NAMES.get("real_yield_10y"), titlefont=dict(color="green"), tickfont=dict(color="green"), overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
