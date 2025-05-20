# plotters.py
"""
Plotting functions for the Streamlit Financial Dashboard.
Uses Plotly for creating interactive charts.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np # For isinstance checks with numeric types

# Import constants from config
import config

def create_corr_plot(rolling_corr_data, ticker1, ticker2, window=config.ROLLING_WINDOW, years=config.YEARS_OF_DATA):
    """Creates the Plotly figure for visualizing rolling correlation."""
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"

    if rolling_corr_data is not None and not rolling_corr_data.empty:
        pos_corr = rolling_corr_data.copy()
        pos_corr[pos_corr < 0] = None 
        neg_corr = rolling_corr_data.copy()
        neg_corr[neg_corr >= 0] = None 

        fig.add_trace(go.Scatter(
            x=pos_corr.index, y=pos_corr, mode='lines', name='Positive Correlation',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=neg_corr.index, y=neg_corr, mode='lines', name='Negative Correlation',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=plot_title,
            xaxis_title='Date',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1], 
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    elif rolling_corr_data is not None and rolling_corr_data.empty:
        error_message = f"No correlation data available for {ticker1} / {ticker2} with the selected parameters."
        fig.update_layout(annotations=[dict(text=error_message, showarrow=False, align='center')])
    else:
        error_message = f"Could not load or process correlation data for {ticker1} / {ticker2}."
        fig.update_layout(annotations=[dict(text=error_message, showarrow=False, align='center')])

    if rolling_corr_data is None or rolling_corr_data.empty:
         fig.update_layout(
            title=plot_title, 
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
    data_plotted = False 

    if calls_df is not None and not calls_df.empty and 'strike' in calls_df.columns and 'impliedVolatility' in calls_df.columns:
        calls_df_cleaned = calls_df.copy()
        calls_df_cleaned['strike'] = pd.to_numeric(calls_df_cleaned['strike'], errors='coerce')
        calls_df_cleaned['impliedVolatility'] = pd.to_numeric(calls_df_cleaned['impliedVolatility'], errors='coerce')
        calls_df_cleaned.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not calls_df_cleaned.empty:
            fig.add_trace(go.Scatter(
                x=calls_df_cleaned['strike'],
                y=calls_df_cleaned['impliedVolatility'] * 100, 
                mode='markers+lines',
                name='Calls IV (%)',
                marker=dict(color='blue'),
                line=dict(color='blue')
            ))
            data_plotted = True

    if puts_df is not None and not puts_df.empty and 'strike' in puts_df.columns and 'impliedVolatility' in puts_df.columns:
        puts_df_cleaned = puts_df.copy()
        puts_df_cleaned['strike'] = pd.to_numeric(puts_df_cleaned['strike'], errors='coerce')
        puts_df_cleaned['impliedVolatility'] = pd.to_numeric(puts_df_cleaned['impliedVolatility'], errors='coerce')
        puts_df_cleaned.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not puts_df_cleaned.empty:
            fig.add_trace(go.Scatter(
                x=puts_df_cleaned['strike'],
                y=puts_df_cleaned['impliedVolatility'] * 100, 
                mode='markers+lines',
                name='Puts IV (%)',
                marker=dict(color='orange'),
                line=dict(color='orange')
            ))
            data_plotted = True

    fig.update_layout(
        title=plot_title,
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if current_price is not None and isinstance(current_price, (int, float, np.number)): 
        fig.add_vline(
            x=current_price,
            line_width=1,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"Current Price: {current_price:.2f}",
            annotation_position="top right"
        )

    if not data_plotted:
        error_message = f"No valid options IV data found for {ticker} expiring {expiry_date}."
        fig.add_annotation(text=error_message, showarrow=False, align='center')
        fig.update_layout(yaxis_range=[0, 100]) 

    return fig

def create_fred_plot(series_data, series_id, series_info):
    """Creates the Plotly figure for a SINGLE FRED time series and displays metadata."""
    fig = go.Figure()
    plot_title = f"FRED Series: {series_id}" 
    y_axis_label = "Value" 

    if series_info is not None and not series_info.empty:
        plot_title = series_info.get('title', plot_title)
        units = series_info.get('units_short', 'Value')
        freq = series_info.get('frequency_short', '')
        adj = series_info.get('seasonal_adjustment_short', 'NSA') 
        y_axis_label = f"{units} ({freq}, {adj})" if freq else f"{units} ({adj})"

    if series_data is not None and not series_data.empty:
        fig.add_trace(go.Scatter(
            x=series_data.index,
            y=series_data.values,
            mode='lines',
            name=series_id 
        ))
        fig.update_layout(
            title=plot_title,
            xaxis_title='Date',
            yaxis_title=y_axis_label,
            height=500
        )
    else:
        error_message = f"Could not load data for FRED series '{series_id}'."
        fig.update_layout(
            title=plot_title, 
            xaxis_title='Date',
            yaxis_title=y_axis_label, 
            height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')]
        )
    return fig

def create_fed_jaws_plot(jaws_data):
    """Creates the Plotly figure for the Fed's Jaws chart."""
    fig = go.Figure()
    plot_title = f"Fed's Jaws: Key Policy Rates (Last {config.FED_JAWS_DURATION_DAYS} Days)"

    if jaws_data is None or jaws_data.empty:
        error_message = f"No data available to plot for the Fed's Jaws chart (last {config.FED_JAWS_DURATION_DAYS} days)."
        fig.update_layout(
            title=plot_title,
            height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')],
            xaxis_title='Date',
            yaxis_title='Percent (%)' 
        )
        return fig

    target_range_line_style = dict(color='red', width=2, dash='dot')
    series_names = {
        'DFEDTARU': "Target Range - Upper Limit", 'IORB': "Interest Rate on Reserve Balances",
        'DPCREDIT': "Discount Window Primary Credit Rate", 'SOFR': "Secured Overnight Financing Rate",
        'DFF': "Effective Federal Funds Rate (EFFR)", 'OBFR': "Overnight Bank Funding Rate",
        'DFEDTARL': "Target Range - Lower Limit",
    }
    plot_order = config.FED_JAWS_SERIES_IDS

    for series_id in plot_order:
        if series_id in jaws_data.columns:
            line_style = target_range_line_style if series_id in ['DFEDTARU', 'DFEDTARL'] else None
            series_name = series_names.get(series_id, series_id)
            fig.add_trace(go.Scatter(
                x=jaws_data.index, y=jaws_data[series_id], mode='lines',
                name=series_name, line=line_style
            ))
        else:
             print(f"Warning: Series ID '{series_id}' for Jaws plot not found in fetched jaws_data.")

    fig.update_layout(
        title=plot_title, xaxis_title='Date', yaxis_title='Percent (%)', height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig

def create_ffr_pce_comparison_plot(data_df, ffr_series_id, pce_index_series_id, threshold=2.0):
    """
    Creates a plot comparing Fed Funds Rate and Core PCE YoY inflation (calculated from index).
    Colors the area between the two lines based on whether their difference exceeds a threshold.
    `data_df` is expected to contain columns for `ffr_series_id` and `pce_index_series_id`.
    """
    fig = go.Figure()
    plot_title = "Federal Funds Rate vs. Core PCE Inflation (YoY)"
    
    if data_df is None or data_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="No data available for FFR vs PCE plot.", showarrow=False, align='center')])
        return fig

    plot_df = data_df.copy()

    # Ensure the source columns for FFR and PCE Index exist
    if ffr_series_id not in plot_df.columns or pce_index_series_id not in plot_df.columns:
        missing_cols = []
        if ffr_series_id not in plot_df.columns: missing_cols.append(ffr_series_id)
        if pce_index_series_id not in plot_df.columns: missing_cols.append(pce_index_series_id) # This should be PCEPILFE
        fig.update_layout(title=plot_title, annotations=[dict(text=f"Missing source data columns: {', '.join(missing_cols)}", showarrow=False, align='center')])
        return fig

    # Calculate Year-over-Year for PCE Index
    plot_df.sort_index(inplace=True)
    pce_yoy_calculated_col_name = 'PCE_YoY_Calculated' # Internal column name
    plot_df[pce_yoy_calculated_col_name] = plot_df[pce_index_series_id].pct_change(periods=12) * 100
    
    # Drop rows where FFR or the *calculated* PCE YoY is NaN
    plot_df.dropna(subset=[ffr_series_id, pce_yoy_calculated_col_name], inplace=True)

    if plot_df.empty:
        fig.update_layout(title=plot_title, annotations=[dict(text="Not enough overlapping data after calculating PCE YoY.", showarrow=False, align='center')])
        return fig

    ffr_rate = plot_df[ffr_series_id]
    pce_rate_yoy = plot_df[pce_yoy_calculated_col_name] 
    difference = ffr_rate - pce_rate_yoy

    # Plot the main lines
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=ffr_rate, mode='lines', name=config.FFR_VS_PCE_NAMES.get("ffr", ffr_series_id),
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=pce_rate_yoy, mode='lines', name=config.FFR_VS_PCE_NAMES.get("core_pce_yoy_calculated", "Core PCE YoY (Calc.)"),
        line=dict(color='red', width=2)
    ))
    
    is_risk_condition = difference > threshold
    group_ids = is_risk_condition.ne(is_risk_condition.shift()).cumsum()
    legend_added_risk = False
    legend_added_normal = False

    for _, segment in plot_df.groupby(group_ids):
        if segment.empty or len(segment) < 2: 
            continue
            
        segment_ffr_val = segment[ffr_series_id].iloc[0]
        segment_pce_yoy_val = segment[pce_yoy_calculated_col_name].iloc[0] 
        
        if pd.isna(segment_ffr_val) or pd.isna(segment_pce_yoy_val):
            continue

        segment_is_risk = (segment_ffr_val - segment_pce_yoy_val) > threshold
        
        x_coords = list(segment.index) + list(segment.index[::-1])
        y_coords = list(segment[ffr_series_id]) + list(segment[pce_yoy_calculated_col_name][::-1]) 
        
        fill_color = 'rgba(231, 76, 60, 0.3)' if segment_is_risk else 'rgba(52, 152, 219, 0.3)'
        legend_name = None
        show_legend_for_segment = False

        if segment_is_risk and not legend_added_risk:
            legend_name = f'Gap (FFR - Calc. PCE YoY > {threshold}%)'
            show_legend_for_segment = True
            legend_added_risk = True
        elif not segment_is_risk and not legend_added_normal:
            legend_name = f'Gap (FFR - Calc. PCE YoY <= {threshold}%)'
            show_legend_for_segment = True
            legend_added_normal = True
            
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor=fill_color,
            line_width=0, 
            name=legend_name,
            showlegend=show_legend_for_segment,
            hoverinfo='skip' 
        ))

    fig.update_layout(
        title=plot_title,
        xaxis_title='Date',
        yaxis_title='Percent (%)',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

