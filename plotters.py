# plotters.py
"""
Plotting functions for the Streamlit Financial Dashboard.
Uses Plotly for creating interactive charts.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np # For isinstance checks with numeric types

# Import constants from config
# Ensure config.py is in the same directory or accessible in PYTHONPATH
import config

def create_corr_plot(rolling_corr_data, ticker1, ticker2, window=config.ROLLING_WINDOW, years=config.YEARS_OF_DATA):
    """Creates the Plotly figure for visualizing rolling correlation."""
    fig = go.Figure()
    plot_title = f"{ticker1} vs {ticker2} - {window}-Day Rolling Correlation ({years} Years)"

    if rolling_corr_data is not None and not rolling_corr_data.empty:
        pos_corr = rolling_corr_data.copy()
        pos_corr[pos_corr < 0] = None # Set negative values to NaN for the positive trace
        neg_corr = rolling_corr_data.copy()
        neg_corr[neg_corr >= 0] = None # Set positive values to NaN for the negative trace

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
            yaxis_range=[-1, 1], # Ensure y-axis spans the full correlation range
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Place legend above plot
        )
    # Handle cases where correlation calculation returned an empty series (e.g., not enough data)
    elif rolling_corr_data is not None and rolling_corr_data.empty:
        error_message = f"No correlation data available for {ticker1} / {ticker2} with the selected parameters."
        fig.update_layout(annotations=[dict(text=error_message, showarrow=False, align='center')])
    # Handle cases where correlation calculation failed (returned None)
    else:
        error_message = f"Could not load or process correlation data for {ticker1} / {ticker2}."
        fig.update_layout(annotations=[dict(text=error_message, showarrow=False, align='center')])

    # Ensure basic layout is set even if there's no data or an error
    if rolling_corr_data is None or rolling_corr_data.empty:
         fig.update_layout(
            title=plot_title, # Keep title
            xaxis_title='Date',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1], # Keep y-axis range
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
        # Operate on a copy to avoid SettingWithCopyWarning if calls_df is a slice
        calls_df_cleaned = calls_df.copy()
        calls_df_cleaned['strike'] = pd.to_numeric(calls_df_cleaned['strike'], errors='coerce')
        calls_df_cleaned['impliedVolatility'] = pd.to_numeric(calls_df_cleaned['impliedVolatility'], errors='coerce')
        calls_df_cleaned.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not calls_df_cleaned.empty:
            fig.add_trace(go.Scatter(
                x=calls_df_cleaned['strike'],
                y=calls_df_cleaned['impliedVolatility'] * 100, # Convert IV to percentage
                mode='markers+lines',
                name='Calls IV (%)',
                marker=dict(color='blue'),
                line=dict(color='blue')
            ))
            data_plotted = True

    # Plot Puts IV if data is available and valid
    if puts_df is not None and not puts_df.empty and 'strike' in puts_df.columns and 'impliedVolatility' in puts_df.columns:
        # Operate on a copy
        puts_df_cleaned = puts_df.copy()
        puts_df_cleaned['strike'] = pd.to_numeric(puts_df_cleaned['strike'], errors='coerce')
        puts_df_cleaned['impliedVolatility'] = pd.to_numeric(puts_df_cleaned['impliedVolatility'], errors='coerce')
        puts_df_cleaned.dropna(subset=['strike', 'impliedVolatility'], inplace=True)

        if not puts_df_cleaned.empty:
            fig.add_trace(go.Scatter(
                x=puts_df_cleaned['strike'],
                y=puts_df_cleaned['impliedVolatility'] * 100, # Convert IV to percentage
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
    if current_price is not None and isinstance(current_price, (int, float, np.number)): # np.number for numpy numeric types
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

def create_fed_jaws_plot(jaws_data):
    """Creates the Plotly figure for the Fed's Jaws chart."""
    fig = go.Figure()
    # Use FED_JAWS_DURATION_DAYS from config
    plot_title = f"Fed's Jaws: Key Policy Rates (Last {config.FED_JAWS_DURATION_DAYS} Days)"

    if jaws_data is None or jaws_data.empty:
        error_message = f"No data available to plot for the Fed's Jaws chart (last {config.FED_JAWS_DURATION_DAYS} days)."
        fig.update_layout(
            title=plot_title,
            height=500,
            annotations=[dict(text=error_message, showarrow=False, align='center')],
            xaxis_title='Date',
            yaxis_title='Percent (%)' # Default axis label
        )
        return fig

    target_range_line_style = dict(color='red', width=2, dash='dot')
    series_names = {
        'DFEDTARU': "Target Range - Upper Limit",
        'IORB':     "Interest Rate on Reserve Balances",
        'DPCREDIT': "Discount Window Primary Credit Rate",
        'SOFR':     "Secured Overnight Financing Rate",
        'DFF':      "Effective Federal Funds Rate (EFFR)",
        'OBFR':     "Overnight Bank Funding Rate",
        'DFEDTARL': "Target Range - Lower Limit",
    }
    # Use FED_JAWS_SERIES_IDS from config
    plot_order = config.FED_JAWS_SERIES_IDS

    for series_id in plot_order:
        # Check if the column actually exists before plotting (robustness)
        if series_id in jaws_data.columns:
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
             print(f"Warning: Series ID '{series_id}' for Jaws plot not found in fetched jaws_data.")

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
