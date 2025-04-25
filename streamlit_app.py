import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

# --- Re-use the data fetching function ---
@st.cache_data(ttl=60*60)  # Cache data for 1 hour
def calculate_rolling_correlation(ticker1='EURUSD=X', ticker2='GLD', window=30, years=5):
    """
    Fetches data for two tickers and calculates their rolling correlation.
    (Cached using Streamlit)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + window + 5)  # Add buffer
    try:
        data = yf.download([ticker1, ticker2], start=start_date, end=end_date)['Close']
        data.ffill(inplace=True)
        if ticker1 not in data.columns or ticker2 not in data.columns or data[ticker1].isnull().all() or data[ticker2].isnull().all():
            return None
        returns = data.pct_change().dropna()
        rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2]).dropna()
        final_start_date = end_date - timedelta(days=years*365)
        rolling_corr = rolling_corr[rolling_corr.index >= final_start_date]
        rolling_corr.name = f'{window}d Rolling Corr'
        return rolling_corr
    except Exception as e:
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Macro Dashboard", layout="wide")

st.title("Macro Dashboard")

st.subheader("EUR/USD vs Gold (GLD) - 30-Day Rolling Correlation (10 Years)")

# Create a placeholder for the chart
placeholder = st.empty()

# Button to manually refresh the data
if st.button('Refresh Data'):
    rolling_corr = calculate_rolling_correlation(ticker1='EURUSD=X', ticker2='GLD', window=30, years=10)

    # Create the plot
    fig = go.Figure()

    if rolling_corr is not None and not rolling_corr.empty:
        # Separate positive and negative values for coloring
        pos_corr = rolling_corr.copy()
        pos_corr[pos_corr < 0] = None  # Keep only positive values

        neg_corr = rolling_corr.copy()
        neg_corr[neg_corr >= 0] = None  # Keep only negative values

        # Add green line for positive correlation
        fig.add_trace(go.Scatter(
            x=pos_corr.index,
            y=pos_corr,
            mode='lines',
            name='Positive Correlation',
            line=dict(color='green')
        ))

        # Add red line for negative correlation
        fig.add_trace(go.Scatter(
            x=neg_corr.index,
            y=neg_corr,
            mode='lines',
            name='Negative Correlation',
            line=dict(color='red')
        ))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1],
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig.update_layout(
            title='Data Unavailable',
            xaxis_title='Date',
            yaxis_title='Correlation Coefficient',
            yaxis_range=[-1, 1],
            height=500,
            annotations=[dict(text="Could not load data. Check connection or tickers.", showarrow=False)]
        )

    # Display the chart in the placeholder
    with placeholder.container():
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
