# utils.py
"""
Utility functions for the Streamlit Financial Dashboard.
"""
import streamlit as st
import pandas as pd

def init_state(key, default_value):
    """Initializes a key in session state if it doesn't exist."""
    if key not in st.session_state:
        st.session_state[key] = default_value

def generate_footer_html(your_name, linkedin_url, linkedin_svg_path):
    """Generates the HTML for the footer."""
    return f"""
    <div style="text-align: center; margin-top: 50px; margin-bottom: 20px; font-size: 12px; color: #888;">
        Developed by {your_name} {linkedin_svg_path} 
        <a href="{linkedin_url}" target="_blank" style="color: #0077B5; text-decoration: none;">LinkedIn</a><br>
    </div>
    """

@st.cache_data # Cache the conversion if the DataFrame is large and conversion is slow
def convert_df_to_csv(df: pd.DataFrame):
    """Converts a Pandas DataFrame to a CSV string."""
    if df is None:
        return ""
    return df.to_csv(index=True).encode('utf-8')

