# utils.py
"""
Utility functions for the Streamlit Financial Dashboard.
"""
import streamlit as st

def init_state(key, default_value):
    """Initializes a key in st.session_state if it doesn't exist."""
    if key not in st.session_state:
        st.session_state[key] = default_value

def generate_footer_html(name, url, svg_icon):
    """Generates the HTML for the footer."""
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 15px 0; color: grey; font-size: 0.875rem;">
        <span style="margin-right: 10px;">Created by {name}</span>
        <a href="{url}" target="_blank" style="text-decoration: none; color: grey; display: inline-flex; align-items: center;" title="LinkedIn Profile">
            {svg_icon}
        </a>
    </div>
    """
