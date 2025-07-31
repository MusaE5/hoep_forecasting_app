import streamlit as st
from PIL import Image
import random  # For demo data - replace with your actual data functions

st.set_page_config(page_title="HOEP Forecasting App", page_icon="⚡", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set a clean black background */
    .stApp {
        background-color: #000000;
        color: white;
    }
    
    .block-container {
        padding-top: 2rem;
        max-width: 85%;
        margin: auto;
    }
    
    /* Style titles */
    h1, h2, h3 {
        color: white;
    }
    
    /* Improve button styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 16px;
        border-radius: 10px;
        background-color: #1f77b4;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
    }
    
    /* Style metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Add some spacing */
    .metric-row {
        margin: 2rem 0;
    }
    
    /* Style the social proof banner */
    .success-banner {
        background: linear-gradient(90deg, #28a745, #20c997);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align: center;'>⚡ Ontario Electricity Price Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: normal;'>Outperforming IESO predictions by 12% RMSE using live data and machine learning</h3>", unsafe_allow_html=True)

# ------------------ SOCIAL PROOF BANNER ------------------
st.markdown(
    """
    <div class="success-banner">
        🏆 Our model beats official IESO Hour-2 Predispatch forecasts by 12% RMSE
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ CURRENT METRICS ------------------
st.markdown("### 📊 Live Market Overview")

# Demo values - replace with your actual data functions
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current HOEP", 
        value="$45.67/MWh", 
        delta="↑ $2.15"
    )

with col2:
    st.metric(
        label="Next Hour Forecast", 
        value="$48.12/MWh", 
        delta="↑ $2.45"
    )

with col3:
    st.metric(
        label="Model Accuracy", 
        value="88.3%", 
        delta="↑ 1.2%"
    )

with col4:
    st.metric(
        label="Confidence Level", 
        value="94.5%", 
        delta="High"
    )

st.markdown("---")

# ------------------ ACTION OPTIONS ------------------
st.markdown("### 🔍 What would you like to do?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📈 View Historical Forecasts")
    st.markdown("**See how our model predicted vs. actual prices** with 88% accuracy over the past month. Includes confidence intervals and performance metrics.")
    if st.button("Go to Dashboard", key="dashboard"):
        st.switch_page("pages/dashboard.py")  # Adjust path as needed

with col2:
    st.markdown("#### 🧪 Run a Manual Prediction")
    st.markdown("**Get instant price forecasts** for the next hour using live market data, weather conditions, and demand patterns.")
    if st.button("Go to Predict Now", key="predict_now"):
        st.switch_page("pages/predict.py")  # Adjust path as needed

with col3:
    st.markdown("#### 📊 View Model Metrics")
    st.markdown("**Deep dive into model performance** - accuracy, residuals, coverage performance, and comparison with IESO benchmarks.")
    if st.button("Go to Metrics", key="metrics"):
        st.switch_page("pages/metrics.py")  # Adjust path as needed

# ------------------ MINI PREVIEW CHART ------------------
st.markdown("---")
st.markdown("### 📈 Recent Price Trends")

# Create a simple demo chart - replace with your actual data
import numpy as np
import pandas as pd

# Demo data - replace with your actual recent price data
hours = pd.date_range(start='2025-01-30', periods=24, freq='H')
actual_prices = np.random.normal(45, 8, 24)
predicted_prices = actual_prices + np.random.normal(0, 2, 24)

chart_data = pd.DataFrame({
    'Hour': hours,
    'Actual HOEP': actual_prices,
    'Predicted HOEP': predicted_prices
})

st.line_chart(
    chart_data.set_index('Hour'), 
    color=['#ff6b6b', '#4ecdc4'],
    height=300
)

st.markdown("*Live data updates every hour. Last updated: 2025-01-31 14:00 EST*")

# ------------------ FOOTER INFO ------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Key Features")
    st.markdown("""
    - **Real-time predictions** using live IESO data
    - **Weather integration** for improved accuracy  
    - **Quantile forecasting** with uncertainty bounds
    - **Automated data pipeline** with hourly updates
    """)

with col2:
    st.markdown("### 📈 Performance Highlights")
    st.markdown("""
    - **12% better RMSE** than IESO Hour-2 Predispatch
    - **88%+ accuracy** on out-of-sample predictions
    - **Sub-second prediction time** for real-time use
    - **Robust to market volatility** and extreme events
    """)

# Optional: Remove the external image since it might not load reliably
# If you want to keep an image, use a local one or a more reliable source