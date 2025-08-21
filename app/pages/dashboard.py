import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# Page config
st.set_page_config(page_title="Quantile Dashboard", layout="wide")

# Dark background CSS only
st.markdown("""
<style>
.stApp {
    background-color: #121418 !important;
    color: #f5f5f5 !important;
}
header[data-testid="stHeader"] {
    background-color: #121418 !important;
}
.main .block-container {
    background-color: #121418 !important;
    color: #f5f5f5 !important;
}
</style>
""", unsafe_allow_html=True)

st.title(" 24-Hour Forecast vs. Actual HOEP")
#
try:
    
    with open('cloud_entry/data/chart_buffer.csv', 'r') as f:
        df = pd.read_csv(StringIO(f.read()))
        
    # Sort for rare edge case
    df[['pred_q10', 'pred_q50', 'pred_q90']] = np.sort(
        df[['pred_q10', 'pred_q50', 'pred_q90']], 
        axis=1
    )
    # Convert datetime columns
    df['predicted_for_hour'] = pd.to_datetime(df['predicted_for_hour'], errors='coerce')
    df['timestamp_predicted_at'] = pd.to_datetime(df['timestamp_predicted_at'], errors='coerce')
    # Keep only rows with actual HOEP available (i.e., drop last 2 rows)
    df = df[df['actual_hoep'].notna()].tail(24)
except Exception as e:
    st.error(f"Failed to load chart buffer: {e}")
    st.stop()
# Plot quantile band with shading
fig = go.Figure()
# Quantile shading
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['pred_q90'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip',
    name='q90',
))
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['pred_q10'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(0, 153, 255, 0.2)',
    line=dict(width=0),
    name='80% Quantile Band',
    hoverinfo='skip'
))
# Median prediction
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['pred_q50'],
    mode='lines+markers',
    name='Median Prediction (q50)',
    line=dict(color='blue'),
    hovertemplate=
        "Hour: %{x}<br>" +
        "q50: $%{y:.2f}<br>" +
        "Predicted at: %{customdata}<extra></extra>",
    customdata=df['timestamp_predicted_at'].dt.strftime("%Y-%m-%d %H:%M:%S")
))
# Actual HOEP
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['actual_hoep'],
    mode='lines+markers',
    name='Actual HOEP',
    line=dict(color='yellow'),
    hovertemplate=
        "Hour: %{x}<br>" +
        "Actual: $%{y:.2f}<extra></extra>"
))
# Layout settings
fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=20),
    height=700,
    xaxis_title="Forecasted Hour",
    yaxis_title="$/MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)
# Performance Metrics
st.markdown("###  Model Performance Metrics")
# Quantile Coverage
coverage = ((df['actual_hoep'] >= df['pred_q10']) & (df['actual_hoep'] <= df['pred_q90'])).mean()
# MAE for median prediction
mae = np.abs(df['actual_hoep'] - df['pred_q50']).mean()
col1, col2 = st.columns(2)
col1.metric("Quantile Coverage (80%)", f"{coverage:.1%}")
col2.metric("MAE (q50 vs HOEP)", f"${mae:.2f}")
