import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# Page config
st.set_page_config(page_title="Quantile Dashboard", layout="wide")

# Dark background CSS + metric contrast fix
st.markdown("""
<style>
:root {
    --bg: #121418;
    --fg: #f5f5f5;
    --muted: #a0a0a0;
}
.stApp { background-color: var(--bg) !important; color: var(--fg) !important; }
header[data-testid="stHeader"] { background-color: var(--bg) !important; }
.main .block-container { background-color: var(--bg) !important; color: var(--fg) !important; }

/* Make Streamlit metrics readable on dark */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: var(--fg) !important;
    opacity: 1 !important;               /* override Streamlit's dimming */
}
</style>
""", unsafe_allow_html=True)

st.title(" 24-Hour Forecast vs. Actual HOEP")

try:
    with open('cloud_entry/data/chart_buffer.csv', 'r') as f:
        df = pd.read_csv(StringIO(f.read()))

    # Sort for rare edge case
    df[['pred_q10', 'pred_q50', 'pred_q90']] = np.sort(
        df[['pred_q10', 'pred_q50', 'pred_q90']], axis=1
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

# Quantile shading (q10..q90)
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['pred_q90'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip',
    name='q90',
))
fig.add_trace(go.ScatteR(
    x=df['predicted_for_hour'],
    y=df['pred_q10'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(0, 153, 255, 0.20)',
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
    hovertemplate="Hour: %{x}<br>q50: $%{y:.2f}<br>Predicted at: %{customdata}<extra></extra>",
    customdata=df['timestamp_predicted_at'].dt.strftime("%Y-%m-%d %H:%M:%S")
))

# Actual HOEP
fig.add_trace(go.Scatter(
    x=df['predicted_for_hour'],
    y=df['actual_hoep'],
    mode='lines+markers',
    name='Actual HOEP',
    line=dict(color='yellow'),
    hovertemplate="Hour: %{x}<br>Actual: $%{y:.2f}<extra></extra>"
))

#  Dark layout 
fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=20),
    height=700,
    xaxis_title="Forecasted Hour",
    yaxis_title="$/MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(color="#f5f5f5")),
    paper_bgcolor="rgba(0,0,0,0)",     # transparent to blend into Streamlit dark bg
    plot_bgcolor="#121418",            # chart area dark
    font=dict(color="#f5f5f5"),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.20)",
        tickfont=dict(color="#f5f5f5"),
        title_font=dict(color="#f5f5f5"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.20)",
        tickfont=dict(color="#f5f5f5"),
        title_font=dict(color="#f5f5f5"),
    ),
    # NOTE: intentionally no Plotly template here
)

st.plotly_chart(fig, use_container_width=True)

# Performance Metrics
st.markdown("###  Model Performance Metrics")
coverage = ((df['actual_hoep'] >= df['pred_q10']) & (df['actual_hoep'] <= df['pred_q90'])).mean()
mae = np.abs(df['actual_hoep'] - df['pred_q50']).mean()
col1, col2 = st.columns(2)
col1.metric("Quantile Coverage (80%)", f"{coverage:.1%}")
col2.metric("MAE (q50 vs HOEP)", f"${mae:.2f}")
