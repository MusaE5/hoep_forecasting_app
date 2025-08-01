import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


# ----------------------------------------------------------------------
# 1) Page-level settings
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="HOEP Forecasting App",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------------------------------------------------
# 2) Global dark theme + tweaks
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #000000;
            --fg: #ffffff;
            --card: #1a1a1a;
            --accent1: #ffd700;      /* gold */
            --accent2: #4ecdc4;      /* teal */
        }

        .stApp { background: var(--bg); color: var(--fg); }

        h1, h2, h3, h4 { color: var(--fg); }

        /* Card container (metric) */
        [data-testid="metric-container"]{
            background: var(--card);
            padding: 1rem 1.25rem;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 0 2px 4px rgba(0,0,0,0.6);
        }

        /* Bigger label inside metric (Last / Next Prediction) */
        [data-testid="metric-container"] > label {
            font-size: 1.15rem;
            font-weight: 600;
        }

        /* Make the delta pill + timer larger */
        [data-testid="metric-container"] .stMarkdown, 
        [data-testid="metric-delta"] {
            font-size: 1.05rem !important;
            font-weight: 600;
        }

        /* Auto-refresh fade indicator */
        .refresh-fade {
            animation: fade 30s infinite;
        }
        @keyframes fade { 0%,95%{opacity:1;} 100%{opacity:0.7;} }
    </style>

    <script>
        // Hard refresh every 30 s so countdown stays true
        setTimeout(()=>window.location.reload(), 30000);
    </script>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 3) Header
# ----------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center'>⚡ Ontario Electricity Price Forecaster</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align:center; font-weight:normal'>"
    "Live HOEP predictions"
    "</h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ----------------------------------------------------------------------
# 4) Metrics  (Next-on-left, Last-on-right)
# ----------------------------------------------------------------------
st.markdown("### 📊 Latest Forecast")

# -------  TEMP placeholders  ------------------------------------------
# Conig variables for data insertion
df = pd.read_csv('data/predictions_log.csv')
df['predicted_for_hour'] = pd.to_datetime(df['predicted_for_hour'])
df['timestamp_predicted_at'] = pd.to_datetime(df['timestamp_predicted_at'])
# Set variable to read most recent prediction
latest_row = df.sort_values('timestamp_predicted_at', ascending=False).iloc[0]

latest_predicted_hour = latest_row['predicted_for_hour']
latest_prediction_time = latest_row['timestamp_predicted_at']
latest_pred_q10 = latest_row['pred_q10']
latest_pred_q50 = latest_row['pred_q50']
latest_pred_q90 = latest_row['pred_q90']

first_actual_hoep = df.loc[0, 'actual_hoep'] if pd.notna(df.loc[0, 'actual_hoep']) else None

# Get current time
now = datetime.now()
current_hour = now.replace(minute=0, second=0, microsecond=0)

# Prediction happens at HH:55 of current hour
next_prediction_time = current_hour.replace(minute=55)
if now.minute >= 55:
    next_prediction_time += timedelta(hours=1)

# Forecast window is 2 hours **after the current hour**, not after the prediction time
target_start = (next_prediction_time.replace(minute=0) + timedelta(hours=2)).strftime('%H:%M')
target_end = (next_prediction_time.replace(minute=0) + timedelta(hours=3) - timedelta(minutes=1)).strftime('%H:%M')
target_hour_label = f"{target_start}–{target_end} EST"

# Countdown to next prediction
delta = next_prediction_time - now
countdown = f"{delta.seconds // 60}m {delta.seconds % 60}s"

# Create layout
col_next, col_last = st.columns(2)

with col_next:
    st.markdown(
        f"""
        <div style="
            background:#1a1a1a; border:1px solid #333; border-radius:12px;
            padding:1.5rem 1.8rem; height:100%;">
          <div style="font-size:1.4rem; font-weight:700; line-height:1.2;">⏳ Next Prediction</div>
          <div style="font-size:1.2rem; color:#ccc; margin-top:0.4rem;">
            Predicting for:
          </div>
          <div style="font-size:2.2rem; font-weight:800; margin:0.2rem 0 0.2rem 0;">
            {target_hour_label}
          </div>
          <div style="font-size:1.2rem; color:#aaa;">
            in {countdown}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_last:
    st.markdown(
        f"""
        <div style="
            background:#1a1a1a; border:1px solid #333; border-radius:12px;
            padding:1.5rem 1.8rem; height:100%;">
          <div style="font-size:1.4rem; font-weight:700; line-height:1.2;">🕒 Most Recent Prediction</div>
          <div style="font-size:2.6rem; font-weight:800; margin:0.3rem 0 0.2rem 0;">
            ${latest_pred_q50:.2f}/MWh
          </div>
          <div style="font-size:1.15rem; color:#ccc; margin-bottom:0.8rem;">
            for hour {latest_predicted_hour.strftime('%H:%M')}–{(latest_predicted_hour + pd.Timedelta(hours=1) - pd.Timedelta(minutes=1)).strftime('%H:%M')} EST
          </div>
          <div style="
               display:inline-block; padding:0.35rem 0.9rem;
               background:#0b6623; color:#fff; font-weight:600;
               border-radius:999px; font-size:1.05rem;">
               Predicted range: ${latest_pred_q10:.2f}–${latest_pred_q90:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ----------------------------------------------------------------------
# 5) Three CTA panels with descriptions
# ----------------------------------------------------------------------
st.markdown("### 🔍 Quick Actions")

cta1, cta2, cta3 = st.columns(3)

with cta1:
    st.markdown("#### 📈 Historical Dashboard")
    st.markdown("See quantile predictions vs. actual prices with 80 % confidence bands for the last 24 h.")
    if st.button("Open Dashboard"):
        st.switch_page("pages/dashboard.py")

with cta2:
    st.markdown("#### 🧪 Manual Prediction")
    st.markdown("Trigger an on-demand forecast for the upcoming hour using the latest market + weather data.")
    if st.button("Predict Now"):
        st.switch_page("pages/predict.py")

with cta3:
    st.markdown("#### 📊 Model Metrics")
    st.markdown("Inspect accuracy, residuals, and how we stack up against IESO Hour-2 benchmarks.")
    if st.button("View Metrics"):
        st.switch_page("pages/metrics.py")

st.markdown("---")

# ----------------------------------------------------------------------
# 6) Historical HOEP trend chart
# ----------------------------------------------------------------------
BUFFER_PATH = "data/hoep_buffer.csv"

# Load buffer or create if missing
if os.path.exists(BUFFER_PATH):
    hoep_df = pd.read_csv(BUFFER_PATH)
    hoep_df['timestamp'] = pd.to_datetime(hoep_df['timestamp'])
else:
    hoep_df = pd.DataFrame(columns=["timestamp", "zonal_price"])

# Only keep latest 24
hoep_df = hoep_df.tail(24)

# Plot chart
st.markdown("### 📈 Recent HOEP Trends")
st.line_chart(
    hoep_df.set_index("timestamp")[["zonal_price"]],
    color=['#ffd700'],
    height=300,
)

last_refresh = hoep_df["timestamp"].max().strftime('%Y-%m-%d %H:%M EST')
st.markdown(f"*HOEP data updates hourly from IESO — last refresh: {last_refresh}*")

# ----------------------------------------------------------------------
# 7) Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.9rem;'>"
    "Built with Streamlit • Data: IESO • Auto-updates every hour"
    "</div>",
    unsafe_allow_html=True,
)
