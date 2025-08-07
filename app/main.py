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
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------
# 2) Global dark theme + tweaks
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #0f0f13;
            --fg: #f0f0f0;
            --card: #1e1e2a;
            --accent1: #ffd700;      /* gold */
            --accent2: #4ecdc4;      /* teal */
            --accent3: #ff6b6b;     /* coral */
            --text-muted: #a0a0a0;
        }

        .stApp { 
            background: linear-gradient(135deg, var(--bg) 0%, #1a1a24 100%);
            color: var(--fg);
        }

        h1, h2, h3, h4 { 
            color: var(--fg);
            letter-spacing: -0.5px;
        }

        /* Main header gradient */
        h1 {
            background: linear-gradient(90deg, var(--accent1) 0%, var(--accent2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }

        /* Card container (metric) */
        [data-testid="metric-container"]{
            background: var(--card);
            padding: 1.5rem 1.75rem;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
        }

        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }

        /* Bigger label inside metric */
        [data-testid="metric-container"] > label {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-muted) !important;
        }

        /* Make the delta pill + timer larger */
        [data-testid="metric-container"] .stMarkdown, 
        [data-testid="metric-delta"] {
            font-size: 1.05rem !important;
            font-weight: 600;
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(255,215,0,0.3) 50%, transparent 100%);
            margin: 2.5rem 0;
        }

        /* Button styling */
        .stButton button {
            background: linear-gradient(90deg, var(--accent1) 0%, var(--accent3) 100%);
            color: #000;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        }

        /* Chart styling */
        .stLineChart {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Auto-refresh indicator */
        .refresh-indicator {
            display: inline-block;
            background: rgba(78, 205, 196, 0.2);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-left: 0.5rem;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }

        /* Custom card styling */
        .custom-card {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            height: 100%;
            border-left: 4px solid var(--accent1);
        }

        /* Footer styling */
        .footer {
            font-size: 0.85rem;
            text-align: center;
            color: var(--text-muted);
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        /* Forecast section specific */
        .forecast-container {
            position: relative;
            margin-bottom: 2rem;
        }
        .forecast-container::after {
            content: "";
            position: absolute;
            bottom: -15px;
            left: 5%;
            width: 90%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,215,0,0.5), transparent);
        }
    </style>

    <script>
        // Hard refresh every 30 s so countdown stays true
        setTimeout(()=>window.location.reload(), 30000);
    </script>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 3) Header with animated gradient
# ----------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <h1 style='margin-bottom:0;'>Ontario Electricity Price Forecaster</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 4) Super-sized Forecast Display
# ----------------------------------------------------------------------
# Config variables for data insertion
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

# Get current time
now = datetime.now()
current_hour = now.replace(minute=0, second=0, microsecond=0)

# Prediction happens at HH:55 of current hour
next_prediction_time = current_hour.replace(minute=55)
if now.minute >= 55:
    next_prediction_time += timedelta(hours=1)

# Forecast window is 2 hours after the current hour
target_start = (next_prediction_time.replace(minute=0) + timedelta(hours=2)).strftime('%H:%M')
target_end = (next_prediction_time.replace(minute=0) + timedelta(hours=3) - timedelta(minutes=1)).strftime('%H:%M')
target_hour_label = f"{target_start}–{target_end} EST"

# Countdown to next prediction
delta = next_prediction_time - now
countdown = f"{delta.seconds // 60}m {delta.seconds % 60}s"

# Main forecast display
st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <h2 style='margin-bottom:0.5rem;'>⚡ Live Price Forecast</h2>
        <div style='font-size:1.1rem; color:var(--text-muted);'>
            Predictions with 80% confidence range
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Current Prediction with simplified range
st.markdown(
    f"""
    <div style='margin-bottom:1.5rem; padding:1.5rem;'>  
        <div style='text-align:center;'>
            <div style='font-size:1.7rem; color:#a0a0a0; margin-bottom:0.5rem;'>
                Last prediction for {latest_predicted_hour.strftime('%H:%M')}–{(latest_predicted_hour + pd.Timedelta(hours=1) - pd.Timedelta(minutes=1)).strftime('%H:%M')} EST
            </div>
            <div style='font-size:5rem; font-weight:800; color:#ffd700; line-height:1; margin:1rem 0;'>
                ${latest_pred_q50:.2f}
            </div>
            <div style='font-size:1.2rem; color:#a0a0a0; margin-bottom:1.5rem;'>
                CAD/MWh
            </div>
            <div style='
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            '>
                <div style='color:#4cd137; font-size:1.5rem;'>${latest_pred_q10:.2f}</div>
                <div style='color:#a0a0a0; font-size:1rem;'>to</div>
                <div style='color:#ff6b6b; font-size:1.5rem;'>${latest_pred_q90:.2f}</div>
            </div>
            <div style='color:#a0a0a0; font-size:0.9rem;'>
                80% confidence range (10th to 90th percentile)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Second card: Next Prediction Timer (now transparent)
st.markdown(
    f"""
    <div style='text-align:center; padding:1.5rem;'> 
        <div style='font-size:1.6rem; font-weight:700; color:#4ecdc4; margin-bottom:1rem;'>
            ⏳ Next Prediction In
        </div>
        <div style='font-size:4.5rem; font-weight:800; color:#ffd700; line-height:1; margin-bottom:1rem;'>
            {countdown}
        </div>
        <div style='font-size:1.3rem; color:#a0a0a0;'>
            For hour: <span style='font-size:1.5rem; font-weight:600; color:#f0f0f0;'>{target_hour_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------
# 5) Quick Actions
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Quick Actions")

cta1, cta2 = st.columns(2, gap="large")

with cta1:
    st.markdown(
        """
        <div class="custom-card">
            <h4 style="color:var(--accent2); margin-top:0;"> Historical Dashboard</h4>
            <p style="color:var(--text-muted);">See quantile predictions vs. actual prices with 80% confidence bands for the last 24 hours.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Open Dashboard", key="dashboard"):
        st.switch_page("pages/dashboard.py")

with cta2:
    st.markdown(
        """
        <div class="custom-card">
            <h4 style="color:var(--accent2); margin-top:0;"> Manual Prediction</h4>
            <p style="color:var(--text-muted);">Trigger forecast for the upcoming hour using the latest market + weather data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Predict Now", key="manual"):
        st.switch_page("pages/manual_prediction.py")

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
st.markdown("### Recent HOEP Trends")
st.line_chart(
    hoep_df.set_index("timestamp")[["zonal_price"]],
    color='#FFD700',
    height=400,
    use_container_width=True
)

last_refresh = hoep_df["timestamp"].max().strftime('%Y-%m-%d %H:%M EST')
st.markdown(f"<div style='color:var(--text-muted); font-size:0.9rem;'>HOEP data updates hourly from IESO — last refresh: {last_refresh}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 7) Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Built with <span style="color:var(--accent3);">♥</span> using Streamlit • Data: IESO & Canadian Government• Auto-updates every hour
    </div>
    """,
    unsafe_allow_html=True,
)