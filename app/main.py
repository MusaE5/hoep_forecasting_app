# Last data update: 2025-08-14 21:56:09
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import pytz
import time
from io import StringIO


# Page configuration
st.set_page_config(
    page_title="HOEP Forecasting App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme and styling
st.markdown("""
<style>
:root {
    --bg: #121418;
    --card-bg: #1e2027;
    --fg: #f5f5f5;
    --accent1: #FFD166; /* gold */
    --accent2: #06D6A0; /* teal */
    --accent3: #EF476F; /* red/pink */
    --muted: #a0a0a0;
}
.stApp {
    background-color: var(--bg);
    color: var(--fg);
    font-family: "Segoe UI", Arial, sans-serif;
}
h1, h2, h3 {
    font-weight: 700;
    color: var(--fg);
}
.big-number {
    font-size: 4rem;
    font-weight: 800;
    color: var(--accent1);
}
.muted {
    color: var(--muted);
    font-size: 0.9rem;
}
.card {
    background-color: var(--card-bg);
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.button-pill button {
    background-color: var(--accent2);
    color: black;
    font-weight: 600;
    border-radius: 50px;
    border: none;
    padding: 0.5rem 1.5rem;
}
.button-pill button:hover {
    background-color: #04b184;
}
</style>
""", unsafe_allow_html=True)

# Load backend data
with open('cloud_entry/data/predictions_log.csv', 'r') as f:
    df = pd.read_csv(StringIO(f.read()))


df['predicted_for_hour'] = pd.to_datetime(df['predicted_for_hour'])
df['timestamp_predicted_at'] = pd.to_datetime(df['timestamp_predicted_at'])

latest_row = df.iloc[-1]

latest_predicted_hour = latest_row['predicted_for_hour']
latest_pred_q10 = latest_row['pred_q10']
latest_pred_q50 = latest_row['pred_q50']
latest_pred_q90 = latest_row['pred_q90']

# Get current time in Toronto timezone
toronto_tz = pytz.timezone('America/Toronto')
now = datetime.now(toronto_tz)

current_hour = now.replace(minute=0, second=0, microsecond=0)
next_prediction_time = current_hour.replace(minute=56)
if now.minute >= 56:
    next_prediction_time += timedelta(hours=1)

target_start = (next_prediction_time.replace(minute=0) + timedelta(hours=2)).strftime('%H:%M')
target_end = (next_prediction_time.replace(minute=0) + timedelta(hours=3) - timedelta(minutes=1)).strftime('%H:%M')

target_hour_label = f"{target_start}–{target_end} EST"


countdown_target = int(next_prediction_time.timestamp() * 1000)

st.markdown(f"<h1>Ontario Electricity Price Forecast</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='muted'>Last updated: {latest_row['timestamp_predicted_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:1.4rem; color:var(--muted); margin-bottom:0.5rem;'>"
        f"Forecast for {latest_predicted_hour.strftime('%H:%M')}–"
        f"{(latest_predicted_hour + pd.Timedelta(hours=1) - pd.Timedelta(minutes=1)).strftime('%H:%M')} EST"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='big-number'>${latest_pred_q50:.2f}</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Median forecast (CAD/MWh)</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.2rem; font-weight:600;'>80% Confidence Band:</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:1.5rem;'>"
        f"<span style='color:var(--accent2);'>${latest_pred_q10:.2f}</span> – "
        f"<span style='color:#ff6b6b;'>${latest_pred_q90:.2f}</span>"
        f"</p>",
        unsafe_allow_html=True
    )
    with st.expander("What is the 80% Confidence Band?"):
        st.write("""
        The 80% confidence band is the range between the **10th percentile (Q10)** and the **90th percentile (Q90)** predicted prices.
        It means that, based on the model's estimates, there is an 80% probability that the actual price will fall within this range.
        - **Q10:** Lower bound (10th percentile)  
        - **Q90:** Upper bound (90th percentile)  
        - **Q50:** Median (most likely) prediction
        """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:left; margin: 0 0 0.5rem 0;'>Next Prediction In</h3>",
        unsafe_allow_html=True
    )

    
    st.components.v1.html(f"""
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: "Segoe UI", Arial, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        width: 100%;
        height: 100%;
      }}
      .countdown {{
        font-size: 4rem;
        font-weight: 800;
        line-height: 1.05;
        color: #06D6A0;
        text-align: left;
      }}
      .label {{
        margin-top: 6px;
        text-align: left;
        font-size: 1.4rem;
        font-weight: 600;
        color: #a0a0a0;
      }}
    </style>

    <div class="countdown" id="countdown">Loading...</div>
    <div class="label">For hour: {target_hour_label}</div>

    <script>
      var countDownDate = {countdown_target};
      var x = setInterval(function() {{
        var now = new Date().getTime();
        var distance = countDownDate - now;
        
        if (distance < 0) {{
          clearInterval(x);
          document.getElementById("countdown").textContent = "Now";
          return;
        }}
        
        var minutes = Math.floor(distance / (1000 * 60));
        var seconds = Math.floor((distance % (1000 * 60)) / 1000);
        
        document.getElementById("countdown").textContent = 
            minutes + "m " + seconds + "s";
      }}, 1000);
    </script>
    """, height=155)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Quick actions
st.markdown("### Quick Actions")
qa1, qa2 = st.columns(2)

with qa1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:var(--accent2);'>Historical Dashboard</h4>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>See quantile predictions vs. actual prices for the last 24 hours.</p>", unsafe_allow_html=True)
    if st.button("Open Dashboard", key="dashboard", help="View historical performance"):
        st.switch_page("pages/dashboard.py")
    st.markdown("</div>", unsafe_allow_html=True)

with qa2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:var(--accent2);'>Manual Prediction</h4>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Trigger forecast for the upcoming hour using the latest data.</p>", unsafe_allow_html=True)
    if st.button("Predict Now", key="manual", help="Run a manual forecast"):
        st.switch_page("pages/manual_prediction.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Historical chart
with open('cloud_entry/data/hoep_buffer.csv', 'r') as f:
    hoep_df = pd.read_csv(StringIO(f.read()))
    
hoep_df['timestamp'] = pd.to_datetime(hoep_df['timestamp'])
hoep_df = hoep_df.tail(24)

st.markdown("### Ontario Zonal Price Trends (CAD/MWh)")
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hoep_df["timestamp"], y=hoep_df["zonal_price"],
    mode='lines+markers', line=dict(color='#FFD166', width=2),
    marker=dict(size=4), name='HOEP'
))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=20, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    height=400
)
st.plotly_chart(fig, use_container_width=True)

if not hoep_df.empty:
    last_refresh = hoep_df["timestamp"].max().strftime('%Y-%m-%d %H:%M EST')
    st.markdown(f"<p class='muted'>Data updates hourly from IESO — last refresh: {last_refresh}</p>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='muted' style='text-align:center;'>Built with ❤️ using Streamlit • Data: IESO & Canadian Government • Auto-updates hourly</p>", unsafe_allow_html=True)
