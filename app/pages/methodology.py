import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import graphviz

# Page config
st.set_page_config(
    page_title="Methodology",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme & Styling (keeping exactly as provided)
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

/* App + header + main */
.stApp { background-color: var(--bg) !important; color: var(--fg) !important; font-family: "Segoe UI", Arial, sans-serif; }
header[data-testid="stHeader"] { background-color: var(--bg) !important; }
.main .block-container { background-color: var(--bg) !important; color: var(--fg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: var(--bg) !important; color: var(--fg) !important; }
section[data-testid="stSidebar"] a { color: var(--fg) !important; }
section[data-testid="stSidebar"] a[aria-current="page"] {
    background-color: var(--card-bg) !important;
    border-radius: 8px !important;
    color: var(--accent2) !important;
}
section[data-testid="stSidebar"] a:hover {
    background-color: var(--card-bg) !important;
    color: var(--accent2) !important;
    border-radius: 8px !important;
}
/* Prevent faded labels */
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] [role="listitem"],
section[data-testid="stSidebar"] * {
    color: var(--fg) !important;
    opacity: 1 !important;
}

/* Headings */
h1, h2, h3, h4 { color: var(--fg) !important; font-weight: 700; }

/* Cards */
.card {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30);
    margin-bottom: 1.5rem;
}

/* Buttons (teal, consistent with main.py) */
.stButton button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[kind] {
    background: var(--accent2) !important;
    background-color: var(--accent2) !important;
    color: black !important;
    font-weight: 600 !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton button:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[kind]:hover {
    background: #04b184 !important;
    background-color: #04b184 !important;
    color: black !important;
}

/* Custom additions for better visuals */
.metric-card {
    background: linear-gradient(135deg, #1e2027 0%, #252832 100%);
    border-left: 3px solid var(--accent2);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title(" Methodology")

# --- 1. DATA PIPELINE SECTION ---
st.markdown("## Data Pipeline")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("""
We merge IESO Realtime Totals CSV containing the Hourly Ontario Energy Price (HOEP) with IESO Demand reports 
containing Ontario Demand. This combined dataset is then merged with Environment Canada historical weather data 
for Toronto, which includes temperature (°C), humidity (%), and wind speed (km/h). The final dataset contains 
5 raw variables: HOEP, Ontario Demand, temperature, humidity, and wind speed, spanning 2014-2024 at hourly granularity.
""")

# Data sources table
data_sources = pd.DataFrame({
    'Source': ['IESO Realtime Totals', 'IESO Demand Reports', 'Environment Canada'],
    'Variable': ['HOEP (Target)', 'Ontario Demand', 'Temperature, Humidity, Wind Speed'],
    'Frequency': ['Hourly', 'Hourly', 'Hourly'],
    'Period': ['2014-2024', '2014-2024', '2014-2024'],
    'Location': ['Ontario', 'Ontario', 'Toronto']
})

fig = go.Figure(data=[go.Table(
    header=dict(
        values=list(data_sources.columns),
        fill_color='#06D6A0',
        font=dict(color='black', size=13),
        align='left',
        height=35
    ),
    cells=dict(
        values=[data_sources[col] for col in data_sources.columns],
        fill_color='#1e2027',
        font=dict(color='#f5f5f5', size=12),
        align='left',
        height=30
    )
)])
fig.update_layout(
    height=250,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f5f5f5')
)
st.plotly_chart(fig, use_container_width=True)

# Dataset split info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Period", "2014-2022")
with col2:
    st.metric("Validation", "2023")
with col3:
    st.metric("Test Set", "2024")

st.caption("*Deployment model retrained on all available data through April 30, 2025*")

st.markdown("</div>", unsafe_allow_html=True)

# --- 2. FEATURE ENGINEERING SECTION ---
st.markdown("## Feature Engineering")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("""
We engineer 31 features from the 5 raw variables to predict HOEP 2 hours ahead. We use the most recent 
available data: current HOEP, demand, temperature, humidity, and wind speed. We also include 1-hour and 
22-hour historical values (which correspond to 3 hours and 24 hours before the prediction target). 
Rolling averages are calculated using 3-hour and 23-hour windows from current data. Time features 
(hour sin/cos, day of year sin/cos, weekend flag) are computed for the prediction target time (T+2). 
This setup allows us to predict 9:00 HOEP using data available at 7:00.
""")

# Feature breakdown table
feature_breakdown = pd.DataFrame({
    'Category': ['Time Features', 'Current Values', '1-Hour Lags', '22-Hour Lags', 'Rolling Averages (3h)', 'Rolling Averages (23h)'],
    'Count': [6, 5, 5, 5, 5, 5],
    'Features': [
        'hour_sin, hour_cos, day_sin, day_cos, is_weekend, day_of_year',
        'HOEP, demand, temp, humidity, wind_speed',
        'All 5 variables lagged by 1 hour',
        'All 5 variables lagged by 22 hours', 
        'Moving averages of all 5 variables',
        'Moving averages of all 5 variables'
    ]
})

fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Feature Category', 'Count', 'Description'],
        fill_color='#FFD166',
        font=dict(color='black', size=13),
        align='left',
        height=35
    ),
    cells=dict(
        values=[feature_breakdown[col] for col in feature_breakdown.columns],
        fill_color='#1e2027',
        font=dict(color='#f5f5f5', size=12),
        align='left',
        height=30
    )
)])
fig.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f5f5f5')
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MODEL ARCHITECTURE SECTION ---
st.markdown("## Model Architecture")

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Quantile Neural Networks")
    st.markdown("""
    We train three separate neural networks using TensorFlow with quantile regression. Each network 
    predicts a different percentile of the price distribution:
    
    - **Q10 Model**: 10th percentile (lower bound)
    - **Q50 Model**: 50th percentile (median forecast)  
    - **Q90 Model**: 90th percentile (upper bound)
    
    The 80% prediction interval (Q10 to Q90) captures uncertainty in electricity prices, which is 
    crucial for anticipating price spikes that can occur due to sudden demand changes or supply 
    constraints. This helps market participants prepare for both typical prices (Q50) and extreme scenarios.
    """)

with col2:
    st.markdown("### Network Architecture")
    st.code("""
    Input Layer (31 features)
        ↓
    Dense(128) + LeakyReLU(α=0.01)
        ↓
    Dropout(0.2)
        ↓
    Dense(64) + LeakyReLU(α=0.01)
        ↓
    Dropout(0.2)
        ↓
    Dense(32) + LeakyReLU(α=0.01)
        ↓
    Dense(1, linear)
    """, language="text")
    

st.markdown("</div>", unsafe_allow_html=True)

# --- 4. PERFORMANCE SECTION ---
st.markdown("## Model Performance")

st.markdown("<div class='card'>", unsafe_allow_html=True)

# RMSE Comparison Chart
models_df = pd.DataFrame({
    'Model': ['Quantile NN (Q50)', 'IESO Hour-2 Predispatch'],
    'RMSE (CAD/MWh)': [24.17, 29.67]
})

fig = go.Figure()
fig.add_trace(go.Bar(
    x=models_df['Model'], 
    y=models_df['RMSE (CAD/MWh)'],
    marker_color=['#06D6A0', '#FFD166'],
    text=models_df['RMSE (CAD/MWh)'],
    textposition='outside',
    texttemplate='%{text:.2f}',
    textfont=dict(size=16)
))

# Add improvement annotation
fig.add_annotation(
    x=0.5, y=27,
    text="18.5% Improvement",
    showarrow=True,
    arrowhead=2,
    arrowcolor="#06D6A0",
    ax=0,
    ay=-50,
    font=dict(color='#06D6A0', size=16)
)

fig.update_layout(
    title="RMSE Comparison (2024 Test Set)",
    title_font_size=18,
    yaxis_title="RMSE (CAD/MWh)",
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#121418',
    font=dict(color='#f5f5f5', size=14),
    yaxis=dict(range=[0, 35])
)
fig.update_xaxes(gridcolor='#2a2a2a')
fig.update_yaxes(gridcolor='#2a2a2a')
st.plotly_chart(fig, use_container_width=True)

# Key metrics below chart
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Median RMSE", "24.17 CAD/MWh")
with col2:
    st.metric("Baseline RMSE", "29.67 CAD/MWh")
with col3:
    st.metric("Q10 Coverage", "6.5%", "Target: 10%")
with col4:
    st.metric("Q90 Coverage", "85.3%", "Target: 90%")

st.markdown("</div>", unsafe_allow_html=True)

# --- 5. DEPLOYMENT SECTION ---
st.markdown("## Deployment")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### Understanding HOEP and Timing Constraints")

st.markdown("""
The Hourly Ontario Energy Price (HOEP) is calculated as the average of 12 Market Clearing Prices (MCPs) 
collected every 5 minutes throughout each hour. MCPs are published at :00, :05, :10, :15, :20, :25, :30, 
:35, :40, :45, :50, and :55 of each hour. To match the training data structure, we need all 12 MCPs to 
calculate the true HOEP average. Therefore, predictions must wait until the 55th minute of each hour.

At 7:55, we have the complete HOEP for hour 7 and can predict for hour 9. The Manual Prediction tab 
allows forecasts at any time but may be less accurate if triggered before :55, as not all MCPs would 
be available for the current hour's average.
""")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### Pipeline Execution Flow")

# Z-pattern flowchart
graph = graphviz.Digraph()
graph.attr(bgcolor='transparent', fontcolor='white', rankdir='LR', nodesep='0.8', ranksep='1.2')
graph.attr('node', shape='box', style='filled,rounded', fontcolor='black', fontsize='12', width='2.5', height='0.8')
graph.attr('edge', color='#f5f5f5', penwidth='2')

# Top row (left to right)
graph.node('trigger', '7:55\nGoogle Cloud Function\nTriggers', fillcolor='#06D6A0')
graph.node('fetch', '7:55\nFetch Data\nIESO + OpenMeteo', fillcolor='#FFD166')
graph.node('buffer', '7:55-7:56\nUpdate Buffer\n24h Rolling Window', fillcolor='#FFD166')

# Middle row (right side)
graph.node('features', '7:56\nFeature Engineering\n31 Features Created', fillcolor='#EF476F')

# Bottom row (right to left)
graph.node('predict', '7:56\nGenerate Predictions\nQ10, Q50, Q90', fillcolor='#EF476F')
graph.node('store', '7:56\nStore Results\nUpdate CSV', fillcolor='#1e90ff')
graph.node('render', '8:00\nStreamlit Updates\nDisplay Forecasts', fillcolor='#06D6A0')

# Connect in Z pattern
graph.edge('trigger', 'fetch')
graph.edge('fetch', 'buffer')
graph.edge('buffer', 'features')
graph.edge('features', 'predict')
graph.edge('predict', 'store')
graph.edge('store', 'render')

st.graphviz_chart(graph)

st.markdown("""
**Pipeline Details:**
- Google Cloud Function executes hourly at :55 via scheduled trigger
- Data fetching includes web scraping IESO real-time data and OpenMeteo API calls
- CSV buffer maintains rolling 24-hour window for feature engineering
- Three quantile models generate predictions for hour T+2
- Render hosting automatically updates the Streamlit UI at the top of each hour
""")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)


