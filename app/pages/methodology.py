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

# Theme & Styling 
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


st.title("⚙️ Methodology")


st.markdown("## Data Pipeline")

st.markdown("<div class='card'>", unsafe_allow_html=True)

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
        font=dict(color='black', size=12),
        align='left'
    ),
    cells=dict(
        values=[data_sources[col] for col in data_sources.columns],
        fill_color='#1e2027',
        font=dict(color='#f5f5f5'),
        align='left'
    )
)])
fig.update_layout(
    height=200,
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

st.caption("*Deployment model retrained on all available data through April 30, 2025 for maximum accuracy*")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("## Feature Engineering")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("""
The model uses **31 engineered features** designed to capture temporal patterns and avoid data leakage. 
A critical design choice: the model predicts the current hour's HOEP (time T) using only information 
available 2 hours prior (T-2), simulating real-world forecasting constraints.
""")

# Feature breakdown
feature_breakdown = pd.DataFrame({
    'Category': ['Time Features', 'Lagged HOEP', 'Lagged Demand', 'Lagged Weather', 'Rolling Averages'],
    'Count': [6, 3, 3, 9, 10],
    'Description': [
        'Hour (sin/cos), Day of year (sin/cos), Weekend flag',
        'Previous values at 2h, 3h, 24h',
        'Previous values at 2h, 3h, 24h',
        'Temp, humidity, wind at 2h, 3h, 24h',
        '3h and 23h windows for all variables'
    ]
})

fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Feature Category', 'Count', 'Description'],
        fill_color='#FFD166',
        font=dict(color='black', size=12),
        align='left'
    ),
    cells=dict(
        values=[feature_breakdown[col] for col in feature_breakdown.columns],
        fill_color='#1e2027',
        font=dict(color='#f5f5f5'),
        align='left'
    )
)])
fig.update_layout(
    height=250,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f5f5f5')
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("## Model Architecture")

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Quantile Neural Networks")
    st.markdown("""
    Instead of predicting a single point estimate, the system uses **quantile regression** 
    to predict different percentiles of the price distribution. This provides both a best 
    estimate and uncertainty bounds.
    
    **Three separate models trained:**
    - **Q10 Model**: 10th percentile (lower bound)
    - **Q50 Model**: 50th percentile (median forecast)  
    - **Q90 Model**: 90th percentile (upper bound)
    
    Together, these create an 80% prediction interval that quantifies forecast uncertainty.
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
    
    st.markdown("""
    **Training Details:**
    - Loss: Pinball (quantile-specific)
    - Optimizer: Adam (lr=0.001)
    - Early stopping (patience=5)
    - StandardScaler normalization
    """)

st.markdown("</div>", unsafe_allow_html=True)


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
    texttemplate='%{text:.2f}'
))

# Add improvement annotation
fig.add_annotation(
    x=0.5, y=27,
    text="18.5% Improvement",
    showarrow=True,
    arrowhead=2,
    arrowcolor="#06D6A0",
    ax=0,
    ay=-40,
    font=dict(color='#06D6A0', size=14)
)

fig.update_layout(
    title="RMSE Comparison (2024 Test Set)",
    yaxis_title="RMSE (CAD/MWh)",
    height=400,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#121418',
    font=dict(color='#f5f5f5'),
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


st.markdown("##Deployment")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### Understanding HOEP and Timing Constraints")

st.markdown("""
The Hourly Ontario Energy Price (HOEP) is calculated as the **average of 12 Market Clearing Prices (MCPs)** 
collected every 5 minutes throughout each hour. This creates a critical timing constraint for predictions:

- MCPs are published every 5 minutes (e.g., 7:00, 7:05, 7:10... 7:55)
- To match training data, we need all 12 MCPs to calculate the true HOEP average
- Therefore, predictions must wait until the **55th minute** of each hour
- At 7:55, we have the complete HOEP for hour 7 and can predict for hour 9

The Manual Prediction tab allows forecasts at any time but may be less accurate if triggered before :55, 
as not all MCPs would be available.
""")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### Pipeline Execution Flow")

# Create a better flowchart for pipeline
graph = graphviz.Digraph()
graph.attr(bgcolor='transparent', fontcolor='white')
graph.attr('node', shape='box', style='filled,rounded', fontcolor='black', fontsize='11')
graph.attr('edge', color='#f5f5f5')

# Define nodes with times and actions
graph.node('trigger', '7:55\nGoogle Cloud Function\nTriggers', fillcolor='#06D6A0')
graph.node('fetch', '7:55\nFetch Data\n• IESO MCPs & Demand\n• OpenMeteo Weather', fillcolor='#FFD166')
graph.node('buffer', '7:55-7:56\nUpdate Buffer\n• Append new data\n• Keep last 24 hours', fillcolor='#FFD166')
graph.node('features', '7:56\nFeature Engineering\n• Create lags (2h, 3h, 24h)\n• Calculate rolling averages', fillcolor='#EF476F')
graph.node('predict', '7:56\nGenerate Predictions\n• Q10, Q50, Q90 models\n• For hour 9:00-9:59', fillcolor='#EF476F')
graph.node('store', '7:56\nStore Results\n• Save predictions\n• Update CSV', fillcolor='#1e90ff')
graph.node('render', '8:00\nStreamlit Updates\n• Render refreshes\n• Display new forecasts', fillcolor='#06D6A0')

# Connect nodes
graph.edge('trigger', 'fetch')
graph.edge('fetch', 'buffer')
graph.edge('buffer', 'features')
graph.edge('features', 'predict')
graph.edge('predict', 'store')
graph.edge('store', 'render')

st.graphviz_chart(graph)

st.markdown("""
**Key Points:**
- **Google Cloud Function** executes hourly at :55 via scheduled trigger
- **Data fetching** includes web scraping IESO real-time data and OpenMeteo API calls
- **CSV buffer** maintains rolling 24-hour window for feature engineering
- **Render hosting** automatically updates the Streamlit UI at the top of each hour
- Entire pipeline completes in ~1 minute, ready for display at :00
""")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### System Architecture")

col1, col2 = st.columns([2, 1])

with col1:
    # System architecture diagram
    arch = graphviz.Digraph()
    arch.attr(bgcolor='transparent', fontcolor='white', rankdir='TB')
    arch.attr('node', shape='box', style='filled,rounded', fontcolor='black')
    arch.attr('edge', color='#f5f5f5')
    
    # Data layer
    with arch.subgraph(name='cluster_0') as c:
        c.attr(label='Data Sources', fontcolor='#f5f5f5', style='rounded', color='#2a2a2a')
        c.node('ieso', 'IESO\nReal-time Data', fillcolor='#06D6A0')
        c.node('weather', 'OpenMeteo\nWeather API', fillcolor='#06D6A0')
    
    # Processing layer
    with arch.subgraph(name='cluster_1') as c:
        c.attr(label='Processing', fontcolor='#f5f5f5', style='rounded', color='#2a2a2a')
        c.node('gcf', 'Google Cloud\nFunction', fillcolor='#FFD166')
        c.node('buffer', 'CSV Buffer\n(24h rolling)', fillcolor='#FFD166')
    
    # Model layer
    with arch.subgraph(name='cluster_2') as c:
        c.attr(label='ML Models', fontcolor='#f5f5f5', style='rounded', color='#2a2a2a')
        c.node('models', 'Quantile NNs\nQ10, Q50, Q90', fillcolor='#EF476F')
    
    # Frontend
    arch.node('ui', 'Streamlit App\n(Render.com)', fillcolor='#1e90ff')
    
    # Connections
    arch.edge('ieso', 'gcf')
    arch.edge('weather', 'gcf')
    arch.edge('gcf', 'buffer')
    arch.edge('buffer', 'models')
    arch.edge('models', 'ui')
    
    st.graphviz_chart(arch)

with col2:
    st.markdown("### Tech Stack")
    st.markdown("""
    **Data Collection:**
    - IESO web scraping
    - OpenMeteo API
    
    **Backend:**
    - Google Cloud Functions
    - GitHub Actions (scheduling)
    - Python 3.x
    
    **ML Framework:**
    - TensorFlow/Keras
    - Custom quantile loss
    - StandardScaler
    
    **Frontend:**
    - Streamlit
    - Plotly visualizations
    - Render.com hosting
    
    **Data Management:**
    - Rolling CSV buffer
    - 24-hour data retention
    """)

st.markdown("</div>", unsafe_allow_html=True)
