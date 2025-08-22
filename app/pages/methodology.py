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
.timeline-step {
    display: flex;
    align-items: center;
    margin: 1rem 0;
}
.timeline-icon {
    width: 40px;
    height: 40px;
    background: var(--accent2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    font-weight: bold;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("⚙️ Methodology")
st.markdown("**Building a Production-Grade HOEP Forecasting System with Uncertainty Quantification**")

# --- 1. DATA PIPELINE SECTION ---
st.markdown("## Data Pipeline")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Sources & Integration")
    
    # Create data sources table based on actual information provided
    data_sources = pd.DataFrame({
        'Source': ['IESO Reports', 'IESO Reports', 'Environment Canada'],
        'Variable': ['HOEP (Target)', 'Ontario Demand', 'Weather (Temp, Humidity, Wind)'],
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
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Dataset Split")
    st.metric("Training", "2014-2022", "")
    st.metric("Validation", "2023", "")
    st.metric("Test", "2024", "")
    st.caption("*Deployment model retrained on all data through April 30, 2025*")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 2. FEATURE ENGINEERING VISUALIZATION ---
st.markdown("## Feature Engineering")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Temporal Feature Encoding")
    
    # Show cyclic encoding visualization
    hours = np.arange(0, 24)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=hour_sin, mode='lines', name='sin(hour)', 
                             line=dict(color='#FFD166', width=2)))
    fig.add_trace(go.Scatter(x=hours, y=hour_cos, mode='lines', name='cos(hour)', 
                             line=dict(color='#06D6A0', width=2)))
    fig.update_layout(
        title="Cyclic Hour Encoding (No Discontinuity at Midnight)",
        xaxis_title="Hour of Day",
        yaxis_title="Encoded Value",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#121418',
        font=dict(color='#f5f5f5'),
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    fig.update_xaxes(gridcolor='#2a2a2a')
    fig.update_yaxes(gridcolor='#2a2a2a')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Feature Categories (31 Total)")
    
    # Based on actual feature list provided
    feature_breakdown = """
    **Time Features (6):**
    - hour_sin, hour_cos
    - is_weekend
    - day_of_year, doy_sin, doy_cos
    
    **Lagged Features (15):**
    - HOEP: lag_2, lag_3, lag_24
    - Demand: lag_2, lag_3, lag_24
    - Temp: lag_2, lag_3, lag_24
    - Humidity: lag_2, lag_3, lag_24
    - Wind: lag_2, lag_3, lag_24
    
    **Rolling Averages (10):**
    - 3h & 23h windows for:
    - HOEP, Demand, Temp, Humidity, Wind
    """
    st.markdown(feature_breakdown)
    st.markdown("</div>", unsafe_allow_html=True)

# Lag visualization
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 2-Hour Ahead Forecasting Strategy")
st.info("**Key Design Choice:** Model predicts current HOEP using only data available 2 hours prior, simulating real-time 2-hour ahead forecasting without data leakage.")

# Simple diagram showing the forecasting horizon
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    ```
    Time:     t-24h    t-3h    t-2h    NOW    t+2h
               ↓        ↓       ↓              ↑
              [===== Features Used =====]    Target
                                          (Predict this)
    ```
    """)
st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MODEL ARCHITECTURE ---
st.markdown("## Neural Network Architecture")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Three Quantile Models")
    st.info("**Q10 Model** → Lower Bound (10th percentile)")
    st.success("**Q50 Model** → Median Forecast")
    st.warning("**Q90 Model** → Upper Bound (90th percentile)")
    st.caption("Each model trained separately with custom quantile loss")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
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
Dense(1, linear) → Quantile Output
    """, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Training Configuration")
    st.markdown("""
    - **Loss:** Pinball (Quantile)
    - **Optimizer:** Adam (lr=0.001)
    - **Batch Size:** 32
    - **Early Stopping:** patience=5
    - **Validation Split:** 10%
    - **Epochs:** 100 (w/ early stop)
    - **Scaling:** StandardScaler
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- 4. PERFORMANCE METRICS ---
st.markdown("## 📈 Model Performance")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### RMSE Comparison (2024 Test Set)")
    
    # Only using actual provided metrics
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
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Key Metrics")
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Median RMSE", "24.17", "CAD/MWh")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("vs IESO Baseline", "-18.5%", "RMSE Reduction")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("80% Interval", "Q10-Q90", "Uncertainty Band")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Coverage metrics
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Quantile Coverage Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Q10 Coverage")
    coverage_q10 = 0.065
    target_q10 = 0.1
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = coverage_q10,
        title = {'text': "Actual vs Target (0.1)"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 0.2]},
            'bar': {'color': "#EF476F"},
            'steps': [
                {'range': [0, target_q10], 'color': "#1e2027"},
                {'range': [target_q10, 0.2], 'color': "#252832"}],
            'threshold': {
                'line': {'color': "#06D6A0", 'width': 4},
                'thickness': 0.75,
                'value': target_q10}}
    ))
    fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f5f5f5'))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Slightly underestimates lower bound")

with col2:
    st.markdown("#### Q50 Coverage")
    st.info("Median forecast optimized for RMSE, not coverage")

with col3:
    st.markdown("#### Q90 Coverage")
    coverage_q90 = 0.853
    target_q90 = 0.9
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = coverage_q90,
        title = {'text': "Actual vs Target (0.9)"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 1.0]},
            'bar': {'color': "#06D6A0"},
            'steps': [
                {'range': [0, target_q90], 'color': "#1e2027"},
                {'range': [target_q90, 1.0], 'color': "#252832"}],
            'threshold': {
                'line': {'color': "#FFD166", 'width': 4},
                'thickness': 0.75,
                'value': target_q90}}
    ))
    fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f5f5f5'))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Good calibration near target")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("## Production Deployment")

# Timeline visualization
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Hourly Pipeline Execution Timeline")

timeline_data = pd.DataFrame({
    'Time': ['7:55', '7:55-7:56', '7:56', '7:56', '8:00'],
    'Action': ['Data Collection', 'Feature Engineering', 'Generate Forecast', 'Store Results', 'UI Update'],
    'Details': [
        'Fetch 12 MCPs from IESO + OpenMeteo weather',
        'Create lags & rolling features from 24h buffer',
        'Run 3 quantile models for hour 9:00-9:59',
        'Update CSV buffer (keep last 24 hours)',
        'Render refreshes charts with new predictions'
    ]
})

fig = go.Figure()

for i, row in timeline_data.iterrows():
    fig.add_trace(go.Scatter(
        x=[i], y=[1],
        mode='markers+text',
        marker=dict(size=30, color=['#06D6A0', '#FFD166', '#EF476F', '#1e90ff', '#06D6A0'][i]),
        text=row['Time'],
        textposition='top center',
        name=row['Action'],
        hovertemplate=f"<b>{row['Action']}</b><br>{row['Details']}<extra></extra>"
    ))

fig.update_layout(
    height=200,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#121418',
    font=dict(color='#f5f5f5'),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.5, 4.5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0.5, 1.5]
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

# Add connecting line
fig.add_shape(
    type="line",
    x0=0, y0=1, x1=4, y1=1,
    line=dict(color="#2a2a2a", width=2)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# System Architecture
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### System Architecture")
    
    graph = graphviz.Digraph()
    graph.attr(bgcolor='transparent', fontcolor='white', rankdir='LR')
    graph.attr('node', shape='box', style='filled,rounded', fontcolor='black')
    graph.attr('edge', color='#f5f5f5')
    
    # Data sources
    graph.node('IESO', 'IESO\nRealtime Data', fillcolor='#06D6A0')
    graph.node('Weather', 'OpenMeteo\nWeather API', fillcolor='#06D6A0')
    
    # Processing
    graph.node('GCF', 'Google Cloud\nFunction', fillcolor='#FFD166')
    graph.node('Buffer', 'CSV Buffer\n(24h rolling)', fillcolor='#FFD166')
    
    # Models
    graph.node('Models', 'Quantile NNs\n(Q10, Q50, Q90)', fillcolor='#EF476F')
    
    # Output
    graph.node('Render', 'Streamlit App\n(Render.com)', fillcolor='#1e90ff')
    
    # Connections
    graph.edge('IESO', 'GCF', label=':55')
    graph.edge('Weather', 'GCF', label=':55')
    graph.edge('GCF', 'Buffer')
    graph.edge('Buffer', 'Models')
    graph.edge('Models', 'Render', label=':00')
    
    st.graphviz_chart(graph.use_engine('dot'))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Tech Stack")
    st.markdown("""
    **Data Sources:**
    - IESO Realtime Totals CSV
    - IESO Zonal Prices XML
    - OpenMeteo Weather API
    
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
    - Plotly
    - Render.com hosting
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Important note about manual predictions
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.warning("**⚠️ Manual Prediction Note:** Predictions triggered before the 55th minute of each hour may be less accurate as not all 12 Market Clearing Prices (MCPs) have been finalized. The automated pipeline waits until :55 to ensure all MCPs are available for calculating the true HOEP average.")
st.markdown("</div>", unsafe_allow_html=True)

# Final summary
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Key Innovation")
st.success("""
**Production-Ready 2-Hour Ahead Forecasting:** This system delivers median predictions with 18.5% better accuracy than IESO's baseline, 
plus uncertainty quantification through prediction intervals — all updating hourly in real-time with proper data handling to match training conditions.
""")
st.markdown("</div>", unsafe_allow_html=True)
