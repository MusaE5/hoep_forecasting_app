import streamlit as st
import plotly.graph_objects as go
import graphviz

# Page config
st.set_page_config(
    page_title="Methodology",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme & Styling (copied from main.py for consistency)
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
</style>
""", unsafe_allow_html=True)


# --- Content ---
st.title("Methodology")

# Data Sources
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Data Sources")
st.write("""
- **HOEP (Hourly Ontario Energy Price):** IESO Realtime Totals reports (2014–2024)  
- **Demand:** Ontario demand & market demand from IESO reports  
- **Weather:** Government of Canada reports for Toronto (temperature, humidity, wind speed)  
""")
st.markdown("</div>", unsafe_allow_html=True)

# Feature Engineering
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Feature Engineering")
st.write("""
- Time features: hour of day (sin/cos), weekend flag, day of year (sin/cos)  
- Lagged demand, weather, and HOEP (2h, 3h, 24h)  
- Rolling averages (3h, 23h)  
- Ensured no data leakage: predictions only use info available 2 hours before forecast target.  
""")
st.markdown("</div>", unsafe_allow_html=True)

# Models
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Models")
st.write("""
- Three Neural Networks trained with quantile regression loss  
- Separate models for **Q10, Q50, Q90**  
- Architecture: Dense → LeakyReLU → Dropout layers → Dense(1)  
""")
st.markdown("</div>", unsafe_allow_html=True)

# Performance
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Median RMSE", "24.17", "vs 29.67 baseline")
col2.metric("Improvement", "18.5%", "lower RMSE")
col3.metric("Q90 Coverage", "0.853", "target 0.9")

# Example bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Model (Q50)", "IESO Hour-2"],
    y=[24.17, 29.67],
    marker_color=["#06D6A0", "#FFD166"]
))
fig.update_layout(
    title="Forecasting Accuracy (RMSE, 2024 test set)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#121418",
    font=dict(color="#f5f5f5")
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Deployment Pipeline
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Deployment Pipeline")
st.write("""
- At **:55 each hour**, Google Cloud Function fetches MCPs + weather  
- Buffer keeps last 24h (for lags)  
- Model predicts **2h ahead** (7:55 → forecast for 9:00–9:59)  
- Streamlit app (Render) updates around the top of the next hour  
- Manual predictions available, but less accurate until all MCPs finalize  
""")

graph = graphviz.Digraph()
graph.attr(bgcolor="transparent", fontcolor="white")
graph.node("Data", "Live Data (IESO + Weather)", shape="box", style="filled", color="#06D6A0")
graph.node("FE", "Feature Engineering", shape="box", style="filled", color="#FFD166")
graph.node("Model", "Quantile NN Models", shape="box", style="filled", color="#EF476F")
graph.node("Forecasts", "Q10 / Q50 / Q90 Forecasts", shape="box", style="filled", color="#1e90ff")
graph.node("App", "Streamlit App", shape="box", style="filled", color="#06D6A0")

graph.edges([("Data", "FE"), ("FE", "Model"), ("Model", "Forecasts"), ("Forecasts", "App")])
st.graphviz_chart(graph, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
