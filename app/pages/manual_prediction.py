import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import pytz


from cloud_entry.src.live_engineering import load_scaler, load_buffer, calculate_features, process_new_data
from cloud_entry.src.live_fetch import fetch_live_features_only
from cloud_entry.src.quantile_model import quantile_loss, load_quantile_models


# Page Configuration

st.set_page_config(page_title="Manual Prediction", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")


# Theme & Styling (visual-only)

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
.stApp { background-color: var(--bg); color: var(--fg); font-family: "Segoe UI", Arial, sans-serif; }
h1, h2, h3, h4 { color: var(--fg); font-weight: 700; }

.section-title { font-size: 2rem; font-weight: 800; margin: 0 0 0.75rem 0; }
.muted { color: var(--muted); }

.card {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30);
}

.hr { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin: 1.25rem 0; border: 0; }

.stButton > button {
    background: linear-gradient(90deg, #FFD166, #EF476F);
    color: #111;
    border: none;
    border-radius: 10px;
    padding: 0.60rem 1.25rem;
    font-weight: 700;
    transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(239,71,111,0.25); }

.big-number { font-size: 3.6rem; font-weight: 800; color: var(--accent1); line-height: 1.05; margin: 0.25rem 0; }
.band-label { font-size: 1.05rem; font-weight: 700; margin-top: 0.5rem; }
.band-values { font-size: 1.3rem; font-weight: 700; }
.q10 { color: var(--accent2); }
.q90 { color: var(--accent3); }

.metric-card {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border-left: 4px solid rgba(255,255,255,0.06);
}
.metric-label { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.15rem; }
.metric-value { font-size: 1.35rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Manual Forecast Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#a0a0a0; font-size:0.9rem;'>"
    "⚠️ Manual predictions may be less accurate than automatic updates due to data finalization delays."
    "</p>",
    unsafe_allow_html=True
)


# Section 1: Forecast Target (same logic)

with st.container():

    toronto_tz = pytz.timezone('America/Toronto')
    timestamp = datetime.now(toronto_tz).strftime("%Y-%m-%d %H:%M:%S")
    beginning_range = pd.Timestamp(timestamp).ceil('h') + timedelta(hours=1)
    end_range = beginning_range + timedelta(hours=1) - timedelta(seconds=1)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Forecast Target")
    st.markdown(
        f"<div class='muted'>This run will generate a forecast for:</div>"
        f"<div style='font-size:1.2rem; font-weight:700; margin-top:0.25rem;'>"
        f"{beginning_range.strftime('%Y-%m-%d %H:%M')} – {end_range.strftime('%H:%M')} EST</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# Section 2: Execute (same logic)

with st.container():
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Run Manual Prediction")

    if st.button("Predict Now"):
        with st.spinner("Fetching live data and generating prediction..."):
            data_path = 'cloud_entry/data/hoep_buffer.csv'
            df = load_buffer(data_path).tail(23)

            live_feat = fetch_live_features_only()
            if live_feat is None:
                st.error("Failed to fetch live data. Try again later.")
                st.stop()

            df = pd.concat([df, pd.DataFrame([live_feat])], ignore_index=True)

            features_dict = calculate_features(df)
            scaled_features = process_new_data(features_dict)

            models = load_quantile_models()
            predictions = {
                'q10': models['q10'].predict(scaled_features, verbose=0)[0][0],
                'q50': models['q50'].predict(scaled_features, verbose=0)[0][0],
                'q90': models['q90'].predict(scaled_features, verbose=0)[0][0]
            }

            st.success("Prediction complete.")
    st.markdown("</div>", unsafe_allow_html=True)


# Section 3: Results

with st.container():
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("#### Prediction Results")

    if 'predictions' in locals():
        st.markdown(
            f"<div class='muted'>Forecast target</div>"
            f"<div style='font-size:1.05rem; font-weight:700; margin-top:0.1rem;'>"
            f"{beginning_range.strftime('%Y-%m-%d %H:%M')} – {end_range.strftime('%H:%M')} EST</div>",
            unsafe_allow_html=True
        )

        left, right = st.columns([1.4, 1])

        # Left card: median + band
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-number'>${predictions['q50']:.2f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='muted'>Median forecast (CAD/MWh)</div>", unsafe_allow_html=True)

            st.markdown("<div class='band-label'>80% Confidence Band</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='band-values'><span class='q10'>${predictions['q10']:.2f}</span> – "
                f"<span class='q90'>${predictions['q90']:.2f}</span></div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Right card: compact metrics
        with right:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Q10 (Low)</div>"
                    f"<div class='metric-value' style='color:var(--accent2);'>${predictions['q10']:.2f}</div>"
                    f"</div>", unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Q50 (Median)</div>"
                    f"<div class='metric-value' style='color:var(--accent1);'>${predictions['q50']:.2f}</div>"
                    f"</div>", unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Q90 (High)</div>"
                    f"<div class='metric-value' style='color:var(--accent3);'>${predictions['q90']:.2f}</div>"
                    f"</div>", unsafe_allow_html=True
                )
    else:
        st.info("Run the prediction to see results.")
