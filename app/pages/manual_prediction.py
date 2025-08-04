import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import sys
import os

# Add project root (2 levels up from /pages/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.live_engineering import load_scaler, load_buffer, calculate_features, process_new_data
from src.live_fetch import fetch_live_features_only
from src.quantile_model import quantile_loss, load_quantile_models

# ──────────────────────────────────────
# 🔧 Page Configuration
# ──────────────────────────────────────
st.set_page_config(page_title="Manual Prediction", layout="wide")
st.title("🧪 Manual Forecast Generator")

# ──────────────────────────────────────
# ⏱️ Section 1: Forecast Target Time
# ──────────────────────────────────────
with st.container():
    st.markdown("#### ⏱️ Forecast Target")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    beginning_range = pd.Timestamp(timestamp).ceil('h') + timedelta(hours=1)
    end_range = beginning_range + timedelta(hours=1) - timedelta(seconds=1)
    st.markdown(f"Generate forecast for hour **{beginning_range} - {end_range}**")



# ──────────────────────────────────────
# 🔘 Section 2: Button & Execution
# ──────────────────────────────────────
with st.container():
    st.markdown("#### 🔍 Run Manual Prediction")

    if st.button("🔮 Predict Now"):
        with st.spinner("Fetching live data and generating prediction..."):
            data_path = 'data/hoep_buffer.csv'
            df = load_buffer(data_path).tail(23)

            live_feat = fetch_live_features_only()
            if live_feat is None:
                st.error("❌ Failed to fetch live data. Try again later.")
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

            st.success("✅ Prediction complete!")
# ──────────────────────────────────────
# 📊 Section 3: Display Results (Optional)
# ──────────────────────────────────────
with st.container():
    st.markdown("#### 📉 Prediction Results")

    if 'predictions' in locals():
        st.markdown(f"📅 Forecast Target: **{beginning_range.strftime('%Y-%m-%d %H:%M')} – {end_range.strftime('%H:%M')}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("🔻 Quantile 10 (Low)", f"{predictions['q10']:.2f} $/MWh")
        col2.metric("🎯 Quantile 50 (Median)", f"{predictions['q50']:.2f} $/MWh")
        col3.metric("🔺 Quantile 90 (High)", f"{predictions['q90']:.2f} $/MWh")
    else:
        st.info("Run the prediction to see results.")
