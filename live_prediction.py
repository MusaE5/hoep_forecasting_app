import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.quantile_model import quantile_loss, load_quantile_models
from src.live_fetch import fetch_realtime_totals, fetch_current_weather, fetch_and_store, append_to_buffer, get_ontario_zonal_average


def load_scaler():
    return joblib.load("models/quantile_feature_scaler.pkl")


def load_buffer(buffer_file):
    """Loads existing data buffer"""
    if os.path.exists(buffer_file):
        return pd.read_csv(buffer_file)
    return pd.DataFrame(columns=[
        'timestamp', 'demand_MW', 
        'temp_C', 
        'humidity_%', 'wind_mps', 'zonal_price'
    ])


# Live feature engineering from buffer csv
def calculate_features(buffer_df):
    """Creates features for t+1 hour ahead forecast"""
    hour_col = buffer_df['hour']
    current_hour = hour_col.iloc[-1]
    prediction_hour = (current_hour + 2) % 24
    time_features = {
        'hour_sin': np.sin(2 * np.pi * prediction_hour / 24),
        'hour_cos': np.cos(2 * np.pi * prediction_hour / 24)
    }

    df = buffer_df.copy()
    timestamp = pd.to_datetime(df['timestamp'])
    target_timestamp = timestamp.dt.ceil('h') + pd.Timedelta(hours=1)
    df['timestamp'] = target_timestamp

    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    price_col = df['zonal_price']
    demand_col = df['demand_MW']
    wind_col = df['wind_mps'] * 3.6
    temp_col = df['temp_C']
    humid_col = df['humidity_%']

    return {
        **time_features,
        'is_weekend': df['is_weekend'].iloc[-1],
        'day_of_year': df['day_of_year'].iloc[-1],
        'doy_sin': df['doy_sin'].iloc[-1],
        'doy_cos': df['doy_cos'].iloc[-1],
        'HOEP_lag_2': price_col.iloc[-1],
        'HOEP_lag_3': price_col.iloc[-2],
        'HOEP_lag_24': price_col.iloc[-23],
        'HOEP_ma_3': price_col.iloc[-3:].mean(),
        'HOEP_ma_23': price_col.iloc[-24:].mean(),
        'demand_lag_2': demand_col.iloc[-1],
        'demand_lag_3': demand_col.iloc[-2],
        'demand_lag_24': demand_col.iloc[-23],
        'demand_ma_3': demand_col.iloc[-3:].mean(),
        'demand_ma_23': demand_col.iloc[-23:].mean(),
        'wind_speed_lag_2': wind_col.iloc[-1],
        'wind_speed_lag_3': wind_col.iloc[-2],
        'wind_speed_lag_24': wind_col.iloc[-23],
        'wind_speed_ma_3': wind_col.iloc[-3:].mean(),
        'wind_speed_ma_23': wind_col.iloc[-23:].mean(),
        'temp_lag_2': temp_col.iloc[-1],
        'temp_lag_3': temp_col.iloc[-2],
        'temp_lag_24': temp_col.iloc[-23],
        'temp_ma_3': temp_col.iloc[-3:].mean(),
        'temp_ma_23': temp_col.iloc[-23:].mean(),
        'humidity_lag_2': humid_col.iloc[-1],
        'humidity_lag_3': humid_col.iloc[-2],
        'humidity_lag_24': humid_col.iloc[-23],
        'humidity_ma_3': humid_col.iloc[-3:].mean(),
        'humidity_ma_23': humid_col.iloc[-23:].mean()
    }



def process_new_data(features_dict):
    training_feature_order = [
        "hour_sin", "hour_cos", "is_weekend", "day_of_year", "doy_sin", "doy_cos",
        "demand_lag_2", "demand_lag_3", "demand_lag_24",
        "temp_lag_2", "temp_lag_3", "temp_lag_24",
        "humidity_lag_2", "humidity_lag_3", "humidity_lag_24",
        "wind_speed_lag_2", "wind_speed_lag_3", "wind_speed_lag_24",
        "HOEP_lag_2", "HOEP_lag_3", "HOEP_lag_24",
        "demand_ma_3", "demand_ma_23",
        "temp_ma_3", "temp_ma_23",
        "humidity_ma_3", "humidity_ma_23",
        "wind_speed_ma_3", "wind_speed_ma_23",
        "HOEP_ma_3", "HOEP_ma_23"
    ]
    features_ordered = [features_dict[k] for k in training_feature_order]
    features_df = pd.DataFrame([features_ordered], columns=training_feature_order)
    scaler = load_scaler()
    return scaler.transform(features_df)



if __name__ == "__main__":

    feat = fetch_and_store()
    if feat is not None:
        actual_hoep = feat['zonal_price']

    buffer_file = "data/hoep_buffer.csv"
    df = load_buffer(buffer_file)
    features_dict = calculate_features(df)
    scaled_features = process_new_data(features_dict)
    # ───────────────────────────────────────────────
    # ✅ Sanity check: Validate model input features
    # ───────────────────────────────────────────────
    expected_num_features = 31  # Adjust if your model uses a different number
    expected_shape = (1, expected_num_features)

    print(f"📐 Scaled features shape: {scaled_features.shape}")
    print("🔍 Any NaNs?", np.isnan(scaled_features).any())
    print("🔍 Any Infs?", np.isinf(scaled_features).any())

    if scaled_features.shape != expected_shape:
        print(f"❌ Feature shape mismatch! Expected {expected_shape}, got {scaled_features.shape}")
        raise ValueError("Model input shape is incorrect.")

    # Optional: print the feature vector if debugging further
    print("🧾 Sample features row:", scaled_features[0])


    models = load_quantile_models()
    


    predictions = {
        'q10': models['q10'].predict(scaled_features, verbose=0)[0][0],
        'q50': models['q50'].predict(scaled_features, verbose=0)[0][0],
        'q90': models['q90'].predict(scaled_features, verbose=0)[0][0]
    }

    print("HOEP Predictions:")
    print(f"10th percentile: ${predictions['q10']:.2f}")
    print(f"50th percentile: ${predictions['q50']:.2f}")
    print(f"90th percentile: ${predictions['q90']:.2f}")
    
     # Step 3: Update predictions_log.csv
   
    # Step 4: Append new prediction row
    predicted_for = pd.to_datetime(df['timestamp'].iloc[-1]).ceil('h') + timedelta(hours=1)
    new_entry = {
        "predicted_for_hour": predicted_for.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_q10": predictions['q10'],
        "pred_q50": predictions['q50'],
        "pred_q90": predictions['q90'],
        "timestamp_predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actual_hoep": None
    }
  # Step 3: Update predictions_log.csv
    log_path = "data/predictions_log.csv"

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)

        if len(log_df) == 3:
            # Inject actual HOEP into the middle row
            if pd.isna(log_df.loc[1, 'actual_hoep']):
                log_df.loc[1, 'actual_hoep'] = actual_hoep
                print(f"✅ Injected actual HOEP {actual_hoep:.2f} into predicted_for_hour {log_df.loc[1, 'predicted_for_hour']}")
            else:
                print("⚠️ Middle row already has actual HOEP — skipping injection.")

            # Drop the oldest row (index 0)
            log_df = log_df.iloc[1:]

    else:
        log_df = pd.DataFrame(columns=[
            "predicted_for_hour", "pred_q10", "pred_q50", "pred_q90", "timestamp_predicted_at", "actual_hoep"
        ])
        print("📂 Created new predictions_log.csv")

    # Step 4: Append new prediction row
    predicted_for = pd.to_datetime(df['timestamp'].iloc[-1]).ceil('h') + timedelta(hours=1)
    new_entry = {
        "predicted_for_hour": predicted_for.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_q10": predictions['q10'],
        "pred_q50": predictions['q50'],
        "pred_q90": predictions['q90'],
        "timestamp_predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actual_hoep": None
    }

    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)

    numeric_cols = ['pred_q10', 'pred_q50', 'pred_q90']
    log_df[numeric_cols] = log_df[numeric_cols].round(2)

    log_df.to_csv(log_path, index=False)
    print("✅ Appended new prediction row.")
   

    # ────────────────────────────────────────────────
    # 🔁 Maintain 24-row rolling chart buffer (no actual HOEP here)
    # ────────────────────────────────────────────────


    chart_buffer_path = "data/chart_buffer.csv"
    chart_cols = ["predicted_for_hour", "pred_q10", "pred_q50", "pred_q90", "timestamp_predicted_at", "actual_hoep"]

    # Try to load chart buffer or initialize
    if os.path.exists(chart_buffer_path):
        chart_df = pd.read_csv(chart_buffer_path)
    else:
        chart_df = pd.DataFrame(columns=chart_cols)

    
    chart_entry = {k: new_entry[k] for k in chart_cols if k in new_entry}

    # Round float predictions to 2 decimals before saving
    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        if col in chart_entry and pd.notna(chart_entry[col]):
            chart_entry[col] = round(float(chart_entry[col]), 2)

    # Append
    chart_df = pd.concat([chart_df, pd.DataFrame([chart_entry])], ignore_index=True)

    # Keep only the latest 26 entries
    chart_df = chart_df.tail(26).reset_index(drop=True)

    # Inject actual_hoep into the 3rd-last row
    if len(chart_df) >= 3:
        chart_df.loc[-3 % len(chart_df), "actual_hoep"] = round(actual_hoep, 2)

    # Save
    chart_df.to_csv(chart_buffer_path, index=False)
    print("📈 Updated chart_buffer.csv with rolling 24 predictions.")