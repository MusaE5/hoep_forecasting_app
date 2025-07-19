import pandas as pd
import numpy as np
import os
from datetime import timedelta
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.quantile_model import quantile_loss, load_quantile_models


# --- Loaders ---
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


# --- Feature Engineering ---
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
    target_timestamp = timestamp.dt.ceil('H') + pd.Timedelta(hours=1)
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


# --- Execution ---
if __name__ == "__main__":
    buffer_file = "data/hoep_buffer.csv"
    df = load_buffer(buffer_file)
    features_dict = calculate_features(df)
    scaled_features = process_new_data(features_dict)

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
    print(predictions)
