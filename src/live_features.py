import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent  # Goes up two levels from src/

# Helper functions
def load_scaler():
    scaler_path = PROJECT_ROOT / "models" / "feature_scaler.pkl"
    return joblib.load(scaler_path)

def load_buffer(buffer_file):
    """Loads existing data buffer"""
    if Path(buffer_file).exists():
        return pd.read_csv(buffer_file)
    return pd.DataFrame(columns=[
        'timestamp', 'demand_MW', 
        'or10_sync_MW', 'or30_MW', 'temp_C', 
        'humidity_%', 'wind_mps', 'zonal_price'
    ])


def calculate_features(buffer_df):
    """
    Creates features PREDICTING t+1 (1-hour ahead).
    Uses data up to t-2 to forecast t+1.
    """
    # Time features (unchanged)
    hour = pd.to_datetime(buffer_df['timestamp']).hour
    time_features = {
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24)
    }
    
    # Price features (shift lags +1 vs nowcasting)
    price_col = buffer_df['zonal_price']
    price_features = {
        'HOEP_lag_2': price_col.iloc[-1],  # Most recent price (just aquired)
        'HOEP_lag_3': price_col.iloc[-2],  # Hour before
        'HOEP_lag_24': price_col.iloc[-23],  
        'HOEP_ma_3': price_col.iloc[-3:].mean(),
        'HOEP_ma_24': demand_col.iloc[-24:].mean()


    }
    
    # Demand features (same shift logic)
    demand_col = buffer_df['demand_MW']
    demand_features = {
        'Demand_lag_2': demand_col.iloc[-1],  
        'Demand_lag_3': demand_col.iloc[-2],  
        'Demand_lag_24': demand_col.iloc[-23], 
        'Demand_ma_3': demand_col.iloc[-3:].mean(),
        'Demand_ma_24': demand_col.iloc[-23:].mean()  
    }
    
    return {
        'OR 10 Min Sync': ['or10_sync_MW'],
        'OR 30 Min': ['or30_MW'],
        'Ontario Demand': ['demand_MW'],
        'temp': ['temp_C'],
        'humidity': ['humidity_%'],
        'wind_speed': ['wind_mps'],
        **time_features,
        **price_features,
        **demand_features
    }

def process_new_data(df):
    
    if len(buffer_df) < 24:
        raise ValueError("Need at least 24hours of historical data")
    
    # Calculate features (now matches training)
    features = calculate_features(df)
    
    # Ensure features are in the SAME ORDER as training
    training_feature_order = [
        "hour_sin", "hour_cos", "is_weekend",
    "day_of_year", "doy_sin", "doy_cos",
    "demand_lag_2", "demand_lag_3", "demand_lag_24",
    "temp_lag_2", "temp_lag_3", "temp_lag_24",
    "humidity_lag_2", "humidity_lag_3", "humidity_lag_24",
    "wind_speed_lag_2", "wind_speed_lag_3", "wind_speed_lag_24",
    "HOEP_lag_2", "HOEP_lag_3", "HOEP_lag_24",
    "demand_ma_3", "demand_ma_24",
    "temp_ma_3", "temp_ma_24",
    "humidity_ma_3", "humidity_ma_24",
    "wind_speed_ma_3", "wind_speed_ma_24",
    "HOEP_ma_3", "HOEP_ma_24", "OR_30_Min_lag_2",
    "OR_30_Min_lag_3",
    "OR_30_Min_lag_24",
    "OR_10_Min_sync_lag2",
    "OR_10_Min_sync_lag3",
    "OR_10_Min_sync_lag24",
    "OR_10_Min_non-sync_lag2",
    "OR_10_Min_non-sync_lag3",
    "OR_10_Min_non-sync_lag24",
    ]
    
    # Reorder features to match training
    features_ordered = {k: features[k] for k in training_feature_order}
    
    # Scale and return
    scaler = load_scaler()
    scaled_features = scaler.transform(pd.DataFrame([features_ordered]))
    
    return scaled_features

# Example usage with live_features.py
if __name__ == "__main__":
    from live_fetch import fetch_realtime_totals, fetch_current_weather, get_ontario_zonal_average
    from tensorflow.keras.models import load_model

    buffer_file = "data/hoep_buffer.csv"

    df = load_buffer(buffer_file)
    
    print(df.head())