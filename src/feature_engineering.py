import numpy as np
import pandas as pd

def create_features(df, lags=[1, 2, 24], roll_windows=[3, 24]):
    """
    Takes the merged DataFrame with raw columns:
      - timestamp, HOEP,
      - Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min,
      - Ontario Demand, Market Demand, temp, humidity, wind_speed
    Returns a DataFrame with engineered, non-leaky features and no raw measured columns.
    """
    df = df.copy()
    
    # --- Time features (safe, no leakage) ---
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    # Day-of-year cyclical
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # --- Lagged measured features (no peek) ---
    for k in lags:
        df[f'demand_lag_{k}']       = df['Ontario Demand'].shift(k)
        df[f'temp_lag_{k}']         = df['temp'].shift(k)
        df[f'humidity_lag_{k}']     = df['humidity'].shift(k)
        df[f'wind_speed_lag_{k}']   = df['wind_speed'].shift(k)
        df[f'HOEP_lag_{k}']         = df['HOEP'].shift(k)
    
    # --- Rolling means on lagged measured & HOEP features ---
    for win in roll_windows:
        df[f'demand_ma_{win}']     = df['demand_lag_1'].rolling(win).mean()
        df[f'temp_ma_{win}']       = df['temp_lag_1'].rolling(win).mean()
        df[f'humidity_ma_{win}']   = df['humidity_lag_1'].rolling(win).mean()
        df[f'wind_speed_ma_{win}'] = df['wind_speed_lag_1'].rolling(win).mean()
        df[f'HOEP_ma_{win}']       = df['HOEP_lag_1'].rolling(win).mean()
    
    # --- Keep raw forecast features as-is (no shift) ---
    #    Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min
    # Raw measured columns must be removed to avoid leakage
    df = df.drop(columns=[
        'Ontario Demand', 'temp', 'humidity', 'wind_speed'
    ])
    
    # Drop rows with any NaNs introduced by shifts/rolls
    return df.dropna()

