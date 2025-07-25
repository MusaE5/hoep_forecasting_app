import numpy as np
import pandas as pd

def create_features(df, lags=[2,3, 24], roll_windows=[3, 23]):
    """
    Takes the merged DataFrame with raw columns:
      - timestamp, HOEP,
      - Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min,
      - Ontario Demand, Market Demand, temp, humidity, wind_speed
    Returns a DataFrame with engineered, non-leaky features and no raw measured columns.
    """
    df = df.copy()
    
    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    # Day of year 
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lagged features with no leakage
    for k in lags:
        df[f'demand_lag_{k}']       = df['Ontario Demand'].shift(k)
        df[f'temp_lag_{k}']         = df['temp'].shift(k)
        df[f'humidity_lag_{k}']     = df['humidity'].shift(k)
        df[f'wind_speed_lag_{k}']   = df['wind_speed'].shift(k)
        df[f'HOEP_lag_{k}']         = df['HOEP'].shift(k)

    
    # Rolling features
    for win in roll_windows:
        df[f'demand_ma_{win}']     = df['demand_lag_2'].rolling(win).mean()
        df[f'temp_ma_{win}']       = df['temp_lag_2'].rolling(win).mean()
        df[f'humidity_ma_{win}']   = df['humidity_lag_2'].rolling(win).mean()
        df[f'wind_speed_ma_{win}'] = df['wind_speed_lag_2'].rolling(win).mean()
        df[f'HOEP_ma_{win}']       = df['HOEP_lag_2'].rolling(win).mean()
    
    df = df.drop(columns=[
        'Ontario Demand', 'temp', 'humidity', 'wind_speed'
    ])
    
    # Drop rows with any NaNs introduced by shifts/rolls
    return df.dropna()

