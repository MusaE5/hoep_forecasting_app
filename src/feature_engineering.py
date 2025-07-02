
import pandas as pd
import numpy as np

def create_features(df, lags=[1, 2, 3], roll_windows=[3]):
    """
    Add time-based, lagged, and rolling features to the merged dataframe.
    Returns enhanced feature DataFrame.
    """
    df = df.copy()

    # Time-based features 
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)

    #  Lag features 
    for lag in lags:
        df[f'HOEP_lag_{lag}'] = df['HOEP'].shift(lag)
        df[f'Demand_lag_{lag}'] = df['Ontario Demand'].shift(lag)
       

    #  Rolling features 
    for win in roll_windows:
        df[f'HOEP_ma_{win}'] = df['HOEP'].rolling(window=win).mean()
        df[f'Demand_ma_{win}'] = df['Ontario Demand'].rolling(window=win).mean()

    # Drop rows with any NA created by shift/rolling
    df = df.dropna()

    return df
