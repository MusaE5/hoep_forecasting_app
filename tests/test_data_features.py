# tests/test_data_validation.py

import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_hoep_and_demand, load_weather, merge_all

def validate_columns(df, columns, title="VALIDATION"):
    print(f"\n=== {title.upper()} ===")
    for col in columns:
        if col not in df.columns:
            print(f"  Missing column: {col}")
            continue

        print(f"\n--- {col} ---")
        series = df[col].dropna()

        print(f"  Count: {len(series):,}")
        print(f"  Missing: {df[col].isnull().sum():,}")
        print(f"  Range: {series.min():.2f} to {series.max():.2f}")
        print(f"  Mean: {series.mean():.2f}")
        print(f"  Std Dev: {series.std():.2f}")
        print(f"  Zero values: {(series == 0).sum():,}")
        print(f"  Negative values: {(series < 0).sum():,}")

        if (series == series.mode()[0]).sum() > 0.1 * len(series):
            print(f" High frequency :{series.mode()[0]}")

def test_data_loader():
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    weather_data_dir = os.path.join(raw_data_dir, "weather")

    print(" Loading HOEP + Demand + Weather data...")
    df = merge_all(
        load_hoep_and_demand(raw_data_dir, validate_data=True),
        load_weather(weather_data_dir)
    )

    print(f"\nLoaded {len(df):,} rows")

    price_cols = ['HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min', 'OR 10 Min non-sync']
    env_cols = ['Ontario Demand', 'temp', 'humidity', 'wind_speed']

    validate_columns(df, price_cols, title="PRICE DATA VALIDATION")
    validate_columns(df, env_cols, title="DEMAND & WEATHER VALIDATION")

if __name__ == "__main__":
    test_data_loader()
