# tests/test_data_validation.py

import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_hoep_and_demand

def validate_price_data(df, price_cols):
    print("\n=== PRICE DATA VALIDATION ===")
    for col in price_cols:
        if col not in df.columns:
            print(f"  ❌ Missing column: {col}")
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

        if series.max() > 2000 or series.min() < -200:
            print("  🚨 Extreme values detected!")
        if (series == series.mode()[0]).sum() > 0.1 * len(series):
            print(f"  ⚠️ High frequency of a single value: {series.mode()[0]}")

def test_data_loader():
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    print("📥 Loading HOEP + Demand data...")
    df = load_hoep_and_demand(raw_data_dir, validate_data=True)

    print(f"\n✅ Loaded {len(df):,} rows")

    price_cols = ['HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min']
    validate_price_data(df, price_cols)

if __name__ == "__main__":
    test_data_loader()
