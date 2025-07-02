import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_hoep_and_demand, load_weather, merge_all
from src.feature_engineering import create_features
import pandas as pd


def test_full_data_pipeline():
    raw_hoep_dir = "data/raw"
    raw_weather_dir = "data/raw/weather"

    # Load data
    hd_df = load_hoep_and_demand(raw_hoep_dir)
    weather_df = load_weather(raw_weather_dir)

    # Merge
    merged_df = merge_all(hd_df, weather_df)
    merged_df = create_features(merged_df)

    # Preview
    print("\n Preview merged + engineered data:")
    print(merged_df.head())
    print(f"Shape: {merged_df.shape}")
    print(f"Time range: {merged_df['timestamp'].min()} → {merged_df['timestamp'].max()}")
    print("Columns:", merged_df.columns.tolist())

    # Clean columns
    merged_df.columns = merged_df.columns.str.strip().str.lower()
    merged_df['hoep'] = pd.to_numeric(merged_df['hoep'], errors='coerce')

    # Correlation test
    corr = merged_df.corr(numeric_only=True)
    print("\n Top correlated features with HOEP:")
    print(corr['hoep'].sort_values(ascending=False))


if __name__ == "__main__":
    test_full_data_pipeline()
