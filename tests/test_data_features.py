import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_hoep_and_demand, load_weather, merge_all, identify_suspicious_zeros
from src.feature_engineering import create_features
import pandas as pd

def test_full_data_pipeline():
    raw_hoep_dir = "data/raw"
    raw_weather_dir = "data/raw/weather"
    
    print("="*60)
    print("ELECTRICITY PRICE DATA PIPELINE TEST")
    print("="*60)
    
    # Load HOEP and Demand data with validation
    print("\n1. Loading HOEP and Demand data...")
    hd_df = load_hoep_and_demand(raw_hoep_dir, validate_data=True)
    
    # Load weather data
    print("\n2. Loading weather data...")
    weather_df = load_weather(raw_weather_dir)
    
    # Merge all data
    print("\n3. Merging datasets...")
    merged_df = merge_all(hd_df, weather_df)
    
    print(f"\nFinal merged dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
    
    # Test for suspicious zeros
    print("\n4. Analyzing suspicious zero patterns...")
    price_cols = ['HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min']
    available_price_cols = [col for col in price_cols if col in merged_df.columns]
    
    if available_price_cols:
        suspicious_flags = identify_suspicious_zeros(merged_df, available_price_cols)
        
        # Count suspicious patterns
        total_suspicious = suspicious_flags.any(axis=1).sum()
        print(f"\nTotal records with suspicious patterns: {total_suspicious}")
        
        if total_suspicious > 0:
            print("\nSample of suspicious records:")
            suspicious_records = merged_df[suspicious_flags.any(axis=1)].head(10)
            print(suspicious_records[['timestamp'] + available_price_cols + ['Ontario Demand']])
            
            # Show what specific flags were triggered
            print("\nSuspicious pattern breakdown:")
            for col in suspicious_flags.columns:
                count = suspicious_flags[col].sum()
                if count > 0:
                    print(f"  {col}: {count} records")
    
    # Data quality checks
    print("\n5. Data quality summary...")
    print(f"Missing values per column:")
    missing_counts = merged_df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    # Price distribution analysis
    print("\n6. Price distribution analysis...")
    for col in available_price_cols:
        prices = merged_df[col].dropna()
        if len(prices) > 0:
            print(f"\n{col} statistics:")
            print(f"  Min: ${prices.min():.2f}")
            print(f"  Max: ${prices.max():.2f}")
            print(f"  Mean: ${prices.mean():.2f}")
            print(f"  Median: ${prices.median():.2f}")
            print(f"  Zeros: {(prices == 0).sum()} ({(prices == 0).sum()/len(prices)*100:.1f}%)")
            print(f"  Negative: {(prices < 0).sum()} ({(prices < 0).sum()/len(prices)*100:.1f}%)")
    
    # Time series completeness check
    print("\n7. Time series completeness check...")
    expected_records = (merged_df['timestamp'].max() - merged_df['timestamp'].min()).total_seconds() / 3600 + 1
    actual_records = len(merged_df)
    completeness = actual_records / expected_records * 100
    print(f"Expected hourly records: {expected_records:.0f}")
    print(f"Actual records: {actual_records}")
    print(f"Completeness: {completeness:.1f}%")
    
    if completeness < 95:
        print("WARNING: Significant gaps in time series data detected!")
        
        # Find gaps
        merged_df_sorted = merged_df.sort_values('timestamp')
        time_diffs = merged_df_sorted['timestamp'].diff()
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
        
        if len(gaps) > 0:
            print(f"Found {len(gaps)} gaps longer than 1 hour:")
            for idx, gap in gaps.head(5).items():
                gap_start = merged_df_sorted.loc[idx-1, 'timestamp']
                gap_end = merged_df_sorted.loc[idx, 'timestamp']
                print(f"  Gap from {gap_start} to {gap_end} ({gap})")
    
    # Preview final data
    print("\n8. Preview merged data (last 5 rows):")
    print(merged_df.tail())
    
    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETED")
    print("="*60)
    
    return merged_df

if __name__ == "__main__":
    df = test_full_data_pipeline()