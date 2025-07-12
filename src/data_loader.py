
import os
import pandas as pd

def load_hoep_and_demand(raw_dir, start_year=2002, end_year=2025, validate_data=True):
    """
    Load and combine HOEP and Demand CSVs from `raw_dir`.
    Returns a DataFrame with:
      - timestamp
      - HOEP, Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min
      - Ontario Demand
    """
    hoep_dfs = []
    demand_dfs = []
    
    for year in range(start_year, end_year + 1):
        hoep_file = f'PUB_PriceHOEPPredispOR_{year}.csv'
        demand_file = f'PUB_Demand_{year}.csv'
        hoep_path = os.path.join(raw_dir, hoep_file)
        demand_path = os.path.join(raw_dir, demand_file)

        # Load HOEP data
        if os.path.exists(hoep_path):
            try:
                df = pd.read_csv(hoep_path, skiprows=3)
                df['Year'] = year
                hoep_dfs.append(df)
            except Exception as e:
                print(f"Error loading HOEP file {hoep_file}: {e}")
        else:
            print(f"Warning: HOEP file not found: {hoep_file}")

        # Load Demand data
        if os.path.exists(demand_path):
            try:
                df = pd.read_csv(demand_path, skiprows=3)
                df['Year'] = year
                demand_dfs.append(df)
            except Exception as e:
                print(f"Error loading Demand file {demand_file}: {e}")
        else:
            print(f"Warning: Demand file not found: {demand_file}")

    if not hoep_dfs:
        raise ValueError("No HOEP files could be loaded")
    if not demand_dfs:
        raise ValueError("No Demand files could be loaded")

    # Concatenate
    hoep_df = pd.concat(hoep_dfs, ignore_index=True)
    demand_df = pd.concat(demand_dfs, ignore_index=True)

    # Clean column names
    hoep_df.columns = hoep_df.columns.str.strip()
    demand_df.columns = demand_df.columns.str.strip()

    # Build timestamp
    hoep_df['Date'] = pd.to_datetime(hoep_df['Date'], format='%Y-%m-%d', errors='coerce')
    demand_df['Date'] = pd.to_datetime(demand_df['Date'], format='%Y-%m-%d', errors='coerce')
    
    hoep_df = hoep_df.dropna(subset=['Date'])
    demand_df = demand_df.dropna(subset=['Date'])
    
    hoep_df['timestamp'] = hoep_df['Date'] + pd.to_timedelta(hoep_df['Hour'] - 1, unit='h')
    demand_df['timestamp'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'] - 1, unit='h')

    # Select relevant columns
    price_cols = ['HOEP', 'Hour 1 Predispatch','Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 30 Min', 'OR 10 Min non-sync']
    hoep_clean = hoep_df[['timestamp'] + price_cols].copy()
    demand_clean = demand_df[['timestamp', 'Ontario Demand']].copy()

    # Convert price columns to numeric
    for col in price_cols:
        if col in hoep_clean.columns:
            hoep_clean[col] = pd.to_numeric(hoep_clean[col], errors='coerce')
    
    demand_clean['Ontario Demand'] = pd.to_numeric(demand_clean['Ontario Demand'], errors='coerce')

    # Merge HOEP and Demand on timestamp
    merged_hd = pd.merge(hoep_clean, demand_clean, on='timestamp', how='inner')

    # Forward fill ONLY Hour 1 Predispatch
    merged_hd['Hour 1 Predispatch'] = merged_hd['Hour 1 Predispatch'].fillna(method='ffill')
    
    merged_hd['Hour 2 Predispatch'] = merged_hd['Hour 2 Predispatch'].fillna(method='ffill')

    merged_hd['Hour 3 Predispatch'] = merged_hd['Hour 3 Predispatch'].fillna(method='ffill')

    # Optional data validation
    if validate_data:
        print("\nData validation summary:")
        print(f"Total records: {len(merged_hd)}")
        print(f"Date range: {merged_hd['timestamp'].min()} to {merged_hd['timestamp'].max()}")
        
        for col in price_cols:
            if col in merged_hd.columns:
                zero_count = (merged_hd[col] == 0).sum()
                total_count = len(merged_hd)
                print(f"{col}: {zero_count} zeros ({zero_count / total_count * 100:.1f}%)")
        
        missing_summary = merged_hd.isnull().sum()
        if missing_summary.sum() > 0:
            print("\nMissing values:")
            print(missing_summary[missing_summary > 0])
    
    return merged_hd


def load_weather(raw_weather_dir):
    """
    Loads and concatenates all monthly weather CSVs from raw_weather_dir.
    Cleans and aligns the data, returning a DataFrame indexed by timestamp.
    """
    import pandas as pd
    import os

    weather_files = [f for f in os.listdir(raw_weather_dir) if f.endswith(".csv")]
    dfs = []

    for file in sorted(weather_files):
        path = os.path.join(raw_weather_dir, file)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f" Failed to read {file}: {e}")

    weather_df = pd.concat(dfs, ignore_index=True)

    # Clean column names 
    weather_df.columns = weather_df.columns.str.strip().str.replace('"', '')

    # Rename to clean and parse timestamp
    weather_df = weather_df.rename(columns={"Date/Time (LST)": "timestamp"})

    # Parse datetime
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    weather_df = weather_df.dropna(subset=["timestamp"])

    # Keep relevant features only
    cols_to_keep = {
        "timestamp": "timestamp",
        "Temp (°C)": "temp",
        "Rel Hum (%)": "humidity",
        "Wind Spd (km/h)": "wind_speed"
    }

    weather_df = weather_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    weather_df = weather_df.set_index("timestamp")

    # Convert to float32
    weather_df = weather_df.astype("float32")

    # Sort and forward fill missing weather values
    weather_df = weather_df.sort_index()
    weather_df[["temp", "humidity", "wind_speed"]] = weather_df[["temp", "humidity", "wind_speed"]].ffill()

    return weather_df

def merge_all(hoep_demand_df, weather_df):
    """
    Merge the HOEP/Demand DataFrame with weather DataFrame on 'timestamp'.
    Returns a unified DataFrame.
    """
    df_hd = hoep_demand_df.copy()

    # Ensure timestamp is datetime and set as index for merge
    df_hd['timestamp'] = pd.to_datetime(df_hd['timestamp'])
    df_hd = df_hd.set_index('timestamp')

    # Join on timestamp
    df_merged = df_hd.join(weather_df, how='inner')

    return df_merged.reset_index()