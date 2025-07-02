import os
import pandas as pd

def load_hoep_and_demand(raw_dir, start_year=2002, end_year=2025):
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
            df = pd.read_csv(hoep_path, skiprows=3)
            df['Year'] = year
            hoep_dfs.append(df)
        else:
            print(f"Warning: HOEP file not found: {hoep_file}")

        # Load Demand data
        if os.path.exists(demand_path):
            df = pd.read_csv(demand_path, skiprows=3)
            df['Year'] = year
            demand_dfs.append(df)
        else:
            print(f"Warning: Demand file not found: {demand_file}")

    # Concatenate
    hoep_df = pd.concat(hoep_dfs, ignore_index=True)
    demand_df = pd.concat(demand_dfs, ignore_index=True)

    # Clean column names
    hoep_df.columns = hoep_df.columns.str.strip()
    demand_df.columns = demand_df.columns.str.strip()

    # Build timestamp
    hoep_df['Date'] = pd.to_datetime(hoep_df['Date'], format='%Y-%m-%d')
    demand_df['Date'] = pd.to_datetime(demand_df['Date'], format='%Y-%m-%d')
    hoep_df['timestamp'] = hoep_df['Date'] + pd.to_timedelta(hoep_df['Hour'] - 1, unit='h')
    demand_df['timestamp'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'] - 1, unit='h')

    # Select relevant columns
    hoep_clean = hoep_df[['timestamp', 'HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min']].copy()
    demand_clean = demand_df[['timestamp', 'Ontario Demand']].copy()

    # Merge HOEP and Demand on timestamp
    merged_hd = pd.merge(hoep_clean, demand_clean, on='timestamp', how='inner')
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

    # Optimize memory usage
    weather_df = weather_df.astype("float32")

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




   


    