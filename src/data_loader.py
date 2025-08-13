
import os
import pandas as pd

def load_hoep_and_demand(raw_dir, start_year=2013, end_year=2025, validate_data=True):
    """
    Load and combine HOEP and Demand CSVs from `raw_dir`.
    Returns a DataFrame with:
      - timestamp
      - HOEP
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
    price_cols = ['HOEP']
    hoep_clean = hoep_df[['timestamp'] + price_cols].copy()
    demand_clean = demand_df[['timestamp', 'Ontario Demand']].copy()

    # Convert price columns to numeric
    for col in price_cols:
        if col in hoep_clean.columns:
            hoep_clean[col] = pd.to_numeric(hoep_clean[col], errors='coerce')
    
    demand_clean['Ontario Demand'] = pd.to_numeric(demand_clean['Ontario Demand'], errors='coerce')

    # Merge HOEP and Demand on timestamp
    merged_hd = pd.merge(hoep_clean, demand_clean, on='timestamp', how='inner')



    
    return merged_hd

def load_weather(raw_weather_dir):
    """
    Loads and concatenates all monthly weather CSVs from city subdirectories.
    Expects structure: raw_weather_dir/city_name/*.csv
    Returns a DataFrame with averaged weather data indexed by timestamp.
    """
    import pandas as pd
    import os
    
    all_dfs = []
    
    # Get all city subdirectories
    city_dirs = [d for d in os.listdir(raw_weather_dir) 
                 if os.path.isdir(os.path.join(raw_weather_dir, d))]
    
    print(f"Found cities: {city_dirs}")
    
    for city in city_dirs:
        city_path = os.path.join(raw_weather_dir, city)
        weather_files = [f for f in os.listdir(city_path) if f.endswith(".csv")]
        
        city_dfs = []
        for file in sorted(weather_files):
            file_path = os.path.join(city_path, file)
            try:
                df = pd.read_csv(file_path)
                df['city'] = city  # Track which city this data came from
                city_dfs.append(df)
            except Exception as e:
                print(f" Failed to read {file}: {e}")
        
        if city_dfs:
            city_weather = pd.concat(city_dfs, ignore_index=True)
            all_dfs.append(city_weather)
    
    if not all_dfs:
        raise ValueError("No weather files could be successfully loaded from any city")
    
    # Combine all cities
    weather_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean column names
    weather_df.columns = weather_df.columns.str.strip().str.replace('"', '')
    
    # Rename to clean and parse timestamp
    weather_df = weather_df.rename(columns={"Date/Time (LST)": "timestamp"})
    
    # Parse datetime
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    
    # Keep relevant features only
    cols_to_keep = {
        "timestamp": "timestamp",
        "Temp (°C)": "temp",
        "Rel Hum (%)": "humidity",
        "Wind Spd (km/h)": "wind_speed"
    }
    
    
    weather_df = weather_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    
    # Group by timestamp and take mean across cities (Ontario average)
    weather_df = weather_df.groupby("timestamp")[["temp", "humidity", "wind_speed"]].mean().reset_index()
    
    # Set timestamp as index
    weather_df = weather_df.set_index("timestamp")
    
    # Convert to float32
    weather_df = weather_df.astype("float32")
    
    # Sort and forward fill missing weather values
    weather_df = weather_df.sort_index()
    weather_df[["temp", "humidity", "wind_speed"]] = weather_df[["temp", "humidity", "wind_speed"]].ffill()
    
    print(f"Final weather data shape: {weather_df.shape}")
    return weather_df

def merge_all(hoep_demand_df, weather_df):
    # Use left join to keep all HOEP/Demand records
    df_merged = pd.merge(hoep_demand_df, weather_df, on='timestamp', how='left')
    
    # Forward fill weather data after merge
    cols_to_fill = ['temp', 'humidity','wind_speed', 'HOEP',"Ontario Demand"]
    df_merged[cols_to_fill] = df_merged[cols_to_fill].ffill()
    
    # Optionally back fill remaining missing values
    df_merged[cols_to_fill] = df_merged[cols_to_fill].bfill()
    print(df_merged.head())
    
    return df_merged
