import os
import pandas as pd

def load_hoep_and_demand(raw_dir):
    """
    Load and combine HOEP (price) and Ontario Demand CSVs into a single DataFrame.
    Expects yearly files in raw_dir (2013–2025). Returns a DataFrame indexed by timestamp.
    
    """
    
    start_year = 2013
    end_year = 2025
    
    hoep_dfs = []
    demand_dfs = []
    
    for year in range(start_year, end_year + 1): # Include 2025
        hoep_file = f'PUB_PriceHOEPPredispOR_{year}.csv'
        demand_file = f'PUB_Demand_{year}.csv'
        hoep_path = os.path.join(raw_dir, hoep_file)
        demand_path = os.path.join(raw_dir, demand_file)

        # Read and append yearly Demand and price CSVs
        try:
            df = pd.read_csv(hoep_path, skiprows=3) # First 3 rows are comments in the raw CSV's
            hoep_dfs.append(df)
        except Exception as e:
            print(f"Failed to load {hoep_path}: {e}")
        
        
        try:
            df = pd.read_csv(demand_path, skiprows=3)
            demand_dfs.append(df)
        except Exception as e:
            print(f"Failed to load {demand_file}: {e}")
       
    # Concatenate into a single df after the loop
    hoep_df = pd.concat(hoep_dfs, ignore_index=True)
    demand_df = pd.concat(demand_dfs, ignore_index=True)

    hoep_df.columns = hoep_df.columns.str.strip()
    demand_df.columns = demand_df.columns.str.strip()

    # Build timestamp for time-based features
    hoep_df['Date'] = pd.to_datetime(hoep_df['Date'], format='%Y-%m-%d', errors='coerce')
    demand_df['Date'] = pd.to_datetime(demand_df['Date'], format='%Y-%m-%d', errors='coerce')
    
    hoep_df = hoep_df.dropna(subset=['Date'])
    demand_df = demand_df.dropna(subset=['Date'])
    
    # Convert timestamp to hour range 0-23 instead of 1-24
    hoep_df['timestamp'] = hoep_df['Date'] + pd.to_timedelta(hoep_df['Hour'] - 1, unit='h')
    demand_df['timestamp'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'] - 1, unit='h')

    # Select columns needed
    hoep_clean = hoep_df[['timestamp', 'HOEP']].copy()
    demand_clean = demand_df[['timestamp', 'Ontario Demand']].copy()
    
    merged_hd = pd.merge(hoep_clean, demand_clean, on='timestamp', how='inner')
    merged_hd = merged_hd.set_index("timestamp").sort_index()
    merged_hd[['HOEP', 'Ontario Demand']] = merged_hd[['HOEP', 'Ontario Demand']].ffill()
    
    return merged_hd

def load_weather(raw_weather_dir):
    """
    Load and combine monthly Toronto weather CSVs into a single DataFrame.
    Expects raw_weather_dir/toronto/*.csv. Returns a DataFrame indexed by timestamp.
    
    """
    # Get Toronto subdirectory
    city_path = os.path.join(raw_weather_dir, "toronto")
    weather_files = [f for f in os.listdir(city_path) if f.endswith(".csv")] # Get every monthly file that ends in .csv
        
    city_df = []
    for file in sorted(weather_files): # Sort by date
        file_path = os.path.join(city_path, file) 
        try:
            df = pd.read_csv(file_path)
            city_df.append(df)
        except Exception as e:
            print(f" Failed to read {file}: {e}")
        

    weather_df = pd.concat(city_df, ignore_index=True)    
    # Clean column names
    weather_df.columns = weather_df.columns.str.strip().str.replace('"', '')
    
    # Rename to clean and create timestamp
    weather_df = weather_df.rename(columns={"Date/Time (LST)": "timestamp"})
    
    # Parse datetime, no need for timedelta, already in correct format (0-23)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    
    # Keep relevant features only
    cols_to_keep = {
        "timestamp": "timestamp",
        "Temp (°C)": "temp",
        "Rel Hum (%)": "humidity",
        "Wind Spd (km/h)": "wind_speed"
    }
    
    weather_df = weather_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    weather_df = weather_df.set_index("timestamp")
    
    # Sort and forward fill missing weather values
    weather_df = weather_df.sort_index()
    weather_df[["temp", "humidity", "wind_speed"]] = weather_df[["temp", "humidity", "wind_speed"]].ffill()
    
    return weather_df

def merge_all(hoep_demand_df, weather_df):
    df_merged = hoep_demand_df.join(weather_df, how='left')
    # Convert all columns to numeric
    df_merged = df_merged.apply(pd.to_numeric, errors="coerce")
    print(df_merged.head())
    return df_merged
