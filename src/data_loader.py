import os
import pandas as pd
import numpy as np


def load_hoep_and_demand(raw_dir, start_year=2002, end_year=2025, validate_data=True, handle_suspicious_zeros=True):
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
    
    # Handle invalid dates
    hoep_df = hoep_df.dropna(subset=['Date'])
    demand_df = demand_df.dropna(subset=['Date'])
    
    hoep_df['timestamp'] = hoep_df['Date'] + pd.to_timedelta(hoep_df['Hour'] - 1, unit='h')
    demand_df['timestamp'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'] - 1, unit='h')

    # Select relevant columns
    price_cols = ['HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min']
    hoep_clean = hoep_df[['timestamp'] + price_cols].copy()
    demand_clean = demand_df[['timestamp', 'Ontario Demand']].copy()

    # Convert price columns to numeric, handling any string values
    for col in price_cols:
        if col in hoep_clean.columns:
            hoep_clean[col] = pd.to_numeric(hoep_clean[col], errors='coerce')
    
    demand_clean['Ontario Demand'] = pd.to_numeric(demand_clean['Ontario Demand'], errors='coerce')

    # Merge HOEP and Demand on timestamp
    merged_hd = pd.merge(hoep_clean, demand_clean, on='timestamp', how='inner')
    
    # Handle suspicious zeros BEFORE OR imputation
    if handle_suspicious_zeros:
        print("Handling suspicious zero values...")
        merged_hd = fix_suspicious_zeros(merged_hd)
    
    # Smart OR imputation given their high predictive value
    print("Applying smart OR imputation...")
    merged_hd = impute_or_features(merged_hd)
    
    # Optional data validation
    if validate_data:
        print("\nData validation summary:")
        print(f"Total records: {len(merged_hd)}")
        print(f"Date range: {merged_hd['timestamp'].min()} to {merged_hd['timestamp'].max()}")
        
        # Check for suspicious zeros
        for col in price_cols:
            if col in merged_hd.columns:
                zero_count = (merged_hd[col] == 0).sum()
                total_count = len(merged_hd)
                print(f"{col}: {zero_count} zeros ({zero_count/total_count*100:.1f}%)")
        
        # Check for missing values
        missing_summary = merged_hd.isnull().sum()
        if missing_summary.sum() > 0:
            print("\nMissing values:")
            print(missing_summary[missing_summary > 0])
    
    return merged_hd


def fix_suspicious_zeros(df):
    """
    Fix suspicious zero values that are likely data quality issues.
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['is_peak'] = df['hour'].isin([7, 8, 9, 10, 11, 18, 19, 20, 21])
    
    # Flag 1: HOEP zeros during peak hours when predispatch > $10
    suspicious_hoep = (
        (df['HOEP'] == 0) & 
        (df['is_peak']) & 
        (df['Hour 1 Predispatch'] > 10)
    )
    
    # Flag 2: Isolated zeros (abrupt jumps)
    hoep_prev = df['HOEP'].shift(1)
    hoep_next = df['HOEP'].shift(-1)
    isolated_hoep = (
        (df['HOEP'] == 0) & 
        (hoep_prev > 20) & 
        (hoep_next > 20)
    )
    
    # Flag 3: Multiple simultaneous zeros during peak
    multiple_zero_peak = (
        (df['HOEP'] == 0) & 
        (df['Hour 1 Predispatch'] == 0) &
        (df['is_peak']) &
        (df['Ontario Demand'] > df['Ontario Demand'].quantile(0.7))
    )
    
    # Apply fixes
    suspicious_mask = suspicious_hoep | isolated_hoep | multiple_zero_peak
    suspicious_count = suspicious_mask.sum()
    
    if suspicious_count > 0:
        print(f"  Found {suspicious_count} suspicious zeros to fix")
        
        # Replace suspicious zeros with NaN, then interpolate
        df.loc[suspicious_mask, 'HOEP'] = np.nan
        df['HOEP'] = df['HOEP'].interpolate(method='linear')
        
        # Also fix predispatch if it was part of multiple zero pattern
        predispatch_mask = multiple_zero_peak
        if predispatch_mask.sum() > 0:
            df.loc[predispatch_mask, 'Hour 1 Predispatch'] = np.nan
            df['Hour 1 Predispatch'] = df['Hour 1 Predispatch'].interpolate(method='linear')
    
    return df.drop(['hour', 'is_peak'], axis=1)


def impute_or_features(df):
    """
    Smart imputation for OR features given their high predictive value.
    Uses multiple strategies to preserve the strong correlation with HOEP.
    """
    df = df.copy()
    
    # Add time features for seasonality
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['is_peak'] = df['hour'].isin([7, 8, 9, 10, 11, 18, 19, 20, 21])
    
    or_cols = ['OR 10 Min Sync', 'OR 30 Min']
    
    for col in or_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            total_count = len(df)
            
            if missing_count > 0:
                print(f"  Imputing {missing_count} missing values for {col} ({missing_count/total_count*100:.1f}%)")
                
                # Strategy 1: Use available data to establish HOEP relationship
                available_data = df[df[col].notna()]
                if len(available_data) > 100:  # Need sufficient data
                    
                    # Calculate correlation-based estimates
                    # OR prices tend to be 10-20% of HOEP during stress periods
                    correlation_factor = available_data[col].corr(available_data['HOEP'])
                    
                    # Different ratios for peak vs off-peak
                    peak_ratio = available_data[available_data['is_peak']][col].mean() / available_data[available_data['is_peak']]['HOEP'].mean()
                    offpeak_ratio = available_data[~available_data['is_peak']][col].mean() / available_data[~available_data['is_peak']]['HOEP'].mean()
                    
                    # Handle infinite/NaN ratios
                    if np.isnan(peak_ratio) or np.isinf(peak_ratio):
                        peak_ratio = 0.15
                    if np.isnan(offpeak_ratio) or np.isinf(offpeak_ratio):
                        offpeak_ratio = 0.05
                    
                    # Strategy 2: Time-based median for seasonal patterns
                    hourly_medians = df.groupby('hour')[col].median()
                    monthly_medians = df.groupby('month')[col].median()
                    
                    # Apply imputation strategy based on available information
                    missing_mask = df[col].isnull()
                    
                    for idx in df[missing_mask].index:
                        hour = df.loc[idx, 'hour']
                        month = df.loc[idx, 'month']
                        hoep_val = df.loc[idx, 'HOEP']
                        is_peak = df.loc[idx, 'is_peak']
                        
                        # Method 1: If HOEP is available, use correlation
                        if not np.isnan(hoep_val) and hoep_val > 0:
                            if is_peak:
                                estimated_value = hoep_val * peak_ratio
                            else:
                                estimated_value = hoep_val * offpeak_ratio
                            
                            # Add some noise to avoid perfect correlation
                            noise = np.random.normal(0, estimated_value * 0.1)
                            estimated_value = max(0.01, estimated_value + noise)
                        
                        # Method 2: Use hourly median if available
                        elif not np.isnan(hourly_medians.get(hour, np.nan)):
                            estimated_value = hourly_medians[hour]
                        
                        # Method 3: Use monthly median
                        elif not np.isnan(monthly_medians.get(month, np.nan)):
                            estimated_value = monthly_medians[month]
                        
                        # Method 4: Use overall median as fallback
                        else:
                            estimated_value = df[col].median()
                            if np.isnan(estimated_value):
                                estimated_value = 0.2  # Conservative default
                        
                        df.loc[idx, col] = estimated_value
                
                # Fallback: simple forward fill if correlation approach fails
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still NaN, use a small default value
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(0.2)
    
    return df.drop(['hour', 'month', 'is_peak'], axis=1)


def identify_suspicious_zeros(df, price_cols=['HOEP', 'Hour 1 Predispatch', 'OR 10 Min Sync', 'OR 30 Min']):
    """
    Identify potentially suspicious zero values in electricity price data.
    Returns a DataFrame with suspicious zero indicators.
    """
    df_analysis = df.copy()
    df_analysis['hour'] = df_analysis['timestamp'].dt.hour
    df_analysis['month'] = df_analysis['timestamp'].dt.month
    df_analysis['is_peak'] = df_analysis['hour'].isin([7, 8, 9, 10, 11, 18, 19, 20, 21])  # Peak hours
    
    suspicious_flags = pd.DataFrame(index=df.index)
    
    for col in price_cols:
        if col in df.columns:
            # Flag 1: Zero during peak hours with high demand
            peak_zero_with_high_demand = (
                (df_analysis[col] == 0) & 
                (df_analysis['is_peak']) & 
                (df_analysis['Ontario Demand'] > df_analysis['Ontario Demand'].quantile(0.7))
            )
            
            # Flag 2: Isolated zeros (price jumps from high to 0 to high)
            price_shifted_back = df_analysis[col].shift(1)
            price_shifted_forward = df_analysis[col].shift(-1)
            isolated_zeros = (
                (df_analysis[col] == 0) & 
                (price_shifted_back > 20) & 
                (price_shifted_forward > 20)
            )
            
            # Flag 3: Multiple price columns zero simultaneously during peak
            if col == 'HOEP':  # Only check this once
                multiple_zero_peak = (
                    (df_analysis['HOEP'] == 0) & 
                    (df_analysis['Hour 1 Predispatch'] == 0) &
                    (df_analysis['is_peak'])
                )
                suspicious_flags['multiple_zero_peak'] = multiple_zero_peak
            
            suspicious_flags[f'{col}_peak_zero'] = peak_zero_with_high_demand
            suspicious_flags[f'{col}_isolated_zero'] = isolated_zeros
    
    # Summary of suspicious patterns
    suspicious_summary = suspicious_flags.sum()
    if suspicious_summary.sum() > 0:
        print("Suspicious zero patterns found:")
        print(suspicious_summary[suspicious_summary > 0])
    
    return suspicious_flags

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