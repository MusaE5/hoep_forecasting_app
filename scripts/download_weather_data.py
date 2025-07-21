import os
import requests
from time import sleep
import pandas as pd

# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
STATION_ID = 50093  # LONDON A
FORMAT = "csv"
TIMEFRAME = 1  # Hourly
PROVINCE = "ON"

output_dir = "data/raw/weather/london/"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------
# DOWNLOAD LOOP
# ------------------------------------------
for year in range(2013, 2026):  # 2025 inclusive
    for month in range(1, 13):
        if year == 2025 and month > 4:
            break  # Stop at April 2025

        filename = f"london_{year}_{month:02}.csv"
        filepath = os.path.join(output_dir, filename)
        filepath_utc = filepath.replace(".csv", "_utc.csv")

        if os.path.exists(filepath_utc):
            print(f" Already processed: {os.path.basename(filepath_utc)}")
            continue

        # Download
        try:
            print(f"Downloading {filename}...")
            params = {
                "format": FORMAT,
                "stationID": STATION_ID,
                "Year": year,
                "Month": month,
                "Day": 1,
                "timeframe": TIMEFRAME,
                "submit": "Download+Data"
            }

            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"Saved: {filename}")

        except Exception as e:
            print(f" XXXXX Failed to download {filename}: {e}")
            continue

        # Convert LST → UTC
        try:
            df = pd.read_csv(filepath)
            if "Date/Time (LST)" not in df.columns:
                print(f" Missing 'Date/Time (LST)' in {filename}, skipping.")
                continue

            df['Date/Time'] = pd.to_datetime(df['Date/Time (LST)'], errors='coerce')
            df['Date/Time'] = df['Date/Time'].dt.tz_localize(
                'Canada/Eastern',
                ambiguous='NaT',
                nonexistent='shift_forward'
            )
            df['Date/Time_UTC'] = df['Date/Time'].dt.tz_convert('UTC')
            df = df.drop(columns=['Date/Time (LST)'])

            # Save new file
            df.to_csv(filepath_utc, index=False)
            print(f"Converted and saved: {os.path.basename(filepath_utc)}")

            # Optional: delete original
            os.remove(filepath)

        except Exception as e:
            print(f" XXXXX Failed to process {filename}: {e}")

        sleep(1)  # Delay to avoid hammering the server
