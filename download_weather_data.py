# This script automates the download of 88 csv weather files and saves them to data/raw/weather

import os
import requests
from time import sleep

BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
STATION_ID = 51459
FORMAT = "csv"
TIMEFRAME = 1  # Hourly
PROVINCE = "ON"

output_dir = "data/raw/weather/"
os.makedirs(output_dir, exist_ok=True)

for year in range(2018, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 4:
            break  # Stop at April 2025

        params = {
            "format": FORMAT,
            "stationID": STATION_ID,
            "Year": year,
            "Month": month,
            "Day": 1,
            "timeframe": TIMEFRAME,
            "submit": "Download+Data"
        }

        filename = f"toronto_{year}_{month:02}.csv"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"✔️ Already downloaded: {filename}")
            continue

        try:
            print(f"Downloading {filename}...")
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"Saved: {filename}")
            sleep(1)  # delay

        except Exception as e:
            print(f"Failed to download {filename}: {e}")
