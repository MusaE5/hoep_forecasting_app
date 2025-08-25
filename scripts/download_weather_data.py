import os
import requests
from time import sleep


BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
FORMAT = "csv"
TIMEFRAME = 1  # Hourly
PROVINCE = "ON"

CITIES = {
    "toronto": 48549,
    "london": 50093,
    "kitchener": 48569,
    "ottawa": 30578
}


for city, station_id in CITIES.items():
    output_dir = f"data/raw/weather/{city}"
    os.makedirs(output_dir, exist_ok=True)

    for year in range(2013, 2026):  # Include 2025
        for month in range(1, 13):
            if year == 2025 and month > 4:
                break  # Stop after April 2025

            filename = f"{city}_{year}_{month:02}.csv"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"✅ Already exists: {filename}")
                continue

            print(f"⬇️ Downloading {filename}...")

            params = {
                "format": FORMAT,
                "stationID": station_id,
                "Year": year,
                "Month": month,
                "Day": 1,
                "timeframe": TIMEFRAME,
                "submit": "Download+Data"
            }

            try:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                print(f"✅ Saved: {filename}")

            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

            sleep(0.4)  
