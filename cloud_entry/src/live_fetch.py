import io
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from datetime import datetime
import time
import os
from pathlib import Path

# Constants
URL_TOTALS = (
    "https://reports-public.ieso.ca/public/RealtimeTotals/"
    "PUB_RealtimeTotals.csv?download=1"
)
URL_ZONAL_PRICE = "https://reports-public.ieso.ca/public/RealtimeOntarioZonalPrice/PUB_RealtimeOntarioZonalPrice.xml"
HEADERS = {"User-Agent": "Mozilla/5.0"}          # avoids 403s
LAT, LON = 43.676, -79.6305                      # Toronto
TIMEZONE = "America/Toronto"
NAMESPACE = {'ns': 'http://www.ieso.ca/schema'}

# Buffer settings
TMP = "/tmp" if os.path.isdir("/tmp") else "data"
BUFFER_FILE = os.path.join(TMP, "hoep_buffer.csv")

def fetch_realtime_totals():
    """Return dict with demand + OR values from the latest 5-min row, plus market demand."""
    r = requests.get(URL_TOTALS, headers=HEADERS, timeout=15)
    r.raise_for_status()

    # Find the real header row (first line that has TOTAL LOAD in it)
    lines = r.text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if "TOTAL LOAD" in l.upper())

    df = pd.read_csv(
        io.StringIO("\n".join(lines[hdr:])),
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip().str.upper()

    latest = df.iloc[-1]          # last 5-min interval

    return {
        "hour":               int(latest["HOUR"]),
        "interval":           int(latest["INTERVAL"]),
        "demand_MW":          float(latest["TOTAL LOAD"])
    }

def fetch_current_weather():
    """Return dict with weather for the current hour in Toronto."""
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relativehumidity_2m,wind_speed_10m",
        "timezone": TIMEZONE,
    }
    jw = requests.get("https://api.open-meteo.com/v1/forecast",
                     params=params, timeout=15).json()["hourly"]

    now_hr = datetime.now().replace(minute=0, second=0,
                                   microsecond=0).isoformat()
    idx = max(i for i, t in enumerate(jw["time"]) if t <= now_hr)

    return {
        "temp_C": jw["temperature_2m"][idx],
        "humidity_%": jw["relativehumidity_2m"][idx],
        "wind_mps": jw["wind_speed_10m"][idx],
    }

def get_ontario_zonal_average():
    """
    Extracts the official Average Zonal Price from IESO's XML feed.
    Returns: (average_price, timestamp)
    """
    try:
        # Fetch and parse XML
        response = requests.get(URL_ZONAL_PRICE, headers=HEADERS, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Find the first RealTimePriceComponents block (Zonal Price)
        zonal_price_block = root.find('.//ns:RealTimePriceComponents[ns:OntarioZonalPrice="Zonal Price"]', NAMESPACE)
        
        if zonal_price_block is None:
            raise ValueError("Zonal Price section not found in XML")

        # Extract the official average
        average = zonal_price_block.find('ns:AverageHeading', NAMESPACE)
        if average is None or not average.text:
            raise ValueError("Average value not found in Zonal Price section")

        # Extract timestamp
        timestamp = root.find('.//ns:CreatedAt', NAMESPACE).text

        return float(average.text), timestamp

    except Exception as e:
        print(f"Zonal Price Error: {e}")
        return None, None


def append_to_buffer(data_dict):
    """Append new data to the buffer CSV and keep only latest 24 rows (duplicates allowed)"""
    try:
        # Load existing buffer
        if os.path.exists(BUFFER_FILE):
            buffer_df = pd.read_csv(BUFFER_FILE)
        else:
            print("⚠️ No existing buffer found — creating new one")
            buffer_df = pd.DataFrame()

        # Create new row
        new_row = pd.DataFrame([data_dict])

        # Append to buffer
        updated_buffer = pd.concat([buffer_df, new_row], ignore_index=True)

        # Ensure timestamp is datetime 
        updated_buffer['timestamp'] = pd.to_datetime(updated_buffer['timestamp'], errors='coerce')
        

        # Always keep only the last 24 rows
        updated_buffer = updated_buffer.tail(24)

        # Save back to CSV
        updated_buffer.to_csv(BUFFER_FILE, index=False)
       

        print(f"✅ Data appended to buffer. Total rows: {len(updated_buffer)}")
        return True

    except Exception as e:
        print(f"❌ Error appending to buffer: {e}")
        return False

def fetch_and_store():
    """Main function to fetch live data and store in buffer"""
    try:
        print(f"\n Fetching live data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch all data
        feat = fetch_realtime_totals()
        feat.update(fetch_current_weather())
        zonal_price, zonal_timestamp = get_ontario_zonal_average()
        
        # Add zonal price data if available
        if zonal_price is not None:
            feat["zonal_price"] = zonal_price
        else:
            feat["zonal_price"] = None  # Store None if failed
        
        # Add timestamp
        feat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        print("── LIVE FEATURE SNAPSHOT ──")
        for k, v in feat.items():
            print(f"{k:<15}: {v}")
        
        # Store in buffer
        success = append_to_buffer(feat)
        
        if success:
            return feat
        else:
            print("  Failed to store data in buffer")
            
    except Exception as e:
        print(f" Error in fetch_and_store: {e}")


def fetch_live_features_only():
    """
    Fetches live features (market + weather + zonal price) without modifying any buffer files.
    This is safe for manual prediction.
    """
    try:

        feat = fetch_realtime_totals()
        feat.update(fetch_current_weather())
        zonal_price, _ = get_ontario_zonal_average()

        feat["zonal_price"] = zonal_price if zonal_price is not None else None
        feat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        return feat

    except Exception as e:
        print(f"❌ Error in fetch_live_features_only: {e}")
        return None

