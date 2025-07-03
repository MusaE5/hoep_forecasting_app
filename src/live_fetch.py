import io
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from datetime import datetime

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

def fetch_realtime_totals():
    """Return dict with demand + OR values from the latest 5-min row."""
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
        "hour": int(latest["HOUR"]),
        "interval": int(latest["INTERVAL"]),
        "demand_MW": float(latest["TOTAL LOAD"]),
        "or10_sync_MW": float(latest["TOTAL 10S"]),
        "or10_non_MW": float(latest["TOTAL 10N"]),
        "or30_MW": float(latest["TOTAL 30R"]),
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

# Main
if __name__ == "__main__":
    # Fetch all data
    feat = fetch_realtime_totals()
    feat.update(fetch_current_weather())
    zonal_price, zonal_timestamp = get_ontario_zonal_average()
    
    # Add zonal price data if available
    if zonal_price is not None:
        feat["zonal_price"] = zonal_price
        feat["zonal_timestamp"] = zonal_timestamp
    
    feat["timestamp"] = datetime.now().isoformat(timespec="seconds")

    print("── LIVE FEATURE SNAPSHOT ──")
    for k, v in feat.items():
        print(f"{k:<15}: {v}")