
import io
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from datetime import datetime
import schedule
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
BUFFER_FILE = "data/hoep_buffer.csv"

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
        "demand_MW":          float(latest["TOTAL LOAD"]),
        "or10_sync_MW":       float(latest["TOTAL 10S"]),
        "or10_non_MW":        float(latest["TOTAL 10N"]),
        "or30_MW":            float(latest["TOTAL 30R"]),
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

def create_buffer_if_needed():
    """Create buffer CSV file if it doesn't exist"""
    if not os.path.exists(BUFFER_FILE):
        # Create directory if needed
        os.makedirs(os.path.dirname(BUFFER_FILE), exist_ok=True)
        
        # Create empty CSV with headers
        df = pd.DataFrame(columns=[
            'timestamp', 'hour', 'interval', 'demand_MW', 
            'or10_sync_MW', 'or10_non_MW', 'or30_MW',
            'temp_C', 'humidity_%', 'wind_mps', 'zonal_price'
        ])
        df.to_csv(BUFFER_FILE, index=False)
        print(f"Created new buffer file: {BUFFER_FILE}")

def append_to_buffer(data_dict):
    """Append new data to the buffer CSV"""
    try:
        # Load existing buffer
        if os.path.exists(BUFFER_FILE):
            buffer_df = pd.read_csv(BUFFER_FILE)
        else:
            create_buffer_if_needed()
            buffer_df = pd.read_csv(BUFFER_FILE)
        
        # Create new row
        new_row = pd.DataFrame([data_dict])
        
        # Append to buffer
        updated_buffer = pd.concat([buffer_df, new_row], ignore_index=True)
        
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
        print(f"\n🔄 Fetching live data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        feat["timestamp"] = datetime.now().isoformat(timespec="seconds")

        print("── LIVE FEATURE SNAPSHOT ──")
        for k, v in feat.items():
            print(f"{k:<15}: {v}")
        
        # Store in buffer
        success = append_to_buffer(feat)
        
        if success:
            print("📊 Data successfully stored in buffer")
        else:
            print("⚠️  Failed to store data in buffer")
            
    except Exception as e:
        print(f"❌ Error in fetch_and_store: {e}")

def show_buffer_status():
    """Show current buffer status"""
    try:
        if os.path.exists(BUFFER_FILE):
            buffer_df = pd.read_csv(BUFFER_FILE)
            print(f"\n📈 Buffer Status:")
            print(f"   Total rows: {len(buffer_df)}")
            print(f"   Hours of data: {len(buffer_df)}")
            print(f"   Ready for lag_24?: {'✅ Yes' if len(buffer_df) >= 24 else f'❌ No (need {24 - len(buffer_df)} more)'}")
            
            if len(buffer_df) > 0:
                latest = buffer_df.iloc[-1]
                print(f"   Latest: {latest['timestamp']} | HOEP: {latest['zonal_price']} | Demand: {latest['demand_MW']}")
        else:
            print("📈 Buffer Status: No buffer file exists yet")
    except Exception as e:
        print(f"❌ Error checking buffer status: {e}")

# Main execution
if __name__ == "__main__":
    print("🚀 HOEP Live Data Collector Starting...")
    print("⏰ Scheduled to run every hour at :56")
    
    # Create buffer if needed
    create_buffer_if_needed()
    
    # Show current status
    show_buffer_status()
    
    # Option to run immediately for testing
    user_input = input("\n🔍 Run immediately for testing? (y/n): ").lower().strip()
    if user_input == 'y':
        fetch_and_store()
        show_buffer_status()
    
    # Schedule the job
    schedule.every().hour.at(":56").do(fetch_and_store)
    
    print(f"\n⏰ Scheduler started. Next run at next :56 minute mark.")
    print("   Press Ctrl+C to stop")
    
    # Run scheduler
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n👋 Scheduler stopped by user")
        show_buffer_status()  # Show final status