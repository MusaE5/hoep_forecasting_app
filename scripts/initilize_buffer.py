# scripts/initialize_buffer.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_initial_buffer(hours=4):
    """Generates starter data with realistic values"""
    timestamps = [(datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") 
                 for i in range(hours, 0, -1)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'source_timestamp': timestamps,
        'demand_MW': np.linspace(15000, 14900, hours),
        'or10_sync_MW': np.linspace(500, 490, hours),
        'or30_MW': np.linspace(300, 295, hours),
        'temp_C': np.linspace(22.1, 22.0, hours),
        'humidity_%': np.linspace(65, 66, hours),
        'wind_mps': np.linspace(3.2, 3.1, hours),
        'zonal_price': np.linspace(40.1, 40.2, hours)
    })

if __name__ == "__main__":
    df = create_initial_buffer()
    df.to_csv("data/feature_buffer.csv", index=False)
    print("Initial buffer created with 4 hours of synthetic data")