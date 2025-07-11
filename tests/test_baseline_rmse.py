import os
import sys
# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


from src.data_loader import load_hoep_and_demand, load_weather, merge_all
from src.feature_engineering import create_features
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np



# --- Load and process full data ---
raw_hoep_dir = "data/raw"
raw_weather_dir = "data/raw/weather"

df = create_features(
    merge_all(
        load_hoep_and_demand(raw_hoep_dir),
        load_weather(raw_weather_dir)
    )
)

# --- Time slice (optional) ---
df = df[df["timestamp"].dt.year >= 2024]  

# --- Clean ---
df = df.dropna(subset=["HOEP", "Hour 1 Predispatch"])
df = df.copy()  # avoid SettingWithCopyWarning

# --- Extract actual and baseline ---
actual     = df["HOEP"].to_numpy(dtype="float32")
predispatch = df["Hour 1 Predispatch"].to_numpy(dtype="float32")

# --- Filter invalid predispatch (same logic as model eval) ---
valid_mask = (predispatch > 1.0) & (predispatch < 300)
actual     = actual[valid_mask]
predispatch = predispatch[valid_mask]

# --- Compute RMSE & MAE ---
rmse = np.sqrt(mean_squared_error(actual, predispatch))
mae  = mean_absolute_error(actual, predispatch)

print(f"Hour-1 Predispatch RMSE: {rmse:.2f} CAD/MWh")
print(f"Hour-1 Predispatch MAE : {mae:.2f} CAD/MWh")
