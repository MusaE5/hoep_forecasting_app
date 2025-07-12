import os
import sys
# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_hoep_and_demand, load_weather, merge_all
from src.feature_engineering import create_features
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


raw_hoep_dir = "data/raw"
raw_weather_dir = "data/raw/weather"

df = create_features(
    merge_all(
        load_hoep_and_demand(raw_hoep_dir),
        load_weather(raw_weather_dir)
    )
)

# Time slice
df = df[df["timestamp"].dt.year >= 2024]  

#
df = df.dropna(subset=["HOEP", "Hour 1 Predispatch", "Hour 2 Predispatch", "Hour 3 Predispatch"])
df = df.copy()  

actual         = df["HOEP"].to_numpy(dtype="float32")
predispatch_1  = df["Hour 1 Predispatch"].to_numpy(dtype="float32")
predispatch_2  = df["Hour 2 Predispatch"].to_numpy(dtype="float32")
predispatch_3  = df["Hour 3 Predispatch"].to_numpy(dtype="float32")


actual_1       = actual
rmse_1 = np.sqrt(mean_squared_error(actual_1, predispatch_1))
mae_1  = mean_absolute_error(actual_1, predispatch_1)
print(f"Hour-1 Predispatch RMSE: {rmse_1:.2f} CAD/MWh")
print(f"Hour-1 Predispatch MAE : {mae_1:.2f} CAD/MWh")


actual_2       = actual
rmse_2 = np.sqrt(mean_squared_error(actual_2, predispatch_2))
mae_2  = mean_absolute_error(actual_2, predispatch_2)
print(f"Hour-2 Predispatch RMSE: {rmse_2:.2f} CAD/MWh")
print(f"Hour-2 Predispatch MAE : {mae_2:.2f} CAD/MWh")


actual_3       = actual
rmse_3 = np.sqrt(mean_squared_error(actual_3, predispatch_3))
mae_3  = mean_absolute_error(actual_3, predispatch_3)
print(f"Hour-3 Predispatch RMSE: {rmse_3:.2f} CAD/MWh")
print(f"Hour-3 Predispatch MAE : {mae_3:.2f} CAD/MWh")
