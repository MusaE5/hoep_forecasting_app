import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
import joblib

from src.data_loader import load_hoep_and_demand, load_weather, merge_all
from src.feature_engineering import create_features

# Load and merge data
RAW_HOEP_DIR    = "data/raw"
RAW_WEATHER_DIR = "data/raw/weather"

hd_df      = load_hoep_and_demand(RAW_HOEP_DIR)
weather_df = load_weather(RAW_WEATHER_DIR)
merged_df  = merge_all(hd_df, weather_df)
full_df    = create_features(merged_df)
full_df.columns = full_df.columns.str.strip()

# Time based, no Train test split
df_train = full_df[full_df["timestamp"] < "2024-01-01"]
df_test  = full_df[full_df["timestamp"] >= "2024-01-01"]

# Select features
features = [
    "hour_sin", "hour_cos", "is_weekend",
    "day_of_year", "doy_sin", "doy_cos",
    "demand_lag_2", "demand_lag_3", "demand_lag_24",
    "temp_lag_2", "temp_lag_3", "temp_lag_24",
    "humidity_lag_2", "humidity_lag_3", "humidity_lag_24",
    "wind_speed_lag_2", "wind_speed_lag_3", "wind_speed_lag_24",
    "HOEP_lag_2", "HOEP_lag_3", "HOEP_lag_24",
    "demand_ma_3", "demand_ma_23",
    "temp_ma_3", "temp_ma_23",
    "humidity_ma_3", "humidity_ma_23",
    "wind_speed_ma_3", "wind_speed_ma_23",
    "HOEP_ma_3", "HOEP_ma_23"
]

target = "HOEP"

X_train = df_train[features].apply(pd.to_numeric, errors="coerce")
y_train     = pd.to_numeric(df_train[target], errors="coerce")
X_test  = df_test[features].apply(pd.to_numeric, errors="coerce")
y_test      = pd.to_numeric(df_test[target], errors="coerce")



# XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\nXGBoost Performance")
print(f"RMSE: {rmse:.2f} CAD/MWh")
print(f"MAE:  {mae:.2f} CAD/MWh")
print(f"R2:   {r2:.3f}\n")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/hoep_xgb_model.pkl")
joblib.dump(scaler, "models/feature_scaler_xgb.pkl")
print("Model and scaler saved in /models")