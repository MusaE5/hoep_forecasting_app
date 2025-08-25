import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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

# Time-based splits
df_train = full_df[full_df["timestamp"] < "2023-01-01"] # up to 2022
df_val   = full_df[(full_df["timestamp"] >= "2023-01-01") & (full_df["timestamp"] < "2024-01-01")] # 2023
df_test  = full_df[full_df["timestamp"] >= "2024-01-01" % (full_df['timestamp'] < '2025-01-01')] # 2024

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

X_train_raw = df_train[features]
y_train     = df_train[target]
X_val_raw   = df_val[features]
y_val       = df_val[target]
X_test_raw  = df_test[features]
y_test      = df_test[target]

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

from src.quantile_model import train_quantile_models, evaluate_quantile_predictions, save_quantile_models

quantile_predictions, quantile_models = train_quantile_models(
    X_train, y_train, X_test, y_test
)

results = evaluate_quantile_predictions(y_test, quantile_predictions)

print("\nQuantile Regression Results ")
for q_name, metrics in results.items():
    print(f"\nQuantile {q_name}:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  Pinball Loss: {metrics['pinball_loss']:.2f}")
    print(f"  Coverage: {metrics['coverage']:.3f} (expected: {metrics['expected_coverage']:.3f})")

save_quantile_models(quantile_models, scaler, model_dir="models")
