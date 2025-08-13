import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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


X_train_raw = df_train[features].apply(pd.to_numeric, errors="coerce")
y_train     = pd.to_numeric(df_train[target], errors="coerce")
X_test_raw  = df_test[features].apply(pd.to_numeric, errors="coerce")
y_test      = pd.to_numeric(df_test[target], errors="coerce")


# Leak check (See highly correlated features)
corr_df = pd.concat([X_train_raw, y_train.rename('HOEP')], axis=1).corr()
print("=== Top Feature Correlations with HOEP ===")
print(corr_df['HOEP'].abs().sort_values(ascending=False).head(10), "\n")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)



# Add helper functions to train quantiles on same scaled features
from src.quantile_model import (
    train_quantile_models,
    evaluate_quantile_predictions,
    calculate_prediction_intervals,
    plot_quantile_predictions,
    save_quantile_models
)


print("Training quantile regression models...")
quantile_predictions, quantile_models = train_quantile_models(
    X_train, y_train, X_test, y_test
)

print("\nEvaluating quantile predictions...")
results = evaluate_quantile_predictions(y_test, quantile_predictions)

print("\n=== Quantile Regression Results ===")
for q_name, metrics in results.items():
    print(f"\nQuantile {q_name}:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  Pinball Loss: {metrics['pinball_loss']:.2f}")
    print(f"  Coverage: {metrics['coverage']:.3f} (expected: {metrics['expected_coverage']:.3f})")

# Calculate prediction intervals
intervals = calculate_prediction_intervals(quantile_predictions)

print("\n=== Prediction Intervals ===")
for interval_name, interval_data in intervals.items():
    avg_width = np.mean(interval_data['width'])
    print(f"{interval_name} interval average width: {avg_width:.2f} CAD/MWh")

# Plot results
plot_quantile_predictions(y_test, quantile_predictions)

print(f"Median quantile RMSE: {results['q_50']['rmse']:.2f}")

# Define config
features_list = features  
save_quantile_models(quantile_models, scaler, features_list, model_dir="models")
