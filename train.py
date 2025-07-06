import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

from src.data_loader import load_hoep_and_demand, load_weather, merge_all
from src.feature_engineering import create_features

# --- Load and merge data ---
RAW_HOEP_DIR    = "data/raw"
RAW_WEATHER_DIR = "data/raw/weather"

hd_df      = load_hoep_and_demand(RAW_HOEP_DIR)
weather_df = load_weather(RAW_WEATHER_DIR)
merged_df  = merge_all(hd_df, weather_df)
full_df    = create_features(merged_df)

# --- Select features and target ---
feature_list = [
    "hour_sin","hour_cos","is_weekend","day_of_year","doy_sin","doy_cos",
    "demand_lag_1","demand_lag_2","demand_lag_24",
    "temp_lag_1","temp_lag_2","temp_lag_24",
    "humidity_lag_1","humidity_lag_2","humidity_lag_24",
    "wind_speed_lag_1","wind_speed_lag_2","wind_speed_lag_24",
    "HOEP_lag_1","HOEP_lag_2","HOEP_lag_24",
    "demand_ma_3","demand_ma_24","temp_ma_3","temp_ma_24",
    "humidity_ma_3","humidity_ma_24","wind_speed_ma_3","wind_speed_ma_24",
    "HOEP_ma_3","HOEP_ma_24",
    "Hour 1 Predispatch","OR 10 Min Sync","OR 30 Min"
]
target = "HOEP"

# --- Sanity-check before feature matrix ---
print("=== Data Preview ===")
print(full_df[["timestamp", target] + feature_list].head(), "\n")

# --- Build feature matrix and target ---
full_df.columns = full_df.columns.str.strip()
X = full_df[feature_list].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(full_df[target], errors="coerce")

# --- Leak check: top feature correlations ---
corr_df = pd.concat([X, y.rename('HOEP')], axis=1).corr()
print("=== Top Feature Correlations with HOEP ===")
print(corr_df['HOEP'].abs().sort_values(ascending=False).head(10), "\n")

# --- Drop NaNs ---
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# --- Scale and split ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.to_numpy(dtype="float32"), test_size=0.2, shuffle=False
)

# --- Build and train NN ---
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss="mse")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- Evaluate ---
y_pred = model.predict(X_test, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f} CAD/MWh")
print(f"MAE:  {mae:.2f} CAD/MWh")
print(f"R2:   {r2:.3f}\n")

# --- Diagnostic Plots ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.title("HOEP: Actual vs Predicted")
plt.xlabel("Actual HOEP")
plt.ylabel("Predicted HOEP")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Save model and scaler ---
os.makedirs("models", exist_ok=True)
model.save("models/hoep_nn_weather.keras")
joblib.dump(scaler, "models/feature_scaler.pkl")
print("Model and scaler saved in /models")
