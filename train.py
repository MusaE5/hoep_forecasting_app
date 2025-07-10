import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
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
full_df.columns = full_df.columns.str.strip()

# --- Time-based split ---
df_train = full_df[full_df["timestamp"] < "2024-01-01"]
df_test  = full_df[full_df["timestamp"] >= "2024-01-01"]

# --- Select features and target ---
features = [
    "hour_sin", "hour_cos", "is_weekend",
    "day_of_year", "doy_sin", "doy_cos",
    "demand_lag_2", "demand_lag_3", "demand_lag_24",
    "temp_lag_2", "temp_lag_3", "temp_lag_24",
    "humidity_lag_2", "humidity_lag_3", "humidity_lag_24",
    "wind_speed_lag_2", "wind_speed_lag_3", "wind_speed_lag_24",
    "HOEP_lag_2", "HOEP_lag_3", "HOEP_lag_24",
    "demand_ma_3", "demand_ma_24",
    "temp_ma_3", "temp_ma_24",
    "humidity_ma_3", "humidity_ma_24",
    "wind_speed_ma_3", "wind_speed_ma_24",
    "HOEP_ma_3", "HOEP_ma_24",
    "Hour 1 Predispatch", "OR 10 Min Sync", "OR 30 Min"
]
target = "HOEP"

# --- Build features and targets ---
X_train_raw = df_train[features].apply(pd.to_numeric, errors="coerce")
y_train     = pd.to_numeric(df_train[target], errors="coerce")
X_test_raw  = df_test[features].apply(pd.to_numeric, errors="coerce")
y_test      = pd.to_numeric(df_test[target], errors="coerce")


# --- Leak check ---
corr_df = pd.concat([X_train_raw, y_train.rename('HOEP')], axis=1).corr()
print("=== Top Feature Correlations with HOEP ===")
print(corr_df['HOEP'].abs().sort_values(ascending=False).head(10), "\n")

# --- Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# --- Build and Train NN ---
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64),
    LeakyReLU(alpha=0.01),
    Dense(32),
    LeakyReLU(alpha=0.01),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss="mse")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --- Evaluate ---
y_pred = model.predict(X_test, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\nNeural Network Performance ")
print(f"RMSE: {rmse:.2f} CAD/MWh")
print(f"MAE:  {mae:.2f} CAD/MWh")
print(f"R2:   {r2:.3f}\n")


# --- Save model and scaler ---
os.makedirs("models", exist_ok=True)
model.save("models/hoep_nn_weather.keras")
joblib.dump(scaler, "models/feature_scaler.pkl")
print("Model and scaler saved in /models")
