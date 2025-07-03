import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from src.data_loader import (
    load_hoep_and_demand,
    load_weather,
    merge_all,
)
from src.feature_engineering import create_features

# Load and merge data
RAW_HOEP_DIR    = "data/raw"
RAW_WEATHER_DIR = "data/raw/weather"

hd_df      = load_hoep_and_demand(RAW_HOEP_DIR)
weather_df = load_weather(RAW_WEATHER_DIR)
merged_df  = merge_all(hd_df, weather_df)
full_df    = create_features(merged_df)

# Select final feature list
features = [
    "Hour 1 Predispatch",
    "OR 10 Min Sync",
    "OR 30 Min",
    "Ontario Demand",
    "temp",
    "humidity",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "Demand_lag_1",
    "Demand_lag_2",
    "Demand_lag_3",
    "Demand_ma_3",
    "HOEP_lag_1",
    "HOEP_lag_2",
    "HOEP_lag_3",
    "HOEP_ma_3"
]

target = "HOEP"

full_df.columns = full_df.columns.str.strip()  # remove spaces

# Check every requested feature exists
missing = [col for col in features if col not in full_df.columns]
if missing:
    raise ValueError(f"Missing columns in DataFrame: {missing}")

# ---- 2. Force every feature & target numeric ----------------------------
X = full_df[features].apply(pd.to_numeric, errors="coerce")
#=== Inspect Features Before Training ===
print("\n=== Training Features ===")
print("Features:", features)
print("\nColumn Types:\n", X.dtypes.value_counts())
print("\nSummary Stats:\n", X.describe().T)
print("\nMissing values per column:\n", X.isna().sum())
y = pd.to_numeric(full_df[target], errors="coerce")

# Drop any rows with NaNs created by coercion
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# ---- 3. Scale and split --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.to_numpy(dtype="float32"), test_size=0.2, shuffle=False
)

# ---- 4. Build & train NN -------------------------------------------------

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss="mse")


model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
)
# ---- 5. Evaluate ---------------------------------------------------------
y_pred = model.predict(X_test, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE: {rmse:.2f} CAD/MWh")

# ---- 6. Save -------------------------------------------------------------

os.makedirs("models", exist_ok=True)
model.save("models/hoep_nn_weather.keras")
joblib.dump(scaler, "models/feature_scaler.pkl")
print("Model and scaler saved in /models")