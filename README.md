# HOEP Forecasting App

This project forecasts Ontario’s Hourly Ontario Energy Price (HOEP) using weather data, electricity demand, and operating reserve metrics. It uses a neural network trained on historical data and performs live inference using real-time public APIs.

---

##  Project Summary

- Forecast HOEP up to 1 hour ahead
- Use only publicly available real-time features
- Achieve low RMSE suitable for production use
- Support future deployment as a forecasting API or dashboard

---

## Model Summary

- **Type**: Feedforward Neural Network (2 hidden layers)
- **Input Features**: Demand, Price Lagged features, operating reserves, weather, lagged values, and time encodings
- **Loss**: MSE
- **Optimizer**: Adam
- **Frameworks**: TensorFlow, scikit-learn

---
##  Folder Structure

hoep_forecasting_app/
│
├── data/
│   └── raw/
│       ├── weather/               # Historical weather CSVs
│       └── hoep_demand.csv        # HOEP & demand CSVs
│
├── models/
│   ├── hoep_nn_weather.keras      # Trained neural network
│   └── feature_scaler.pkl         # Saved StandardScaler
│
├── src/
│   ├── data_loader.py             # Loading & merging raw data
│   ├── feature_engineering.py     # Lag, rolling, time features
│   └── live_fetchy.py           # Live API fetch for real-time inference
│
├── tests/
│   └── test_data_features.py      # Pipeline test script
│
├── train.py                       # Model training script
└── README.md

---

## Input Features

| Feature             | Description                        |
|---------------------|------------------------------------|
| Hour 1 Predispatch  | Market pre-dispatch price          |
| OR 10 Min Sync      | Operating reserve 10-minute (sync) |
| OR 30 Min           | Operating reserve 30-minute        |
| Ontario Demand      | Total demand (MW)                  |
| temp, humidity, wind| Weather data from Open-Meteo       |
| hour_sin/cos        | Hour of day (cyclical encoding)    |
| HOEP_lag_1/2/3      | Past HOEP values (1–3 hr lag)      |
| HOEP_ma_3           | 3-hour HOEP rolling average        |
| Demand_lag_1/2/3    | Past demand values                 |
| Demand_ma_3         | 3-hour demand moving average       |

---

##  Model Performance

- **Train/Test Split**: 80/20 (no shuffle)
- **RMSE**: ~ **0.37 CAD/MWh** (on 2025 data)
- **Scaler**: StandardScaler (saved as .pkl)

---
## 🔬 Ablation Study

To understand the contribution of different feature groups, we trained the same model with specific features removed. The RMSE (Root Mean Squared Error) was used to evaluate performance on a held-out test set.

| Experiment                  | Features Removed                        | RMSE (↓ better) |
|----------------------------|------------------------------------------|-----------------|
| **Full Model**             | –                                        | **0.37**        |
| No Weather                 | `temp`, `humidity`, `wind_speed`         | 1.97            |
| No HOEP Lag/Rolling       | `HOEP_lag_1/2/3`, `HOEP_ma_3`            | 18.23           |
| No Demand Lag/Rolling     | `Demand_lag_1/2/3`, `Demand_ma_3`        | 13.00           |

---

> Live Inference (Supported)

Live fetch script uses:
- [IESO RealtimeTotals](hhttps://reports-public.ieso.ca/public/RealtimeOntarioZonalPrice/PUB_RealtimeOntarioZonalPrice.xml)
- [Open-Meteo API](https://open-meteo.com/)

To simulate real-time inference with proper lags and scaling.

---

