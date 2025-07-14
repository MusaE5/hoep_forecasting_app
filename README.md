# HOEP Forecasting App

This app forecasts Ontario’s Hourly Ontario Energy Price (HOEP) 1 hour in advance using real-time public data. It uses a neural network trained on 2024–2025 data and performs live inference using APIs from IESO and Open-Meteo. The model produces **point forecasts** and **uncertainty estimates** via quantile regression.

---

##  Project Summary

- **Goal**: Predict HOEP 1 hour ahead using only live-accessible features
- **Approach**: Quantile regression (neural networks) for probabilistic forecasting
- **Application**: Real-time forecasting API or interactive dashboard
- **Deployment Ready**: All inputs can be pulled via API or web scraping in real-time

---

## 🔢 Input Features

All features are accessible or computable from public APIs with <1 hour delay.

| Category             | Features Included                                                              |
|----------------------|----------------------------------------------------------------------------------|
| **Time Encodings**   | `hour_sin`, `hour_cos`, `is_weekend`, `day_of_year`, `doy_sin`, `doy_cos`       |
| **HOEP Lags**        | `HOEP_lag_2`, `HOEP_lag_3`, `HOEP_lag_24`                                       |
| **Demand Lags**      | `demand_lag_2`, `demand_lag_3`, `demand_lag_24`                                 |
| **Demand MA**        | `demand_ma_3`, `demand_ma_24`                                                   |
| **Weather Lags**     | `temp/humidity/wind_speed_lag_2/3/24`                                           |
| **Weather MA**       | `temp/humidity/wind_speed_ma_3/24`                                              |
| **Operating Reserve**| `OR_30_Min_lag_2/3/24`, `OR_10_Min_sync_lag2/3/24`, `OR_10_Min_non-sync_lag2/3/24`|

**Total Features**: 39  
**Scaling**: StandardScaler (saved as `.pkl`)

---

## 📊 Model Summary

- **Architecture**: Separate neural network per quantile (q10, q50, q90)
- **Loss**: Quantile loss (pinball loss)
- **Frameworks**: TensorFlow + scikit-learn
- **Inference**: Fully automated using real-time API pipelines

---

##  Model Performance (2024–2025 Hour-1 Forecast)

| Metric                  | Value              |
|-------------------------|--------------------|
| **RMSE (q_50)**         | **33.94 CAD/MWh**  |
| **MAE (q_50)**          | 11.19 CAD/MWh      |
| **R² (q_50)**           | 0.287              |
| **80% Interval Width**  | ~5–6 CAD/MWh       |
| **Hour-1 Predispatch RMSE** | 39.50 CAD/MWh |
| **XGBoost RMSE (q_50)** | 34.35 CAD/MWh      |

---

##  Quantile Regression Coverage

| Quantile | Expected Coverage | Actual Coverage | RMSE   |
|----------|-------------------|------------------|--------|
| q_10     | 10%               | 7.3%             | 38.36  |
| q_50     | 50%               | 44.4%            | 33.94  |
| q_90     | 90%               | 85.8%            | 38.02  |

---

## 🌐 Live Inference

- **Weather**: [Open-Meteo API](https://open-meteo.com/)
- **Demand + HOEP**: [IESO Real-Time Reports](https://www.ieso.ca/en/Power-Data/Data-Directory)
- **Lag Simulation**: Uses hourly polling & aggregation to simulate input availability
- **Prediction**: 1-hour ahead price & 80% confidence interval

---


