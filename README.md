# HOEP Forecasting App ⚡
---

A real-time electricity price forecasting application for the Hourly Ontario Energy Price (HOEP) using machine learning and live data integration.

---

## 🚀 Live Demo

**Try it now**: [Ontario Electricity Forecasting App](https://hoep-forecasting-app.onrender.com/)

The live application provides:
- Real-time price forecasts with uncertainty quantification
- 24-hour performance tracking with actual vs predicted comparisons  
- Manual prediction capabilities
- Live countdown to next automated forecast

---
## Application Features

### Main Dashboard
![Homepage](assets/home_page)

- **Live Countdown**: Timer to next prediction at the 56th minute when MCPs finalize
- **Current Forecast**: $XX.XX median price with 80% confidence band ($XX.XX - $XX.XX)
- **Price Trends**: Interactive 24-hour HOEP chart showing market volatility (scroll below)

### Performance Analytics Dashboard
![Actual vs Predicted](assets/plot)

- **Prediction vs Actual**: 24-hour rolling comparison with actual HOEP prices
- **Model Performance**: Quantile coverage analysis (target: 80%, actual: XX%)
- **Error Metrics**: Mean Absolute Error tracking for forecast accuracy

### Manual Prediction Interface
![Manual Prediction](assets/manualprediction)

- **On-Demand Forecasting**: Generate predictions outside the regular hourly schedule
- **Real-Time Data**: Uses latest market and weather data for forecasts
- **Results**: Q10, Q50, Q90 predictions with confidence intervals
---

## Technical Architecture

**System Overview**
```
Google Cloud Scheduler → Cloud Function (cloud_entry) → Live Data APIs → Quantile Models → GitHub → Streamlit Dashboard
```

### Key Components

**Data Pipeline**
- **IESO APIs**: Real-time electricity demand, zonal pricing, market data
- **Weather Integration**: Open-Meteo API for Toronto conditions
- **Feature Engineering**: 31 features including lags, rolling averages, temporal encoding

**Machine Learning Models**
- **Quantile Regression**: Separate neural networks for Q10, Q50, Q90 predictions
- **Custom Loss Functions**: Asymmetric quantile loss for uncertainty quantification (pinball loss)
- **Training Data**: 10+ years of historical HOEP, demand, and weather data

**Production Infrastructure**
- **Google Cloud Functions**: Serverless execution triggered every hour at 55 minutes
- **Timing Logic**: Predictions generated after Market Clearing Prices finalize (T+2 forecasting)
- **Data Persistence**: CSV buffers updated and synced via GitHub for Streamlit consumption 


**Core Training Pipeline**
- `src/data_loader.py`: Data preprocessing and merging from raw CSVs
- `src/feature_engineering.py`: Lag and rolling window calculations
- `src/quantile_model.py`: Custom loss functions and model architecture
- `train.py`: Full model training with quantile regression


**Live Prediction System**
- `cloud_entry/live_prediction.py`: Main prediction pipeline for Cloud Functions
- `cloud_entry/src/live_fetch.py`: Real-time data collection from APIs
- `cloud_entry/src/live_engineering.py`: Live feature calculation and scaling

**Web Application**
- `app/main.py`: Streamlit dashboard with live updates
- `app/pages/dashboard.py`: Performance analytics and model evaluation
- `app/pages/manual_prediction.py`: On-demand forecasting interface
- `app/pages/methodology.py`: Deeper dive into data preprocessing, model architecture, and deployment pipeline

</details>

---

## Model Performance

**Metric Results**
- **Median Forecast RMSE**: 24.17 CAD/MWh (2024 test data)
- **Hour-2 Predispatch RMSE**: 29.67 CAD/MWh (baseline comparison)
- **Performance Improvement**: 18.5% lower RMSE than IESO predispatch
- **Training Period**: 2014-2022, Validation: 2023, Test: 2024
- Model retrained on all available data (2014-2025) for deployment
  
---

## Future Enhancements

**Model Improvements**
- **Feature Selection**: Remove noisy variables identified in research
- **Architecture Optimization**: Explore ensemble methods and alternative quantile approaches
- **Multi-Horizon**: Extend to 1-6 hour forecasting
---

## Acknowledgments

- **IESO**: Independent Electricity System Operator for real-time market data
- **Open-Meteo**: Weather API for current conditions  
- **Environment Canada**: Historical weather datasets
- **Google Cloud**: Serverless infrastructure for automated predictions
- **Streamlit**: Rapid ML application deployment

---

## Author
- **Musa Elashaal**
---

*Research-driven electricity price forecasting with production deployment*
