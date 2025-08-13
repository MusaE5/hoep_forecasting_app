# HOEP Forecasting App ⚡

A real-time electricity price forecasting application for Ontario's Hourly Ontario Energy Price (HOEP) using machine learning and live data integration.

---

## 📚 Research Background

This application is an offshoot of my ongoing electricity market research, prioritizing **uncertainty quantification over speed** using feed-forward neural networks with quantile predictions. Rather than pursuing millisecond inference times, this system focuses on providing robust confidence intervals for price forecasts, which is crucial for risk management in electricity markets.

The research-to-production pipeline demonstrates end-to-end deployment of machine learning research, taking academic concepts through to live, automated forecasting systems with real market data.

*Research paper link will be added upon publication.*

---

## 🚀 Live Demo

**Try it now**: [Ontario Electricity Forecasting App](https://ontarioelectricityforecasting.streamlit.app/)

The live application provides:
- Real-time price forecasts with uncertainty quantification
- 24-hour performance tracking with actual vs predicted comparisons  
- Manual prediction capabilities
- Live countdown to next automated forecast

---

## 📸 Application Features

### Main Dashboard
*[Screenshot: Live countdown timer, median forecast with confidence bands, 24-hour price chart]*

- **Live Countdown**: Timer to next prediction at the 56th minute when MCPs finalize
- **Current Forecast**: $XX.XX median price with 80% confidence band ($XX.XX - $XX.XX)
- **Price Trends**: Interactive 24-hour HOEP chart showing market volatility

### Performance Analytics Dashboard
*[Screenshot: Quantile accuracy chart, coverage metrics, MAE calculations]*

- **Prediction vs Actual**: 24-hour rolling comparison with actual HOEP prices
- **Model Performance**: Quantile coverage analysis (target: 80%, actual: XX%)
- **Error Metrics**: Mean Absolute Error tracking for forecast accuracy

### Manual Prediction Interface  
*[Screenshot: Manual trigger interface, immediate prediction results]*

- **On-Demand Forecasting**: Generate predictions outside the regular hourly schedule
- **Real-Time Data**: Uses latest market and weather data for immediate forecasts
- **Instant Results**: Q10, Q50, Q90 predictions with confidence intervals

---

## 🏗️ Technical Architecture

### System Overview
```
Google Cloud Scheduler → Cloud Function → Live Data APIs → Quantile Models → GitHub → Streamlit Dashboard
```

### Key Components

**Data Pipeline**
- **IESO APIs**: Real-time electricity demand, zonal pricing, market data
- **Weather Integration**: Open-Meteo API for Toronto conditions
- **Feature Engineering**: 31 features including lags, rolling averages, temporal encoding

**Machine Learning Models**
- **Quantile Regression**: Separate neural networks for Q10, Q50, Q90 predictions
- **Custom Loss Functions**: Asymmetric quantile loss for uncertainty quantification
- **Training Data**: 10+ years of historical HOEP, demand, and weather data

**Production Infrastructure**
- **Google Cloud Functions**: Serverless execution triggered every hour at 55 minutes
- **Timing Logic**: Predictions generated after Market Clearing Prices finalize (T+2 forecasting)
- **Data Persistence**: CSV buffers updated and synced via GitHub for Streamlit consumption

<details>
<summary>Detailed Implementation Files</summary>

**Core Training Pipeline**
- `train.py`: Full model training with quantile regression
- `src/feature_engineering.py`: Lag and rolling window calculations
- `src/quantile_model.py`: Custom loss functions and model architecture

**Live Prediction System**
- `cloud_entry/live_prediction.py`: Main prediction pipeline for Cloud Functions
- `cloud_entry/src/live_fetch.py`: Real-time data collection from APIs
- `cloud_entry/src/live_engineering.py`: Live feature calculation and scaling

**Web Application**
- `app/main.py`: Streamlit dashboard with live updates
- `app/pages/dashboard.py`: Performance analytics and model evaluation
- `app/pages/manual_prediction.py`: On-demand forecasting interface

</details>

---

## 📊 Model Performance

### Quantile Regression Results
- **80% Confidence Intervals**: Effective uncertainty quantification for risk management
- **Coverage Analysis**: Actual prices fall within Q10-Q90 bands ~80% of target time
- **Median Forecast**: Competitive accuracy with IESO predispatch prices
- **Real-time Validation**: Continuous performance tracking with live market data

### Training & Validation
- **Training Period**: 2014-2025 historical data (10+ years)
- **Time-based Splits**: Chronological train/test to prevent data leakage
- **Live Performance**: Updated hourly with actual market outcomes
- **Feature Count**: 31 engineered features after preprocessing

---

## 🔮 Future Enhancements

### Model Improvements
- **Feature Selection**: Remove noisy variables identified in ongoing research
- **Architecture Optimization**: Explore ensemble methods and alternative quantile approaches
- **Multi-Horizon**: Extend to 6, 12, and 24-hour forecasts

### System Enhancements  
- **API Development**: RESTful endpoints for external integration
- **Enhanced Monitoring**: MLOps pipeline with model drift detection
- **Real-time Updates**: Online learning capabilities with streaming data

---

## 🛠️ Development Setup

<details>
<summary>For Contributors: Local Development</summary>

### Prerequisites
- Python 3.10+
- Google Cloud Account
- Git

### Local Setup
```bash
git clone <your-repo-url>
cd hoep_forecasting_app
pip install -r requirements.txt
streamlit run app/main.py
```

### Model Training
```bash
python train.py  # Full retraining pipeline
python tests/test_baseline_rmse.py  # Performance validation
```

### Cloud Function Deployment
1. Deploy `cloud_entry/` directory to Google Cloud Functions
2. Set up Cloud Scheduler for hourly triggers at 55 minutes
3. Configure GitHub token for CSV repository updates

</details>

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **IESO**: Independent Electricity System Operator for real-time market data
- **Open-Meteo**: Weather API for current conditions  
- **Environment Canada**: Historical weather datasets
- **Google Cloud**: Serverless infrastructure for automated predictions
- **Streamlit**: Rapid ML application deployment

---

## 📞 Contact

- **Live Application**: [Ontario Electricity Forecasting](https://ontarioelectricityforecasting.streamlit.app/)
- **Project Repository**: [GitHub](https://github.com/yourusername/hoep_forecasting_app)

---

*Research-driven electricity price forecasting with production deployment*
