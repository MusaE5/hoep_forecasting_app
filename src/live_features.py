import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import os
import tensorflow as tf
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

def load_quantile_models():
    """Load all three quantile models with custom loss functions"""
    model_dir = PROJECT_ROOT / "models"
    
    custom_objects_10 = {'loss': quantile_loss(0.1)}
    custom_objects_50 = {'loss': quantile_loss(0.5)}  
    custom_objects_90 = {'loss': quantile_loss(0.9)}
    
    models = {
        'q10': load_model(model_dir / "hoep_quantile_q_10.keras", custom_objects=custom_objects_10),
        'q50': load_model(model_dir / "hoep_quantile_q_50.keras", custom_objects=custom_objects_50), 
        'q90': load_model(model_dir / "hoep_quantile_q_90.keras", custom_objects=custom_objects_90)
    }
    
    return models

PROJECT_ROOT = Path(__file__).parent.parent  # Goes up two levels from src/

# Helper functions
def load_scaler():
    scaler_path = PROJECT_ROOT / "models" / "feature_scaler.pkl"
    return joblib.load(scaler_path)


def load_buffer(buffer_file):
    """Loads existing data buffer"""
    if Path(buffer_file).exists():
        return pd.read_csv(buffer_file)
    return pd.DataFrame(columns=[
        'timestamp', 'demand_MW', 
        'temp_C', 
        'humidity_%', 'wind_mps', 'zonal_price'
    ])


def calculate_features(buffer_df):
    """
    Creates features PREDICTING t+1 (1-hour ahead).
    Uses data up to t-2 to forecast t+1.
    """
    # Time features (unchanged)
    hour_col = buffer_df['hour']
    current_hour = hour_col.iloc[-1] 
    prediction_hour = (current_hour + 2) % 24
    time_features = {
        'hour_sin': np.sin(2 * np.pi * prediction_hour / 24),
        'hour_cos': np.cos(2 * np.pi * prediction_hour / 24)
    }

   # Work on a copy to avoid modifying original data
    df = buffer_df.copy()

    # Convert and clean timestamp
    timestamp = pd.to_datetime(df['timestamp'])
    target_timestamp = timestamp.dt.ceil('H') + pd.Timedelta(hours=1)

    # Keep as datetime object for features
    df['timestamp'] = target_timestamp

    # Create time features (excluding hour sin/cos since you use hour column)
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

 
    
    # Price features (shift lags +1 vs nowcasting)
    price_col = buffer_df['zonal_price']
    price_features = {
        'HOEP_lag_2': price_col.iloc[-1],  # Most recent price (just aquired)
        'HOEP_lag_3': price_col.iloc[-2],  # Hour before
        'HOEP_lag_24': price_col.iloc[-23],  
        'HOEP_ma_3': price_col.iloc[-3:].mean(),
        'HOEP_ma_23': price_col.iloc[-24:].mean()


    }
    
    # Demand features (same shift logic)
    demand_col = buffer_df['demand_MW']
    demand_features = {
        'demand_lag_2': demand_col.iloc[-1],  
        'demand_lag_3': demand_col.iloc[-2],  
        'demand_lag_24': demand_col.iloc[-23], 
        'demand_ma_3': demand_col.iloc[-3:].mean(),
        'demand_ma_23': demand_col.iloc[-23:].mean()  
    }

    wind_col = buffer_df['wind_mps'] *3.6
    wind_features = {
        'wind_speed_lag_2': wind_col.iloc[-1],  
        'wind_speed_lag_3': wind_col.iloc[-2],  
        'wind_speed_lag_24': wind_col.iloc[-23], 
        'wind_speed_ma_3': wind_col.iloc[-3:].mean(),
        'wind_speed_ma_23': wind_col.iloc[-23:].mean()  
    }

    temp_col = buffer_df['temp_C']
    temp_features = {
        'temp_lag_2': temp_col.iloc[-1],  
        'temp_lag_3': temp_col.iloc[-2],  
        'temp_lag_24': temp_col.iloc[-23], 
        'temp_ma_3': temp_col.iloc[-3:].mean(),
        'temp_ma_23': temp_col.iloc[-23:].mean()  
        }
    
    humid_col = buffer_df['humidity_%']
    humid_features = {
   'humidity_lag_2': humid_col.iloc[-1],          
   'humidity_lag_3': humid_col.iloc[-2],          
   'humidity_lag_24': humid_col.iloc[-23],         
   'humidity_ma_3': humid_col.iloc[-3:].mean(),        
   'humidity_ma_23': humid_col.iloc[-23:].mean()      
}
    
    
    return {
   **time_features,
   **price_features, 
   **demand_features,
   **wind_features,
   **temp_features,
   **humid_features,
   'is_weekend': df['is_weekend'].iloc[-1],
   'doy_sin': df['doy_sin'].iloc[-1],
   'day_of_year': df['day_of_year'].iloc[-1],
   'doy_cos': df['doy_cos'].iloc[-1]
    }

def process_new_data(features_dict):
   # Ensure features are in the SAME ORDER as training
   training_feature_order = [
       "hour_sin", "hour_cos", "is_weekend", "day_of_year", "doy_sin", "doy_cos",
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
   
   # Create ordered list from dictionary
   features_ordered = [features_dict[k] for k in training_feature_order]
   
   # Convert to DataFrame for scaler (scaler expects 2D input)
   features_df = pd.DataFrame([features_ordered], columns=training_feature_order)
   
   # Scale features
   scaler = load_scaler()
   scaled_features = scaler.transform(features_df)
   
   return scaled_features  # Returns 2D array ready for model.predict()


if __name__ == "__main__":
    from tensorflow.keras.models import load_model

    buffer_file = "data/hoep_buffer.csv"

    df = load_buffer(buffer_file)

    features_dict = calculate_features(df)

    scaled_features = process_new_data(features_dict)
    
    models = load_quantile_models()
    predictions = {
    'q10': models['q10'].predict(scaled_features, verbose=0)[0][0],
    'q50': models['q50'].predict(scaled_features, verbose=0)[0][0],
    'q90': models['q90'].predict(scaled_features, verbose=0)[0][0]
      }
    
    print("HOEP Predictions:")
    print(f"10th percentile: ${predictions['q10']:.2f}")
    print(f"50th percentile: ${predictions['q50']:.2f}")
    print(f"90th percentile: ${predictions['q90']:.2f}")
    print(predictions)

    
    