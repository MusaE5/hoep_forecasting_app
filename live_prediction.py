import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.quantile_model import quantile_loss, load_quantile_models
from src.live_fetch import fetch_realtime_totals, fetch_current_weather, fetch_and_store, append_to_buffer, get_ontario_zonal_average
from src.live_engineering import load_scaler, load_buffer, calculate_features, process_new_data
import time



if __name__ == "__main__":

    feat = fetch_and_store()
    if feat is not None:
        actual_hoep = feat['zonal_price']

    df = pd.read_csv(data/hoep_buffer.csv)
    features_dict = calculate_features(df)
    scaled_features = process_new_data(features_dict)


    models = load_quantile_models()

    predictions = {
        'q10': models['q10'].predict(scaled_features, verbose=0)[0][0],
        'q50': models['q50'].predict(scaled_features, verbose=0)[0][0],
        'q90': models['q90'].predict(scaled_features, verbose=0)[0][0]
    }

    # Set up dictionary for appending
    predicted_for = pd.to_datetime(df['timestamp'].iloc[-1]).ceil('h') + timedelta(hours=1)
    new_entry = {
        "predicted_for_hour": predicted_for.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_q10": predictions['q10'],
        "pred_q50": predictions['q50'],
        "pred_q90": predictions['q90'],
        "timestamp_predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actual_hoep": None

    }
    

    log_url = "https://raw.githubusercontent.com/MusaE5/hoep_forecasting_app/data-updates/data/predictions_log.csv"
    log_df = pd.read_csv(log_url)
    log_path = "data/predictions_log.csv"

    

    if len(log_df) == 3:
            # Inject actual HOEP into the middle row
        if pd.isna(log_df.loc[1, 'actual_hoep']):
            log_df.loc[1, 'actual_hoep'] = actual_hoep
            print(f"✅ Injected actual HOEP {actual_hoep:.2f} into predicted_for_hour {log_df.loc[1, 'predicted_for_hour']}")
        else:
            print("Middle row already has actual HOEP — skipping injection.")

        # Drop the oldest row (index 0)
        log_df = log_df.iloc[1:]

 

    # Step 4: Append new prediction row
    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)

    numeric_cols = ['pred_q10', 'pred_q50', 'pred_q90']
    log_df[numeric_cols] = log_df[numeric_cols].round(2)

    log_df.to_csv(log_path, index=False)
   
    


    chart_url_path = "https://raw.githubusercontent.com/MusaE5/hoep_forecasting_app/data-updates/data/chart_buffer.csv"
    chart_df = pd.read_csv(chart_url_path)
    chart_buffer_path = "data/chart_buffer.csv"
    
    chart_cols = ["predicted_for_hour", "pred_q10", "pred_q50", "pred_q90", "timestamp_predicted_at", "actual_hoep"]
    chart_entry = {k: new_entry[k] for k in chart_cols if k in new_entry}

    # Round float predictions to 2 decimals before saving
    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        if col in chart_entry and pd.notna(chart_entry[col]):
            chart_entry[col] = round(float(chart_entry[col]), 2)

    # Append
    chart_df = pd.concat([chart_df, pd.DataFrame([chart_entry])], ignore_index=True)

    # Keep only the latest 26 entries
    chart_df = chart_df.tail(26).reset_index(drop=True)

    # Inject actual_hoep into the 3rd-last row
    if len(chart_df) >= 3:
        chart_df.loc[-3 % len(chart_df), "actual_hoep"] = round(actual_hoep, 2)

    # Save
    chart_df.to_csv(chart_buffer_path, index=False)
