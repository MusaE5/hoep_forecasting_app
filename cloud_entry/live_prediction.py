import os
import base64
import requests
import pandas as pd
from datetime import timedelta, datetime
import pytz
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.quantile_model import quantile_loss, load_quantile_models
from src.live_fetch import fetch_and_store
from src.live_engineering import load_buffer, calculate_features, process_new_data

# Set timezone
TORONTO_TZ = pytz.timezone('America/Toronto')

# GitHub repo info
REPO = "MusaE5/hoep_forecasting_app"
BRANCH = "main"
GH_API = "https://api.github.com"

def gh_headers():
    return {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json"
    }

def gh_get(path_in_repo: str) -> bytes:
    """Download a file from GitHub repo."""
    url = f"{GH_API}/repos/{REPO}/contents/{path_in_repo}?ref={BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    return base64.b64decode(r.json()["content"])

def gh_put(path_in_repo: str, local_bytes: bytes, message: str):
    """Upload/update a file in GitHub repo."""
    url = f"{GH_API}/repos/{REPO}/contents/{path_in_repo}"
    sha = None
    existing = requests.get(url, headers=gh_headers(), timeout=30)
    if existing.status_code == 200:
        sha = existing.json()["sha"]

    data = {
        "message": message,
        "content": base64.b64encode(local_bytes).decode(),
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=data, timeout=30)
    r.raise_for_status()
    return r.json()


# Use /tmp in Cloud Functions, "data" locally

TMP = "/tmp" if os.path.isdir("/tmp") else "data"

if __name__ == "__main__":

    # Step 2: Download latest CSVs from GitHub into TMP
    for repo_path in ["cloud_entry/data/hoep_buffer.csv", "cloud_entry/data/predictions_log.csv", "cloud_entry/data/chart_buffer.csv"]:
        content = gh_get(repo_path)
        with open(os.path.join(TMP, os.path.basename(repo_path)), "wb") as f:
            f.write(content)

    # Step 1: Fetch and append live HOEP/weather data
    feat = fetch_and_store()
    actual_hoep = feat['zonal_price'] if feat is not None else None        

    # Step 3: Load buffer & prepare features
    buffer_file = os.path.join(TMP, "hoep_buffer.csv")
    df = load_buffer(buffer_file)
    features_dict = calculate_features(df)
    scaled_features = process_new_data(features_dict)

    # Step 4: Load models & make predictions
    models = load_quantile_models()
    predictions = {
        'q10': models['q10'].predict(scaled_features, verbose=0)[0][0],
        'q50': models['q50'].predict(scaled_features, verbose=0)[0][0],
        'q90': models['q90'].predict(scaled_features, verbose=0)[0][0]
    }

    # Step 5: Prepare new prediction entry
    predicted_for = pd.to_datetime(df['timestamp'].iloc[-1]).ceil('h') + timedelta(hours=1)
    toronto_now = datetime.now(TORONTO_TZ)
    new_entry = {
        "predicted_for_hour": predicted_for.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_q10": predictions['q10'],
        "pred_q50": predictions['q50'],
        "pred_q90": predictions['q90'],
        "timestamp_predicted_at": toronto_now.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_hoep": None
    }

   
    # Update predictions_log.csv
    
    log_path = os.path.join(TMP, "predictions_log.csv")
    log_df = pd.read_csv(log_path)

    if len(log_df) == 3:
        if pd.isna(log_df.loc[1, 'actual_hoep']) and actual_hoep is not None:
            log_df.loc[1, 'actual_hoep'] = actual_hoep
            print(f" Injected actual HOEP {actual_hoep:.2f} into predicted_for_hour {log_df.loc[1, 'predicted_for_hour']}")
        log_df = log_df.iloc[1:]

    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
    log_df[['pred_q10', 'pred_q50', 'pred_q90']] = log_df[['pred_q10', 'pred_q50', 'pred_q90']].round(2)
    log_df.to_csv(log_path, index=False)

    # Update chart_buffer.csv
    chart_buffer_path = os.path.join(TMP, "chart_buffer.csv")
    chart_df = pd.read_csv(chart_buffer_path)
    chart_entry = {k: new_entry[k] for k in chart_df.columns if k in new_entry}

    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        if col in chart_entry and pd.notna(chart_entry[col]):
            chart_entry[col] = round(float(chart_entry[col]), 2)

    chart_df = pd.concat([chart_df, pd.DataFrame([chart_entry])], ignore_index=True)
    chart_df = chart_df.tail(26).reset_index(drop=True)

    if len(chart_df) >= 3 and actual_hoep is not None:
        chart_df.loc[-3 % len(chart_df), "actual_hoep"] = round(actual_hoep, 2)

    chart_df.to_csv(chart_buffer_path, index=False)

    # Push updates to github
    with open(log_path, "rb") as f:
        gh_put("cloud_entry/data/predictions_log.csv", f.read(), "Update predictions_log.csv from Cloud Function")

    with open(chart_buffer_path, "rb") as f:
        gh_put("cloud_entry/data/chart_buffer.csv", f.read(), "Update chart_buffer.csv from Cloud Function")

    with open(buffer_file, "rb") as f:
        gh_put("cloud_entry/data/hoep_buffer.csv", f.read(), "Update hoep_buffer.csv from Cloud Function")

    print(" Predictions updated and pushed to GitHub.")
