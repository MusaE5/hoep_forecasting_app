import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import joblib
import json

# --- Global Quantiles ---
quantiles = [0.1, 0.5, 0.9]

# --- Loss Functions ---
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

def combined_quantile_loss(y_true, y_pred):
    total_loss = 0
    for i, q in enumerate(quantiles):
        q_pred = y_pred[:, i:i+1]
        error = y_true - q_pred
        total_loss += tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return total_loss

# --- Model Builders ---
def create_multi_quantile_model(input_shape, quantiles=quantiles):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(32),
        LeakyReLU(alpha=0.01),
        Dense(len(quantiles), activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss=combined_quantile_loss)
    return model

def create_single_quantile_model(input_shape, quantile):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dense(32),
        LeakyReLU(alpha=0.01),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss=quantile_loss(quantile))
    return model

# --- Training ---
def train_quantile_models(X_train, y_train, X_test, y_test, method='separate'):
    if method == 'combined':
        model = create_multi_quantile_model(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)
        predictions = model.predict(X_test, verbose=0)
        quantile_predictions = {f'q_{int(q * 100)}': predictions[:, i] for i, q in enumerate(quantiles)}
        quantile_models = {'combined': model}
    else:
        quantile_models = {}
        quantile_predictions = {}
        for q in quantiles:
            print(f"Training quantile {q} model...")
            model = create_single_quantile_model(X_train.shape[1], q)
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
            quantile_models[f'q_{int(q * 100)}'] = model
            quantile_predictions[f'q_{int(q * 100)}'] = model.predict(X_test, verbose=0).flatten()
    return quantile_predictions, quantile_models

# --- Save/Load ---
def save_quantile_models(quantile_models, scaler, features_list, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    for q_name, model in quantile_models.items():
        model_path = os.path.join(model_dir, f"hoep_quantile_{q_name}.keras")
        model.save(model_path)
    joblib.dump(scaler, os.path.join(model_dir, "quantile_feature_scaler.pkl"))
    config = {
        'quantiles': quantiles,
        'model_files': {q_name: f"hoep_quantile_{q_name}.keras" for q_name in quantile_models},
        'scaler_file': "quantile_feature_scaler.pkl",
        'features': features_list,
        'method': 'combined' if 'combined' in quantile_models else 'separate'
    }
    with open(os.path.join(model_dir, "quantile_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

# --- Load ---
def load_quantile_models(model_dir="models"):
    with open(os.path.join(model_dir, "quantile_config.json"), 'r') as f:
        config = json.load(f)
    scaler = joblib.load(os.path.join(model_dir, config['scaler_file']))
    quantile_models = {}
    for q_name, model_file in config['model_files'].items():
        model_path = os.path.join(model_dir, model_file)
        if config['method'] == 'combined':
            model = load_model(model_path, custom_objects={'combined_quantile_loss': combined_quantile_loss})
        else:
            q_value = float(q_name.split('_')[1]) / 100
            model = load_model(model_path, custom_objects={'loss': quantile_loss(q_value)})
        quantile_models[q_name] = model
    return quantile_models, scaler, config

# --- Prediction ---
def predict_quantiles(quantile_models, scaler, X_new, config):
    X_scaled = scaler.transform(X_new)
    predictions = {}
    if config['method'] == 'combined':
        model = quantile_models['combined']
        pred_array = model.predict(X_scaled, verbose=0)
        for i, q in enumerate(config['quantiles']):
            predictions[f'q_{int(q * 100)}'] = pred_array[:, i]
    else:
        for q_name, model in quantile_models.items():
            predictions[q_name] = model.predict(X_scaled, verbose=0).flatten()
    return predictions

def predict_single_sample(quantile_models, scaler, config, features_dict):
    feature_array = np.array([features_dict[feat] for feat in config['features']]).reshape(1, -1)
    predictions = predict_quantiles(quantile_models, scaler, feature_array, config)
    return {q_name: float(pred[0]) for q_name, pred in predictions.items()}

def get_prediction_summary(predictions):
    median = predictions.get('q_50')
    low = predictions.get('q_10')
    high = predictions.get('q_90')
    return {
        'predicted_price': median,
        'confidence_interval_80': {
            'lower': low,
            'upper': high,
            'width': high - low if (high is not None and low is not None) else None
        }
    }

# --- Evaluation ---
def evaluate_quantile_predictions(y_true, quantile_predictions):
    results = {}
    for q_name, q_pred in quantile_predictions.items():
        q_value = float(q_name.split('_')[1]) / 100
        error = y_true - q_pred
        pinball = np.mean(np.maximum(q_value * error, (q_value - 1) * error))
        coverage = np.mean(y_true <= q_pred)
        rmse = np.sqrt(np.mean((y_true - q_pred) ** 2))
        results[q_name] = {
            'pinball_loss': pinball,
            'coverage': coverage,
            'expected_coverage': q_value,
            'rmse': rmse
        }
    return results

def calculate_prediction_intervals(quantile_predictions, confidence_levels=[0.8]):
    """Return lower, upper bounds and widths for confidence intervals"""
    intervals = {}
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        lower_q = f"q_{int((alpha/2)*100)}"
        upper_q = f"q_{int((1-alpha/2)*100)}"
        if lower_q in quantile_predictions and upper_q in quantile_predictions:
            intervals[f'{int(conf_level * 100)}%'] = {
                'lower': quantile_predictions[lower_q],
                'upper': quantile_predictions[upper_q],
                'width': quantile_predictions[upper_q] - quantile_predictions[lower_q]
            }
    return intervals


# --- Visualization ---
def plot_quantile_predictions(y_true, quantile_predictions, start_idx=0, end_idx=200):
    x = range(start_idx, min(end_idx, len(y_true)))
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true[start_idx:end_idx], 'k-', label='Actual HOEP', linewidth=2)
    if 'q_50' in quantile_predictions:
        plt.plot(x, quantile_predictions['q_50'][start_idx:end_idx], 'r--', label='Median Prediction', linewidth=2)
    if 'q_10' in quantile_predictions and 'q_90' in quantile_predictions:
        plt.fill_between(x, quantile_predictions['q_10'][start_idx:end_idx], quantile_predictions['q_90'][start_idx:end_idx], alpha=0.3, color='blue', label='80% Prediction Interval')
    plt.xlabel('Time (hours)')
    plt.ylabel('HOEP (CAD/MWh)')
    plt.title('Quantile Regression with Uncertainty Bands')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
