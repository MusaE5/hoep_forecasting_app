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

#  Global Quantiles 
quantiles = [0.1, 0.5, 0.9]

#  Loss Function 
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss


#  Model Builders
def create_single_quantile_model(input_shape, q):
    """Create a model for a single quantile"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.2),
        Dense(64),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.2),
        Dense(32),
        LeakyReLU(negative_slope=0.01),
        Dense(1, activation='linear')  # Single output for one quantile
    ])
    model.compile(optimizer=Adam(0.001), loss=quantile_loss(q))
    return model


#  Training 
def train_quantile_models(X_train, y_train, X_test, y_test):
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

# Save/Load 
def save_quantile_models(quantile_models, scaler, features_list, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    for q_name, model in quantile_models.items():
        model_path = os.path.join(model_dir, f"hoep_quantile_{q_name}.keras")
        model.save(model_path)
    joblib.dump(scaler, os.path.join(model_dir, "quantile_feature_scaler.pkl"))
  
