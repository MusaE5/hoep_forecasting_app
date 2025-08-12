import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json

# Global Quantiles
quantiles = [0.1, 0.5, 0.9]

# Loss Functions 
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

# Load quantile models for live prediction
def load_quantile_models(model_dir="models"):
    """Load all three quantile models with custom loss functions"""
    custom_objects_10 = {'loss': quantile_loss(0.1)}
    custom_objects_50 = {'loss': quantile_loss(0.5)}  
    custom_objects_90 = {'loss': quantile_loss(0.9)}
    
    models = {
        'q10': load_model(f"{model_dir}/hoep_quantile_q_10.keras", custom_objects=custom_objects_10),
        'q50': load_model(f"{model_dir}/hoep_quantile_q_50.keras", custom_objects=custom_objects_50), 
        'q90': load_model(f"{model_dir}/hoep_quantile_q_90.keras", custom_objects=custom_objects_90)
    }
    
    return models
