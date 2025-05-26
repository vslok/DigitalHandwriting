import tensorflow as tf
import joblib
import numpy as np
import os

from .base_predictor import BasePredictor

class LSTMPredictor(BasePredictor):
    def _load_model_and_scaler(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"LSTM Model ({self.model_path}) or scaler ({self.scaler_path}) not found for {self.login}, N_{self.n_value}")

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.scaler = joblib.load(self.scaler_path)

    def predict_authentication(self, combined_features: np.ndarray) -> bool:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded for LSTMPredictor.")

        features_reshaped = combined_features.reshape(1, -1)
        scaled_features = self.scaler.transform(features_reshaped)
        prediction_array = self.model.predict(scaled_features)
        positive_class_probability = float(prediction_array[0,0])

        is_authenticated = positive_class_probability > 0.5
        return is_authenticated
