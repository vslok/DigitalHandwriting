import joblib
import numpy as np
import os

from .base_predictor import BasePredictor

class KNNPredictor(BasePredictor):
    def _load_model_and_scaler(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"KNN Model ({self.model_path}) or scaler ({self.scaler_path}) not found for {self.login}, N_{self.n_value}")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict_authentication(self, combined_features: np.ndarray) -> bool:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded for KNNPredictor.")

        features_reshaped = combined_features.reshape(1, -1)
        scaled_features = self.scaler.transform(features_reshaped)
        is_authenticated: bool

        try:
            prediction = self.model.predict(scaled_features)
            is_authenticated = (int(prediction[0]) == 1)
        except Exception as e:
            raise RuntimeError(f"Error during KNN prediction: {str(e)}")

        return is_authenticated
