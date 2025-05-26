import torch
import joblib
import numpy as np
import os

from .base_predictor import BasePredictor
from ..config import DEVICE
from ...ml_models.CNN.cnn_model import KeystrokeCNN1D

class CNNPredictor(BasePredictor):
    def _load_model_and_scaler(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"CNN Model ({self.model_path}) or scaler ({self.scaler_path}) not found for {self.login}, N_{self.n_value}")

        num_features = self.get_num_features_expected()
        self.model = KeystrokeCNN1D(num_features=num_features, seq_length=1)

        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        self.scaler = joblib.load(self.scaler_path)

    def predict_authentication(self, combined_features: np.ndarray) -> bool:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded for CNNPredictor.")

        features_reshaped = combined_features.reshape(1, -1)
        scaled_features = self.scaler.transform(features_reshaped)
        cnn_input_tensor = torch.FloatTensor(scaled_features).reshape(1, 1, -1).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(cnn_input_tensor)
            _, predicted_class = torch.max(outputs, 1)
            is_authenticated = predicted_class.item() == 1

        return is_authenticated
