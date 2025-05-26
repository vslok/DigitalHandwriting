import os
from typing import Dict, Type

from .base_predictor import BasePredictor
from .cnn_predictor import CNNPredictor
from .svm_predictor import SVMPredictor
from .knn_predictor import KNNPredictor
from .randomforest_predictor import RandomForestPredictor
from .naivebayes_predictor import NaiveBayesPredictor
from .xgboost_predictor import XGBoostPredictor
from .mlp_predictor import MLPPredictor
from .gru_predictor import GRUPredictor
from .lstm_predictor import LSTMPredictor
from ..config import MODEL_WEIGHTS_DIRS

class ModelFactory:
    def __init__(self):
        self._predictors_map: Dict[str, Type[BasePredictor]] = {
            "CNN": CNNPredictor,
            "SVM": SVMPredictor,
            "KNN": KNNPredictor,
            "RANDOMFOREST": RandomForestPredictor,
            "NAIVEBAYES": NaiveBayesPredictor,
            "XGBOOST": XGBoostPredictor,
            "MLP": MLPPredictor,
            "GRU": GRUPredictor,
            "LSTM": LSTMPredictor,
            # ... and so on for other model types
        }

    def get_predictor(self, model_type: str, n_value: int, login: str) -> BasePredictor:
        model_type_upper = model_type.upper()
        cache_key = (model_type_upper, n_value, login)

        predictor_class = self._predictors_map.get(model_type_upper)
        if not predictor_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Construct paths
        base_dir_for_type = MODEL_WEIGHTS_DIRS.get(model_type_upper)
        if not base_dir_for_type:
            raise ValueError(f"Configuration missing for model type weights directory: {model_type_upper}")

        user_specific_path = os.path.join(base_dir_for_type, f"N_{n_value}", login)

        # Determine model file extension based on type (this might need more sophistication)
        if model_type_upper == "CNN":
            model_filename = "model.pth"
        elif model_type_upper == "SVM": # Keras
            model_filename = "model.keras"
        elif model_type_upper == "KNN":
            model_filename = "model.pkl" # Assuming .pkl for KNN
        elif model_type_upper == "RANDOMFOREST":
            model_filename = "model.pkl" # Assuming .pkl for RandomForest
        elif model_type_upper == "NAIVEBAYES":
            model_filename = "model.pkl" # Assuming .pkl for NaiveBayes
        elif model_type_upper == "XGBOOST":
            model_filename = "model.pkl" # Assuming .pkl for XGBoost (or .bst if native)
        elif model_type_upper == "MLP":
            model_filename = "model.keras" # Assuming .keras for MLP (Keras)
        elif model_type_upper == "GRU":
            model_filename = "model.keras" # Assuming .keras for GRU (Keras)
        elif model_type_upper == "LSTM":
            model_filename = "model.keras" # Assuming .keras for LSTM (Keras)
        else:
            raise NotImplementedError(f"Model filename convention not defined for type: {model_type_upper}")

        scaler_filename = "scaler.pkl"

        model_full_path = os.path.join(user_specific_path, model_filename)
        scaler_full_path = os.path.join(user_specific_path, scaler_filename)

        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found: {model_full_path}")
        if not os.path.exists(scaler_full_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_full_path}")

        try:
            predictor_instance = predictor_class(model_full_path, scaler_full_path, n_value, login)
            return predictor_instance
        except Exception as e:
            raise RuntimeError(f"Failed to create predictor for {model_type}: {str(e)}")

model_factory = ModelFactory()
