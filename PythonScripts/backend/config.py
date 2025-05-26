import os
import torch

TARGET_LENGTH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..', '..'))
BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'PythonScripts', 'ml_models')

MODEL_WEIGHTS_DIRS = {
    "CNN": os.path.join(BASE_MODEL_DIR, 'CNN', 'weights'),
    "SVM": os.path.join(BASE_MODEL_DIR, 'SVM', 'weights'),
    "GRU": os.path.join(BASE_MODEL_DIR, 'GRU', 'weights'),
    "KNN": os.path.join(BASE_MODEL_DIR, 'KNN', 'weights'),
    "LSTM": os.path.join(BASE_MODEL_DIR, 'LSTM', 'weights'),
    "MLP": os.path.join(BASE_MODEL_DIR, 'MLP', 'weights'),
    "NAIVEBAYES": os.path.join(BASE_MODEL_DIR, 'NaiveBayes', 'weights'),
    "RANDOMFOREST": os.path.join(BASE_MODEL_DIR, 'RandomForest', 'weights'),
    "XGBOOST": os.path.join(BASE_MODEL_DIR, 'XGBoost', 'weights'),
    # Add other model types and their paths here
}
