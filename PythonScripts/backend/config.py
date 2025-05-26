import os
import torch
from enum import Enum

class ModelType(Enum):
    CNN = "CNN"
    SVM = "SVM"
    GRU = "GRU"
    KNN = "KNN"
    LSTM = "LSTM"
    MLP = "MLP"
    NAIVEBAYES = "NAIVEBAYES"
    RANDOMFOREST = "RANDOMFOREST"
    XGBOOST = "XGBOOST"
    # Add other model types here

TARGET_LENGTH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..', '..'))
BASE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'PythonScripts', 'ml_models')

MODEL_WEIGHTS_DIRS = {
    ModelType.CNN: os.path.join(BASE_MODEL_DIR, 'CNN', 'weights'),
    ModelType.SVM: os.path.join(BASE_MODEL_DIR, 'SVM', 'weights'),
    ModelType.GRU: os.path.join(BASE_MODEL_DIR, 'GRU', 'weights'),
    ModelType.KNN: os.path.join(BASE_MODEL_DIR, 'KNN', 'weights'),
    ModelType.LSTM: os.path.join(BASE_MODEL_DIR, 'LSTM', 'weights'),
    ModelType.MLP: os.path.join(BASE_MODEL_DIR, 'MLP', 'weights'),
    ModelType.NAIVEBAYES: os.path.join(BASE_MODEL_DIR, 'NaiveBayes', 'weights'),
    ModelType.RANDOMFOREST: os.path.join(BASE_MODEL_DIR, 'RandomForest', 'weights'),
    ModelType.XGBOOST: os.path.join(BASE_MODEL_DIR, 'XGBoost', 'weights'),
    # Add other model types and their paths here
}
