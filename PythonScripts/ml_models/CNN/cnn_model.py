import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: Running on CPU. GPU not available!")

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(BASE_DIR, 'PythonScripts', 'data')

def get_data_paths(n_graph: int) -> Tuple[str, str, str]:
    """Get data paths for specific n-graph level"""
    return (
        os.path.join(DATA_DIR, f'ML_KeystrokeData_train_{n_graph}graph.csv'),
        os.path.join(DATA_DIR, f'ML_KeystrokeData_train_val_{n_graph}graph.csv'),
        os.path.join(DATA_DIR, f'ML_KeystrokeData_test_{n_graph}graph.csv')
    )

def get_model_paths(n_graph: int) -> Tuple[str, str]:
    """Get model paths for specific n-graph level"""
    weights_dir = os.path.join(SCRIPT_DIR, 'weights', f'N_{n_graph}')
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return weights_dir, results_dir

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@dataclass
class ModelMetrics:
    Accuracy: float
    Precision: float
    Recall: float
    F1Score: float
    FAR: float
    FRR: float

@dataclass
class ValidationResult:
    login: str
    metrics: ModelMetrics
    total_samples: int
    correct_predictions: int

class KeystrokeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class KeystrokeCNN1D(nn.Module):
    def __init__(self, num_features, seq_length, num_classes=2, dropout=0.3):
        super(KeystrokeCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class KeystrokeCNNModel:
    def __init__(self, n_graph: int = 1, seq_length=1, dropout=0.3, learning_rate=0.001):
        self.n_graph = n_graph
        self.scalers = {}
        self.models = {}
        self.validation_results = []
        self.test_results = []
        self.results_file = None
        self.seq_length = seq_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.best_params = {}

        # Set up paths for this n-graph level
        self.weights_dir, self.results_dir = get_model_paths(n_graph)
        self.train_data_path, self.train_val_data_path, self.test_data_path = get_data_paths(n_graph)

        if n_graph == 1:
            self.num_features = 40  # H and UD features
        else:
            self.num_features = 80  # H, DD, UU, and UD features

    def _open_results_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f'cnn_model_results_{self.n_graph}graph_{timestamp}.txt')
        self.results_file = open(results_path, 'w', encoding='utf-8')

    def _close_results_file(self):
        if self.results_file:
            self.results_file.close()
            self.results_file = None

    def _write_to_results(self, text: str):
        print(text)
        if self.results_file:
            self.results_file.write(text + '\n')

    def _prepare_features(self, row):
        """Prepare features based on n-graph level"""
        if self.n_graph == 1:
            h_array = np.array([float(x) for x in row['H'].split()])
            ud_array = np.array([float(x) for x in row['UD'].split()])
            return np.concatenate([h_array, ud_array])
        else:
            h_array = np.array([float(x) for x in row['H'].split()])
            dd_array = np.array([float(x) for x in row['DD'].split()])
            uu_array = np.array([float(x) for x in row['UU'].split()])
            ud_array = np.array([float(x) for x in row['UD'].split()])
            return np.concatenate([h_array, dd_array, uu_array, ud_array])

    def _calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        total_samples = len(y_true)
        accuracy = (true_positives + true_negatives) / total_samples * 100 if total_samples > 0 else 0
        precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        far = false_positives / (false_positives + true_negatives) * 100 if (false_positives + true_negatives) > 0 else 0
        frr = false_negatives / (false_negatives + true_positives) * 100 if (false_negatives + true_positives) > 0 else 0
        return ModelMetrics(accuracy, precision, recall, f1, far, frr)

    def train_model(self, model, train_loader, val_loader, num_epochs=50, patience=10):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                self._write_to_results(f"Early stopping at epoch {epoch + 1}")
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model

    def train_and_validate(self):
        self._open_results_file()
        self._write_to_results(f"Starting CNN model training and validation for {self.n_graph}-graph...\n")
        train_df = pd.read_csv(self.train_data_path)
        train_val_df = pd.read_csv(self.train_val_data_path)
        test_df = pd.read_csv(self.test_data_path)
        unique_users = train_df['Login'].unique()
        for login in unique_users:
            self._write_to_results(f"\nProcessing user: {login}")
            user_train_data = train_df[train_df['Login'] == login]
            user_train_val_data = train_val_df[train_val_df['Login'] == login]
            user_test_data = test_df[test_df['Login'] == login]
            self._write_to_results(f"Training samples: {len(user_train_data)}")
            self._write_to_results(f"Validation samples: {len(user_train_val_data)}")
            self._write_to_results(f"Test samples: {len(user_test_data)}")
            X_train = np.array([self._prepare_features(row) for _, row in user_train_data.iterrows()])
            y_train = user_train_data['IsLegalUser'].values
            X_train_val = np.array([self._prepare_features(row) for _, row in user_train_val_data.iterrows()])
            y_train_val = user_train_val_data['IsLegalUser'].values
            X_test = np.array([self._prepare_features(row) for _, row in user_test_data.iterrows()])
            y_test = user_test_data['IsLegalUser'].values
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_val_scaled = scaler.transform(X_train_val)
            X_test_scaled = scaler.transform(X_test)
            X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
            X_train_val_reshaped = X_train_val_scaled.reshape(-1, 1, X_train_val_scaled.shape[1])
            X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
            train_dataset = KeystrokeDataset(X_train_reshaped, y_train)
            train_val_dataset = KeystrokeDataset(X_train_val_reshaped, y_train_val)
            test_dataset = KeystrokeDataset(X_test_reshaped, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            train_val_loader = DataLoader(train_val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            model = KeystrokeCNN1D(num_features=self.num_features, seq_length=1, dropout=self.dropout).to(device)
            model = self.train_model(model, train_loader, train_val_loader)
            self._write_to_results(f"Validating model for {login} on train validation set...")
            train_val_predictions = []
            train_val_true = []
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in train_val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(probabilities, 1)
                    train_val_predictions.extend(predicted.cpu().numpy())
                    train_val_true.extend(batch_y.cpu().numpy())
            self._write_to_results(f"Prediction distribution: {np.bincount(train_val_predictions)}")
            self._write_to_results(f"True labels distribution: {np.bincount(train_val_true)}")
            train_val_metrics = self._calculate_metrics(train_val_true, train_val_predictions)
            train_val_correct = np.sum(np.array(train_val_predictions) == np.array(train_val_true))
            train_val_result = ValidationResult(
                login=login,
                metrics=train_val_metrics,
                total_samples=len(train_val_true),
                correct_predictions=train_val_correct
            )
            self.validation_results.append(train_val_result)
            self._write_to_results(f"Testing model for {login} on test set...")
            test_predictions = []
            test_true = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(probabilities, 1)
                    test_predictions.extend(predicted.cpu().numpy())
                    test_true.extend(batch_y.cpu().numpy())
            self._write_to_results(f"Test prediction distribution: {np.bincount(test_predictions)}")
            self._write_to_results(f"Test true labels distribution: {np.bincount(test_true)}")
            test_metrics = self._calculate_metrics(test_true, test_predictions)
            test_correct = np.sum(np.array(test_predictions) == np.array(test_true))
            test_result = ValidationResult(
                login=login,
                metrics=test_metrics,
                total_samples=len(test_true),
                correct_predictions=test_correct
            )
            self.test_results.append(test_result)
            self._write_to_results(f"\nTrain Validation Results for {login}:")
            self._write_to_results(f"Accuracy: {train_val_metrics.Accuracy:.2f}%")
            self._write_to_results(f"Precision: {train_val_metrics.Precision:.2f}%")
            self._write_to_results(f"Recall: {train_val_metrics.Recall:.2f}%")
            self._write_to_results(f"F1 Score: {train_val_metrics.F1Score:.2f}%")
            self._write_to_results(f"FAR: {train_val_metrics.FAR:.2f}%")
            self._write_to_results(f"FRR: {train_val_metrics.FRR:.2f}%")
            self._write_to_results(f"\nTest Results for {login}:")
            self._write_to_results(f"Accuracy: {test_metrics.Accuracy:.2f}%")
            self._write_to_results(f"Precision: {test_metrics.Precision:.2f}%")
            self._write_to_results(f"Recall: {test_metrics.Recall:.2f}%")
            self._write_to_results(f"F1 Score: {test_metrics.F1Score:.2f}%")
            self._write_to_results(f"FAR: {test_metrics.FAR:.2f}%")
            self._write_to_results(f"FRR: {test_metrics.FRR:.2f}%")
            self.models[login] = model
            self.scalers[login] = scaler
            user_weights_dir = os.path.join(self.weights_dir, login)
            os.makedirs(user_weights_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(user_weights_dir, 'model.pth'))
            joblib.dump(scaler, os.path.join(user_weights_dir, 'scaler.pkl'))
        self._write_to_results("\n=== Train Validation Summary ===")
        self.print_validation_summary(self.validation_results)
        self._write_to_results("\n=== Test Summary ===")
        self.print_validation_summary(self.test_results)
        self._close_results_file()

    def print_validation_summary(self, results: List[ValidationResult]):
        if not results:
            self._write_to_results("No results to summarize")
            return
        total_samples = sum(r.total_samples for r in results)
        total_correct = sum(r.correct_predictions for r in results)
        overall_accuracy = (total_correct / total_samples) * 100
        avg_precision = sum(r.metrics.Precision for r in results) / len(results)
        avg_recall = sum(r.metrics.Recall for r in results) / len(results)
        avg_f1 = sum(r.metrics.F1Score for r in results) / len(results)
        avg_far = sum(r.metrics.FAR for r in results) / len(results)
        avg_frr = sum(r.metrics.FRR for r in results) / len(results)
        self._write_to_results(f"\nOverall Statistics:")
        self._write_to_results(f"Total Users: {len(results)}")
        self._write_to_results(f"Total Samples: {total_samples}")
        self._write_to_results(f"Overall Accuracy: {overall_accuracy:.2f}%")
        self._write_to_results(f"Average Precision: {avg_precision:.2f}%")
        self._write_to_results(f"Average Recall: {avg_recall:.2f}%")
        self._write_to_results(f"Average F1 Score: {avg_f1:.2f}%")
        self._write_to_results(f"Average FAR: {avg_far:.2f}%")
        self._write_to_results(f"Average FRR: {avg_frr:.2f}%")
        self._write_to_results("\nPer-User Statistics:")
        self._write_to_results("Login\t\tAccuracy\tPrecision\tRecall\t\tF1 Score\tFAR\t\tFRR")
        self._write_to_results("-" * 100)
        for r in sorted(results, key=lambda x: x.login):
            self._write_to_results(f"{r.login}\t\t{r.metrics.Accuracy:.2f}%\t\t{r.metrics.Precision:.2f}%\t\t{r.metrics.Recall:.2f}%\t\t{r.metrics.F1Score:.2f}%\t\t{r.metrics.FAR:.2f}%\t\t{r.metrics.FRR:.2f}%")

    def predict(self, login: str, features_dict: Dict[str, str]) -> Tuple[float, bool]:
        if login not in self.models:
            raise ValueError(f"No model found for user: {login}")

        if self.n_graph == 1:
            features = self._prepare_features({'H': features_dict['H'], 'UD': features_dict['UD']})
        else:
            features = self._prepare_features(features_dict)

        features = features.reshape(1, 1, -1)
        features = torch.FloatTensor(features).to(device)
        features_scaled = self.scalers[login].transform(features.reshape(1, -1))
        features_scaled = features_scaled.reshape(1, 1, -1)
        features_scaled = torch.FloatTensor(features_scaled).to(device)
        self.models[login].eval()
        with torch.no_grad():
            outputs = self.models[login](features_scaled)
            probabilities = torch.softmax(outputs, dim=1)
            probability = probabilities[0][1].item()
            is_authenticated = probability > 0.5
        return probability, is_authenticated

def main():
    for n_graph in [1, 2, 3]:
        print(f"\nTraining and validating CNN models for {n_graph}-graph...")
        cnn = KeystrokeCNNModel(n_graph=n_graph)
        cnn.train_and_validate()

if __name__ == "__main__":
    main()
