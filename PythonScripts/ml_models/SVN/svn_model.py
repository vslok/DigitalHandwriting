import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

# Configure GPU memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Define base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(BASE_DIR, 'PythonScripts', 'data')
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_data_paths(n_graph: int) -> Tuple[str, str, str]:
    """Get paths for n-graph data files"""
    return (
        os.path.join(DATA_DIR, f'ML_KeystrokeData_train_{n_graph}graph.csv'),
        os.path.join(DATA_DIR, f'ML_KeystrokeData_train_val_{n_graph}graph.csv'),
        os.path.join(DATA_DIR, f'ML_KeystrokeData_test_{n_graph}graph.csv')
    )

def get_model_paths(n_graph: int) -> Tuple[str, str]:
    """Get paths for model weights and results"""
    weights_dir = os.path.join(WEIGHTS_DIR, f'N_{n_graph}')
    os.makedirs(weights_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f'svm_model_results_{n_graph}graph_{timestamp}.txt')
    return weights_dir, results_file

@dataclass
class ModelMetrics:
    Accuracy: float
    Precision: float
    Recall: float
    F1Score: float
    FAR: float  # False Acceptance Rate
    FRR: float  # False Rejection Rate

@dataclass
class ValidationResult:
    login: str
    metrics: ModelMetrics
    total_samples: int
    correct_predictions: int

class KeystrokeSVM:
    def __init__(self, n_graph: int = 1):
        self.n_graph = n_graph
        self.scalers = {}  # Dictionary to store scalers for each user
        self.models = {}   # Dictionary to store models for each user
        self.validation_results = []  # Store validation results
        self.test_results = []  # Store test results
        self.results_file = None  # File handle for results

    def _open_results_file(self):
        """Open a new results file with timestamp"""
        weights_dir, results_path = get_model_paths(self.n_graph)
        self.results_file = open(results_path, 'w', encoding='utf-8')

    def _close_results_file(self):
        """Close the results file"""
        if self.results_file:
            self.results_file.close()
            self.results_file = None

    def _write_to_results(self, text: str):
        """Write text to both console and results file"""
        print(text)
        if self.results_file:
            self.results_file.write(text + '\n')

    def _prepare_features(self, features: Dict[str, str]) -> np.ndarray:
        """Convert string arrays to numpy arrays of features"""
        if self.n_graph == 1:
            h_array = np.array([float(x) for x in features['H'].split()])
            ud_array = np.array([float(x) for x in features['UD'].split()])
            return np.concatenate([h_array, ud_array])
        else:
            h_array = np.array([float(x) for x in features['H'].split()])
            dd_array = np.array([float(x) for x in features['DD'].split()])
            uu_array = np.array([float(x) for x in features['UU'].split()])
            ud_array = np.array([float(x) for x in features['UD'].split()])
            return np.concatenate([h_array, dd_array, uu_array, ud_array])

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all relevant metrics"""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        total_samples = len(y_true)
        accuracy = (true_positives + true_negatives) / total_samples * 100

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate FAR and FRR
        far = false_positives / (false_positives + true_negatives) * 100 if (false_positives + true_negatives) > 0 else 0
        frr = false_negatives / (false_negatives + true_positives) * 100 if (false_negatives + true_positives) > 0 else 0

        return ModelMetrics(
            Accuracy=accuracy,
            Precision=precision,
            Recall=recall,
            F1Score=f1_score,
            FAR=far,
            FRR=frr
        )

    def train_and_validate(self):
        """Train and validate models for each user"""
        # Open results file
        self._open_results_file()
        self._write_to_results(f"Starting model training and validation for {self.n_graph}-graph...\n")

        # Load training, train validation, and test data
        train_path, train_val_path, test_path = get_data_paths(self.n_graph)
        train_df = pd.read_csv(train_path)
        train_val_df = pd.read_csv(train_val_path)
        test_df = pd.read_csv(test_path)

        # Get unique users
        unique_users = train_df['Login'].unique()

        for login in unique_users:
            self._write_to_results(f"\nProcessing user: {login}")

            # Get user's data
            user_train_data = train_df[train_df['Login'] == login]
            user_train_val_data = train_val_df[train_val_df['Login'] == login]
            user_test_data = test_df[test_df['Login'] == login]

            # Prepare training features
            X_train = np.array([self._prepare_features(row) for _, row in user_train_data.iterrows()])
            y_train = user_train_data['IsLegalUser'].values

            # Prepare train validation features
            X_train_val = np.array([self._prepare_features(row) for _, row in user_train_val_data.iterrows()])
            y_train_val = user_train_val_data['IsLegalUser'].values

            # Prepare test features
            X_test = np.array([self._prepare_features(row) for _, row in user_test_data.iterrows()])
            y_test = user_test_data['IsLegalUser'].values

            # Create and fit scaler for this user
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_val_scaled = scaler.transform(X_train_val)
            X_test_scaled = scaler.transform(X_test)

            # Create and train model for this user
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Train model
            self._write_to_results(f"Training model for {login}...")
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_train_val_scaled, y_train_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )

            # Validate on train validation set
            self._write_to_results(f"Validating model for {login} on train validation set...")
            train_val_predictions = (model.predict(X_train_val_scaled) > 0.5).astype(int).flatten()
            train_val_metrics = self._calculate_metrics(y_train_val, train_val_predictions)
            train_val_correct = np.sum(train_val_predictions == y_train_val)

            # Store train validation result
            train_val_result = ValidationResult(
                login=login,
                metrics=train_val_metrics,
                total_samples=len(y_train_val),
                correct_predictions=train_val_correct
            )
            self.validation_results.append(train_val_result)

            # Test on test set
            self._write_to_results(f"Testing model for {login} on test set...")
            test_predictions = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
            test_metrics = self._calculate_metrics(y_test, test_predictions)
            test_correct = np.sum(test_predictions == y_test)

            # Store test result
            test_result = ValidationResult(
                login=login,
                metrics=test_metrics,
                total_samples=len(y_test),
                correct_predictions=test_correct
            )
            self.test_results.append(test_result)

            # Print results
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

            # Store model and scaler
            self.models[login] = model
            self.scalers[login] = scaler

            # Save model and scaler
            weights_dir, _ = get_model_paths(self.n_graph)
            user_weights_dir = os.path.join(weights_dir, login)
            os.makedirs(user_weights_dir, exist_ok=True)

            model.save(os.path.join(user_weights_dir, 'model.keras'))
            joblib.dump(scaler, os.path.join(user_weights_dir, 'scaler.pkl'))

        # Print overall results
        self._write_to_results("\n=== Train Validation Summary ===")
        self.print_validation_summary(self.validation_results)
        self._write_to_results("\n=== Test Summary ===")
        self.print_validation_summary(self.test_results)

        # Close results file
        self._close_results_file()

    def print_validation_summary(self, results: List[ValidationResult]):
        """
        Print summary of validation results
        """
        if not results:
            self._write_to_results("No results to summarize")
            return

        # Calculate overall metrics
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

    def predict(self, login: str, features: Dict[str, str]) -> Tuple[float, bool]:
        """
        Predict if the keystroke pattern belongs to the specified user
        Args:
            login: User login
            features: Dictionary containing feature arrays as space-separated strings
        Returns: (probability, is_authenticated)
        """
        if login not in self.models:
            raise ValueError(f"No model found for user: {login}")

        # Prepare input features
        features_array = self._prepare_features(features)
        features_array = features_array.reshape(1, -1)

        # Scale features using user's scaler
        features_scaled = self.scalers[login].transform(features_array)

        # Make prediction
        probability = self.models[login].predict(features_scaled)[0][0]
        is_authenticated = probability > 0.5

        return probability, is_authenticated

def main():
    # Train models for all n-graph levels
    for n in [1, 2, 3]:
        print(f"\nTraining SVM model for {n}-graph...")
        svm = KeystrokeSVM(n_graph=n)
        svm.train_and_validate()

if __name__ == "__main__":
    main()
