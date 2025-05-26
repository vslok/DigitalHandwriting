from abc import ABC, abstractmethod

class BasePredictor(ABC):
    def __init__(self, model_path: str, scaler_path: str, n_value: int, login: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.n_value = n_value
        self.login = login
        self.model = None
        self.scaler = None
        self._load_model_and_scaler()

    @abstractmethod
    def _load_model_and_scaler(self):
        """Loads the specific model and scaler from files."""
        pass

    @abstractmethod
    def predict_authentication(self, features: list) -> bool:
        """
        Receives already processed (n-grammed, padded, combined) features,
        applies scaling, and returns the authentication decision: bool.
        """
        pass

    def get_num_features_expected(self) -> int:
        """Returns the number of features this model expects (40 or 80)."""
        return 40 if self.n_value == 1 else 80
