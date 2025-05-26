import numpy as np
from typing import List, Dict, Union

from ..config import TARGET_LENGTH
from ..ml_models.model_factory import model_factory

def _pad_sequence(values: List[Union[float, int]], target_length: int) -> List[float]:
    if not isinstance(values, list):
        values = list(values)
    float_values = [float(v) for v in values]
    if len(float_values) < target_length:
        return float_values + [0.0] * (target_length - len(float_values))
    elif len(float_values) > target_length:
        return float_values[:target_length]
    return float_values

def _calculate_ngraph_features(hold_times: List[float], between_times: List[float], n: int) -> Union[Dict[str, List[float]], None]:
    """Calculate n-graph features for a single sequence."""
    if not isinstance(hold_times, list): hold_times = [float(ht) for ht in list(hold_times)]
    if not isinstance(between_times, list): between_times = [float(bt) for bt in list(between_times)]

    if n == 1:
        return {
            'H': hold_times,
            'UD': between_times
        }

    if len(hold_times) < n:
        return None

    n_graph_dd, n_graph_uu, n_graph_h, n_graph_ud = [], [], [], []
    num_possible_n_graphs = len(hold_times) - n + 1

    for i in range(num_possible_n_graphs):
        n_graph_h.append(sum(hold_times[i : i + n]))

        if i + (n - 1) < len(hold_times) and i + (n - 1) <= len(between_times):
            current_dd = sum(hold_times[i+j] + between_times[i+j] for j in range(n - 1)) + hold_times[i+n-1]
            n_graph_dd.append(current_dd)
        else:
            n_graph_dd.append(0.0)

        if i + (n - 1) <= len(between_times) and i + n <= len(hold_times):
            current_uu = sum(between_times[i+j] + hold_times[i+j+1] for j in range(n - 1))
            n_graph_uu.append(current_uu)
        else:
            n_graph_uu.append(0.0)

        if n > 1:
            if i + (n - 1) <= len(between_times):
                n_graph_ud.append(sum(between_times[i : i + n - 1]))
            else:
                n_graph_ud.append(0.0)

    return {
        'H': n_graph_h,
        'DD': n_graph_dd,
        'UU': n_graph_uu,
        'UD': n_graph_ud if n > 1 else []
    }

class PredictionService:
    def get_prediction(self, model_type: str, n_value: int, login: str, h_values: List[float], ud_values: List[float]) -> Dict[str, Union[str, int, bool]]:
        if not isinstance(h_values, list) or not isinstance(ud_values, list):
            raise ValueError("H_values and UD_values must be lists of numbers.")

        combined_features_np: np.ndarray
        num_expected_features: int

        if n_value > 1:
            ngraph_features_dict = _calculate_ngraph_features(h_values, ud_values, n_value)
            if ngraph_features_dict is None:
                raise ValueError(f"Not enough data for n-graph calculation with n={n_value}. Min hold_times: {n_value}. Provided: {len(h_values)}")

            h_processed = _pad_sequence(ngraph_features_dict.get('H', []), TARGET_LENGTH)
            dd_processed = _pad_sequence(ngraph_features_dict.get('DD', []), TARGET_LENGTH)
            uu_processed = _pad_sequence(ngraph_features_dict.get('UU', []), TARGET_LENGTH)
            ud_processed = _pad_sequence(ngraph_features_dict.get('UD', []), TARGET_LENGTH)

            combined_features_list = h_processed + dd_processed + uu_processed + ud_processed
            num_expected_features = 4 * TARGET_LENGTH
        else:
            h_processed = _pad_sequence(h_values, TARGET_LENGTH)
            ud_processed = _pad_sequence(ud_values, TARGET_LENGTH)
            combined_features_list = h_processed + ud_processed
            num_expected_features = 2 * TARGET_LENGTH

        combined_features_np = np.array(combined_features_list, dtype=np.float32)

        if len(combined_features_np) != num_expected_features:
            actual_predictor_expected_len = 40 if n_value == 1 else 80
            if len(combined_features_np) != actual_predictor_expected_len:
                raise ValueError(f"Feature construction error. Expected {actual_predictor_expected_len} features, got {len(combined_features_np)} for n={n_value}")
        try:
            predictor = model_factory.get_predictor(model_type, n_value, login)
        except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as e:
            raise e

        try:
            is_authenticated = predictor.predict_authentication(combined_features_np)
        except Exception as e:
            raise RuntimeError(f"Error during model prediction execution: {str(e)}")

        return {
            "login": login,
            "model_type": model_type,
            "n_value": n_value,
            "authenticated": bool(is_authenticated)
        }

prediction_service = PredictionService()
