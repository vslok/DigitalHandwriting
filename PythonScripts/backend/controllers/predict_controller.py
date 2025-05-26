from flask import Blueprint, request, jsonify
from ..services.prediction_service import prediction_service

predict_blueprint = Blueprint('predict_api', __name__, url_prefix='/api/v1')

@predict_blueprint.route('/predict', methods=['POST'])
def handle_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        model_type = data.get('model_type')
        n_value_str = data.get('n_value')
        login = data.get('login')
        h_values = data.get('H_values')
        ud_values = data.get('UD_values')

        if not all([model_type, n_value_str is not None, login, h_values is not None, ud_values is not None]):
            return jsonify({"error": "Missing required parameters. Required: model_type, n_value, login, H_values, UD_values"}), 400

        try:
            n_value = int(n_value_str)
            if n_value not in [1, 2, 3]:
                return jsonify({"error": "Invalid n_value. Must be an integer (1, 2, or 3)."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "n_value must be a valid integer."}), 400

        if not isinstance(h_values, list) or not isinstance(ud_values, list):
            return jsonify({"error": "H_values and UD_values must be arrays (lists) of numbers."}), 400
        try:
            h_values_numeric = [float(v) for v in h_values]
            ud_values_numeric = [float(v) for v in ud_values]
        except (ValueError, TypeError):
            return jsonify({"error": "All elements in H_values and UD_values must be numbers."}), 400

        result = prediction_service.get_prediction(model_type, n_value, login, h_values_numeric, ud_values_numeric)
        return jsonify(result), 200

    except FileNotFoundError as e:
        print(f"Controller FileNotFoundError: {e}")
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        print(f"Controller ValueError: {e}")
        return jsonify({"error": str(e)}), 400
    except NotImplementedError as e:
        print(f"Controller NotImplementedError: {e}")
        return jsonify({"error": str(e)}), 501
    except RuntimeError as e:
        print(f"Controller RuntimeError: {e}")
        return jsonify({"error": f"An internal processing error occurred: {str(e)}"}), 500
    except Exception as e:
        print(f"Controller Unexpected Error: {e}")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500
