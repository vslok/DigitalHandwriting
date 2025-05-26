from flask import Flask
from .controllers.predict_controller import predict_blueprint
# Optional: Import and configure logging if needed from config.py or here

app = Flask(__name__)

# Register blueprints (controllers)
app.register_blueprint(predict_blueprint) # This will make routes available under /api/predict

@app.route('/') # A simple health check or welcome route for the root
def index():
    return "Welcome to the Keystroke Dynamics Prediction API! Use /api/predict for predictions.", 200

if __name__ == '__main__':
    # To run this:
    # From the DigitalHandwritting/ root directory:
    # 1. Set FLASK_APP:
    #    (PowerShell) $env:FLASK_APP = "PythonScripts.backend.main"
    #    (bash) export FLASK_APP="PythonScripts.backend.main"
    # 2. Set FLASK_ENV (optional, for development):
    #    (PowerShell) $env:FLASK_ENV = "development"
    #    (bash) export FLASK_ENV="development"
    # 3. Run Flask:
    #    flask run --host=0.0.0.0 --port=5000
    #
    # Alternatively, for direct python execution (if running main.py directly):
    # (Ensure PythonScripts/backend/ is the current working directory or adjust paths)
    app.run(host='0.0.0.0', port=5000, debug=True)
