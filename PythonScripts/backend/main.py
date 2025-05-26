from flask import Flask
from PythonScripts.backend.controllers.predict_controller import predict_blueprint

app = Flask(__name__)

app.register_blueprint(predict_blueprint)

@app.route('/')
def index():
    return "Welcome to the Keystroke Dynamics Prediction API! Use /api/predict for predictions.", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
