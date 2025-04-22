from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from utils.model_trainer import train_model
from routes.models import models_bp


port = 8000
app = Flask(__name__)
app.register_blueprint(models_bp)

CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "https://auto-ml-dataset-visualizer-client.vercel.app"
])


@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files or 'task' not in request.form:
        return jsonify({'error': 'Missing file or task type'}), 400

    file = request.files['file']
    task_type = request.form['task']
    algorithm = request.form.get('algorithm')  # optional

    try:
        df = pd.read_csv(file)
        result = train_model(df, task_type, algorithm)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=port, debug=True)
