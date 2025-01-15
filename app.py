from flask import Flask, request, jsonify, render_template
import os
import json
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model results from the results directory
def load_model_results(model_name):
    file_path = os.path.join("results", f"{model_name}_results.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return None

# Load dataset summary
def load_dataset():
    file_path = os.path.join("data", "housing.csv")  # Update with your dataset path
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return {
            "columns": list(data.columns),
            "preview": data.head().to_dict(orient="records"),
            "description": data.describe().to_dict()
        }
    return None

@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/get_model_data', methods=['GET'])
def get_model_data():
    model_name = request.args.get("model")
    data = load_model_results(model_name)
    if data:
        return jsonify({
            "metrics": {
                "mse": data.get("mse", "N/A"),
                "accuracy": data.get("accuracy", "N/A"),
                "r2_score": data.get("r2_score", "N/A"),
            },
            "predictions": {
                "actual": data.get("predictions", {}).get("actual", []),
                "predicted": data.get("predictions", {}).get("predicted", []),
            },
            "model": model_name
        })
    return jsonify({"error": "Model data not found"}), 404

@app.route('/compare_models', methods=['GET'])
def compare_models():
    model1 = request.args.get("model1")
    model2 = request.args.get("model2")

    data1 = load_model_results(model1)
    data2 = load_model_results(model2)

    if data1 and data2:
        # Ensure metrics exist, defaulting to "N/A" if they don't
        model1_metrics = {
            "metrics" : {
                "mse": data1.get("mse", "N/A"),
                "accuracy": data1.get("accuracy", "N/A"),
                "r2_score": data1.get("r2_score", "N/A"),
            },
            "predictions": {
                "actual": data1.get("predictions", {}).get("actual", []),
                "predicted": data1.get("predictions", {}).get("predicted", []),
            },

        }
        model2_metrics = {
            "metrics" : {
                "mse": data2.get("mse", "N/A"),
                "accuracy": data2.get("accuracy", "N/A"),
                "r2_score": data2.get("r2_score", "N/A"),
            },
            "predictions": {
                "actual": data2.get("predictions", {}).get("actual", []),
                "predicted": data2.get("predictions", {}).get("predicted", []),
            },

        }

        return jsonify({
            "model1": {"name": model1, "metrics": model1_metrics},
            "model2": {"name": model2, "metrics": model2_metrics},
        })

    return jsonify({"error": "One or both models not found"}), 404


@app.route('/best_model', methods=['GET'])
def best_model():
    results_dir = "results"
    best_classification = None
    best_regression = None

    for file in os.listdir(results_dir):
        if file.endswith("_results.json"):
            model_name = file.replace("_results.json", "")
            data = load_model_results(model_name)

            if data:
                # Determine the best classification model
                accuracy = data.get("accuracy")
                if accuracy is not None:
                    if not best_classification or accuracy > best_classification["accuracy"]:
                        best_classification = {"name": model_name, "accuracy": accuracy}

                # Determine the best regression model
                mse = data.get("mse")
                if mse is not None:
                    if not best_regression or mse < best_regression["mse"]:
                        best_regression = {"name": model_name, "mse": mse}

    return jsonify({
        "best_classification_model": best_classification,
        "best_regression_model": best_regression
    })



@app.route('/get_data_summary', methods=['GET'])
def get_data_summary():
    data = load_dataset()
    if data:
        return jsonify(data)
    return jsonify({"error": "Dataset not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)