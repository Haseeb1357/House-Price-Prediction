#model_results.py
import os

import json


def save_model_results(results, model_name):
    # Directory to save results
    result_dir = 'results'

    # Ensure the result directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # File path for saving results
    file_path = os.path.join(result_dir, f"{model_name}_results.json")

    # Save results as JSON file
    with open(file_path, 'w') as file:
        json.dump(results, file)

    print(f"Results for {model_name} saved successfully to {file_path}")
