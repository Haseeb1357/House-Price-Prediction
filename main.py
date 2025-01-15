#main.py
from linear_regression import train_linear_regression
from logistic_regression import train_logistic_regression
from knn import train_knn
from naive_bayes import train_naive_bayes
from kmeans import train_kmeans
from decision_tree import train_decision_tree
from hierarchical_clustering import train_hierarchical_clustering
from data_processing import preprocess_data
from model_results import save_model_results
import matplotlib.pyplot as plt

def run_all_models():
    # Load and preprocess data
    path = 'housing.csv'
    X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class, feature_names, data = preprocess_data(path)

    # Train and evaluate all models
    models = {
        'linear_regression': (X_train, y_train_reg, X_test, y_test_reg),
        'logistic_regression': (X_train, y_train_class, X_test, y_test_class),
        'decision_tree': (X_train, y_train_class, X_test, y_test_class),
        'naive_bayes': (X_train, y_train_class, X_test, y_test_class),
        'knn': (X_train, y_train_class, X_test, y_test_class, feature_names),
        'kmeans': X_train,
        'hierarchical_clustering': X_train
    }

    for model_name, data in models.items():
        print(f"Training {model_name}...")
        if model_name == 'linear_regression':
            results = train_linear_regression(*data, feature_names)
        elif model_name == 'logistic_regression':
            results = train_logistic_regression(*data, feature_names)
        elif model_name == 'decision_tree':
            results = train_decision_tree(*data, feature_names)
        elif model_name == 'naive_bayes':
            results = train_naive_bayes(*data, feature_names)
        elif model_name == 'knn':
            results = train_knn(*data)
        elif model_name == 'kmeans':
            results = train_kmeans(data)
        elif model_name == 'hierarchical_clustering':
            results = train_hierarchical_clustering(data)

        if results:
            # Display metrics
            if model_name in ['linear_regression', 'logistic_regression', 'decision_tree', 'naive_bayes', 'knn']:
                print(f"Results for {model_name} - MSE/Accuracy: {results['mse' if model_name == 'linear_regression' else 'accuracy']}")
            elif model_name in ['kmeans', 'hierarchical_clustering']:
                print(f"Results for {model_name} - Cluster Centers/Clusters: {results['cluster_centers' if model_name == 'kmeans' else 'clusters']}")

            # Save results
            save_model_results(results, model_name)



if __name__ == "__main__":
    run_all_models()
