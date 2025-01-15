from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_results import save_model_results

def train_knn(X_train, y_train, X_test, y_test, feature_names):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, output_dict=True)

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': {
            'actual': y_test.tolist(),
            'predicted': predictions.tolist()
        },
        'model': 'knn'
    }

    save_model_results(results, 'knn')
    return results
