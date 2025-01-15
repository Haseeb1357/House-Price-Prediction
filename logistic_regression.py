from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_results import save_model_results

def train_logistic_regression(X_train, y_train_class, X_test, y_test_class, feature_names):
    model = LogisticRegression()
    model.fit(X_train, y_train_class)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test_class, predictions)
    conf_matrix = confusion_matrix(y_test_class, predictions)
    class_report = classification_report(y_test_class, predictions, output_dict=True)

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': {
            'actual': y_test_class.tolist(),
            'predicted': predictions.tolist()
        },
        'model': 'logistic_regression'
    }

    save_model_results(results, 'logistic_regression')
    return results
