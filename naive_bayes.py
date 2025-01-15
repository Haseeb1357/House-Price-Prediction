from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_results import save_model_results

def train_naive_bayes(X_train, y_train_class, X_test, y_test_class, feature_names):
    model = GaussianNB()
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
        'model': 'naive_bayes'
    }

    save_model_results(results, 'naive_bayes')
    return results
