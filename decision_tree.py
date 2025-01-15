from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model_results import save_model_results

def train_decision_tree(X_train, y_train_class, X_test, y_test_class, feature_names):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train_class)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test_class, predictions)
    conf_matrix = confusion_matrix(y_test_class, predictions)
    class_report = classification_report(y_test_class, predictions, output_dict=True)
    feature_importances = model.feature_importances_

    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'feature_importances': dict(zip(feature_names, feature_importances.tolist())),
        'predictions': {
            'actual': y_test_class.tolist(),
            'predicted': predictions.tolist()
        },
        'model': 'decision_tree'
    }

    save_model_results(results, 'decision_tree')
    return results
