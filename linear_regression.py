from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from model_results import save_model_results

def train_linear_regression(X_train, y_train_reg, X_test, y_test_reg, feature_names):
    model = LinearRegression()
    model.fit(X_train, y_train_reg)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test_reg, predictions)
    r2 = r2_score(y_test_reg, predictions)
    residuals = y_test_reg - predictions

    results = {
        'mse': mse,
        'r2_score': r2,
        'predictions': {
            'actual': y_test_reg.tolist(),
            'predicted': predictions.tolist(),
            'residuals': residuals.tolist()
        },
        'feature_importance': dict(zip(feature_names, model.coef_.tolist())),
        'model': 'linear_regression'
    }

    save_model_results(results, 'linear_regression')
    return results
