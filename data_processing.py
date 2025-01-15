#data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

def preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Handle capped values in the target variable
    capped_values = data['median_house_value'] >= 500000
    uncapped_median = data.loc[~capped_values, 'median_house_value'].median()
    data.loc[capped_values, 'median_house_value'] = uncapped_median

    # Log-transform the target variable for regression
    data['median_house_value_log'] = np.log1p(data['median_house_value'])

    # 1. Binary Classification for ocean_proximity
    data['ocean_proximity_binary'] = data['ocean_proximity'].apply(
        lambda x: 1 if x in ['NEAR BAY', '<1H OCEAN'] else 0
    )

    # 2. Encode ocean_proximity with Label Encoding
    label_encoder = LabelEncoder()
    data['ocean_proximity_encoded'] = label_encoder.fit_transform(data['ocean_proximity'])

    # Create new feature: population per household
    data['pop_per_household'] = data['population'] / data['households']

    # Identify numerical and categorical columns
    numerical_cols = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income', 'pop_per_household']
    categorical_cols = ['ocean_proximity_binary', 'ocean_proximity_encoded']

    # Scale numerical columns using RobustScaler
    scaler = RobustScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Combine numerical and encoded categorical features
    features = data[numerical_cols + categorical_cols]

    # Split features and targets
    X = features
    y_regression = data['median_house_value_log']  # Target for regression
    y_classification = data['ocean_proximity_binary']  # Target for classification

    # Split into training and test sets for both regression and classification
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    _, _, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

    feature_names = numerical_cols + categorical_cols

    return X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class, feature_names, data
