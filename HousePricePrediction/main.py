"""
main.py
-------
Train and save the house price prediction model using the California Housing dataset.
"""

import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import joblib


def load_data():
    """Load California housing dataset and return features & target."""
    dataset = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["price"] = dataset.target
    return df


def train_model(df: pd.DataFrame):
    """Train XGBoost model on the dataset."""
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)

    print(f"Model R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    return model


def save_model(model, path="model.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)
    save_model(model)
