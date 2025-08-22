"""
main.py
-------
Train a model to classify Rock vs Mine using the Sonar dataset
and save it for use in Streamlit app.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


def load_data(path="sonardata.csv"):
    """Load Sonar dataset."""
    df = pd.read_csv(path, header=None)
    return df


def train_model(df: pd.DataFrame):
    """Train Logistic Regression classifier on Rock vs Mine dataset."""
    # Features and labels
    X = df.drop(columns=60, axis=1)
    y = df[60].apply(lambda v: 1 if v == "M" else 0)  # M=Mine, R=Rock

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"✅ Model trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    return model


def save_model(model, path="model.joblib"):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"💾 Model saved at {path}")


if __name__ == "__main__":
    # Ensure dataset exists
    if not os.path.exists("sonardata.csv"):
        raise FileNotFoundError("Dataset 'sonar.csv' not found!")

    df = load_data()
    model = train_model(df)
    save_model(model)
