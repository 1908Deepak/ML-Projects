"""
main.py
-------
Train multiple regressors for car price prediction and save each trained model separately.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor


def load_data(path: str = "car_data.csv") -> pd.DataFrame:
    """Load car dataset and encode categorical variables."""
    df = pd.read_csv(path)
    df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
    return df


def train_and_save_models(X_train, X_test, y_train, y_test):
    """Train multiple models and save each separately."""
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(alpha=0.5),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0),
        "Bayesian Ridge": BayesianRidge(),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Huber": HuberRegressor()
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores[name] = score
        # Save each model
        filename = f"models/{name.replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"✅ {name} trained & saved ({filename}), R²={score:.4f}")

    return scores


if __name__ == "__main__":
    # Prepare dataset
    df = load_data("car_data.csv")
    X = df.drop(["Car_Name", "Selling_Price"], axis=1)
    y = df["Selling_Price"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train & Save all models
    scores = train_and_save_models(X_train, X_test, y_train, y_test)

    # Display summary
    print("\n📊 Model Performance Summary:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: R² = {score:.4f}")
