"""
app.py
------
Streamlit app for car price prediction with option to select regression model.
"""

import streamlit as st
import numpy as np
import joblib
import os


MODEL_DIR = "models"


def load_model(model_name: str):
    """Load a selected model from saved files."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"Model {model_name} not found. Please train models with main.py first.")
        return None
    return joblib.load(model_path)


def predict_price(model, features: np.ndarray):
    """Make prediction with selected model."""
    try:
        return float(model.predict([features])[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def main():
    st.set_page_config(page_title="🚗 Car Price Prediction", layout="wide")
    st.title("🚗 Car Price Prediction App")

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Project Details"])

    # ---------------- Prediction Tab ----------------
    with tab1:
        st.header("Enter Car Details")

        year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
        present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
        kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.slider("Number of Previous Owners", 0, 3, 0)

        # Encode categorical
        fuel_dict = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_dict = {"Dealer": 0, "Individual": 1}
        trans_dict = {"Manual": 0, "Automatic": 1}

        features = np.array([
            year, present_price, kms_driven,
            fuel_dict[fuel_type], seller_dict[seller_type],
            trans_dict[transmission], owner
        ])

        # Dropdown to select model
        available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
        model_choice = st.selectbox("Choose Regression Model", available_models)

        if st.button("🔮 Predict Price"):
            model = load_model(model_choice)
            if model:
                price = predict_price(model, features)
                if price is not None:
                    st.success(f"Estimated Selling Price using {model_choice.replace('_',' ').replace('.pkl','')}: ₹ {price:.2f} lakhs")

    # ---------------- Project Details Tab ----------------
    with tab2:
        st.header("📊 Project Details")
        st.markdown("""
        ### 📌 Overview
        This project predicts **car resale prices** using multiple regression models.  
        The user can **choose their preferred algorithm** for prediction.  

        ### 🗂 Dataset
        - Source: `car data.csv`  
        - Features: Year, Present Price, Kms Driven, Fuel Type, Seller Type, Transmission, Owner  
        - Target: Selling Price  

        ### ⚙️ Algorithms Available
        - Linear Regression, Lasso, Ridge, ElasticNet  
        - Decision Tree, Random Forest  
        - XGBoost, LightGBM, CatBoost  
        - Bayesian Ridge, Gradient Boosting, Huber  

        ### 👨‍💻 Author
        - Name: Deepak Singh  
        - Role: Data Science & ML Enthusiast  
        - GitHub: [1908Deepak](https://github.com/1908Deepak)  

        ### 🚀 Future Improvements
        - Add deep learning models  
        - Deploy full-stack (React + Flask/FASTAPI backend)  
        - Add visualizations comparing model performance  
        """)


if __name__ == "__main__":
    main()
