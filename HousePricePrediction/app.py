"""
app.py
------
Streamlit app for predicting house prices using a trained model.
"""

import streamlit as st
import numpy as np
import joblib
import os


# -------------------------------
# Utility Functions
# -------------------------------

@st.cache_resource
def load_model(model_path="model.pkl"):
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run main.py to train the model first.")
        return None
    return joblib.load(model_path)


def predict(model, features: np.ndarray):
    """Predict house price given features."""
    try:
        prediction = model.predict([features])
        return float(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# -------------------------------
# Streamlit UI
# -------------------------------

def main():
    st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
    st.title("🏠 House Price Prediction App")

    # Tabs for navigation
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Project Details"])

    # ------------------- Prediction Tab -------------------
    with tab1:
        st.header("Enter Housing Details for Prediction")
        st.write("Fill in the housing features below to estimate the price (California Housing dataset).")

        col1, col2 = st.columns(2)
        with col1:
            MedInc = st.number_input("Median Income (10k$)", min_value=0.0, max_value=20.0, step=0.1)
            HouseAge = st.slider("House Age (years)", 1, 100, 20)
            AveRooms = st.number_input("Avg Rooms", min_value=1.0, max_value=50.0, step=0.1)
            AveBedrms = st.number_input("Avg Bedrooms", min_value=1.0, max_value=10.0, step=0.1)
        with col2:
            Population = st.number_input("Population", min_value=1.0, max_value=50000.0, step=1.0)
            AveOccup = st.number_input("Avg Occupants per Household", min_value=1.0, max_value=20.0, step=0.1)
            Latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, step=0.01)
            Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-100.0, step=0.01)

        features = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])

        model = load_model()

        if model and st.button("🔮 Predict Price"):
            price = predict(model, features)
            if price:
                st.success(f"Estimated House Price: ${price:,.2f} (in 100k USD units)")

    # ------------------- Project Details Tab -------------------
    with tab2:
        st.header("📊 Project Details")
        st.markdown("""
        ### 📌 Overview
        This project demonstrates a **House Price Prediction System** built with:
        - **Dataset**: [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (from Scikit-learn)  
        - **Algorithm**: XGBoost Regressor  
        - **Language/Frameworks**: Python, Scikit-learn, XGBoost, Streamlit  

        ### ⚙️ How It Works
        1. The dataset contains housing data like median income, house age, number of rooms, latitude, longitude, etc.  
        2. An **XGBoost regression model** is trained to predict house prices.  
        3. Users input values, and the trained model predicts the price in real-time.  

        ### 👨‍💻 Author
        - **Name**: Deepak Singh  
        - **Role**: Data Science & Machine Learning Enthusiast  
        - **GitHub**: [1908Deepak](https://github.com/1908Deepak)  

        ### 🚀 Future Improvements
        - Add more advanced ML/DL models  
        - Deploy as a full-stack app with React + Flask backend  
        - Integrate interactive visualizations  
        """)


# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    main()
