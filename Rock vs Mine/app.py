"""
app.py
------
Streamlit app for Rock vs Mine classification using trained ML model.
"""

import streamlit as st
import numpy as np
import joblib
import os


@st.cache_resource
def load_model(path="model.joblib"):
    """Load trained model from file."""
    if not os.path.exists(path):
        st.error("❌ Model file not found. Run main.py to train the model first.")
        return None
    return joblib.load(path)


def predict(model, features: np.ndarray):
    """Make prediction with trained model."""
    pred = model.predict([features])[0]
    return "Mine" if pred == 1 else "Rock"


def main():
    st.set_page_config(page_title="⛏ Rock vs Mine Classifier", layout="wide")
    st.title("⛏ Rock vs Mine Prediction App")

    # Tabs for navigation
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Project Details"])

    # ---------------- Prediction ----------------
    with tab1:
        st.header("Enter 60 Sonar Features")

        inputs = []
        for i in range(60):
            val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01)
            inputs.append(val)

        features = np.array(inputs)

        model = load_model()
        if model and st.button("🔮 Predict"):
            result = predict(model, features)
            st.success(f"Prediction: {result}")

    # ---------------- Project Details ----------------
    with tab2:
        st.header("📊 Project Details")
        st.markdown("""
        ### 📌 Overview
        This project predicts whether a sonar signal is reflected from a **Rock (R)** or a **Mine (M)**.  

        ### 🗂 Dataset
        - **Name**: Sonar Dataset (UCI Machine Learning Repository)  
        - **Features**: 60 numeric attributes (0–1 values) representing energy in frequency bands.  
        - **Target**:  
            - `R` → Rock  
            - `M` → Mine  

        ### ⚙️ Algorithm
        - **Model Used**: Logistic Regression (Scikit-learn)  
        - Optimized for binary classification (Rock vs Mine).  

        ### 👨‍💻 Author
        - Name: Deepak Singh  
        - Role: Data Science & ML Enthusiast  
        - GitHub: [1908Deepak](https://github.com/1908Deepak)  

        ### 🚀 Future Improvements
        - Try more models (Random Forest, XGBoost, Neural Networks)  
        - Deploy as full-stack web app with Flask/React backend  
        - Add feature importance visualization  
        """)


if __name__ == "__main__":
    main()
