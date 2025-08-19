import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
from PIL import Image  # For image handling

# ======================================================
# 1. Load the Saved Model
# ======================================================
def load_model(filepath='heart_disease_model.sav'):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# ======================================================
# 2. Predictive System
# ======================================================
def make_prediction(model, input_data):
    input_data_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_array)
    if prediction[0] == 0:
        return "The person does not have heart disease."
    else:
        return "The person has heart disease."

# ======================================================
# 3. Streamlit App
# ======================================================
def main():
    # Page title
    st.title("Heart Disease Prediction App")
    
    # Layout: Create two columns
    left_col, right_col = st.columns([3, 1])  # Wider left column, narrower right column
    
    # Left column: Input fields and prediction button
    with left_col:
        st.subheader("Enter the Health Parameters:")
        age = st.text_input("Age")
        sex = st.text_input("Sex (0: Female, 1: Male)")
        cp = st.text_input("Chest Pain Type (0-3)")
        trestbps = st.text_input("Resting Blood Pressure (mm Hg)")
        chol = st.text_input("Cholesterol Level (mg/dl)")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)")
        restecg = st.text_input("Resting ECG Results (0-2)")
        thalach = st.text_input("Max Heart Rate Achieved")
        exang = st.text_input("Exercise Induced Angina (1: Yes, 0: No)")
        oldpeak = st.text_input("ST Depression Induced by Exercise")
        slope = st.text_input("Slope of the Peak Exercise ST Segment (0-2)")
        ca = st.text_input("Number of Major Vessels (0-3)")
        thal = st.text_input("Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)")
        
        # Prediction button
        if st.button("Predict Heart Disease"):
            # Validate inputs
            if all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
                try:
                    # Convert inputs to floats
                    input_data = [
                        float(age), float(sex), float(cp), float(trestbps), float(chol),
                        float(fbs), float(restecg), float(thalach), float(exang),
                        float(oldpeak), float(slope), float(ca), float(thal)
                    ]
                    # Load model and make prediction
                    model = load_model()
                    result = make_prediction(model, input_data)
                    st.success(result)
                except ValueError:
                    st.error("Please enter valid numeric values.")
            else:
                st.warning("All fields are required for prediction.")
    
    # Right column: Display an image
    with right_col:
        st.subheader("Stay Healthy!")
        image = Image.open("Heart-Disease/heart_image.jpg")  # Replace with your image path
        st.image(image, caption="Heart Health Awareness", use_container_width=True)

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
