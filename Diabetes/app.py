import numpy as np
import pickle
import streamlit as st

# Load the saved model
def load_model(model_path = 'diabetes_model.sav'):
    """
    Load the saved machine learning model from the specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function for disease prediction
def predict_disease(input_data, model):
    """
    Predict whether a person has diabetes based on input data.

    Args:
        input_data (list): List of input features for the prediction.
        model: The trained machine learning model.

    Returns:
        str: Prediction result as a string.
    """
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

# Main Streamlit app
def main():
    """
    Main function for the Streamlit application.
    """
    # App title
    st.title("Diabetes Prediction App")
    st.markdown(
        """
        This application uses a machine learning model to predict whether a person is diabetic 
        based on the following health parameters:
        - Number of pregnancies
        - Glucose level
        - Blood pressure
        - Skin thickness
        - Insulin level
        - BMI
        - Diabetes pedigree function
        - Age
        """
    )
    
    # Collect user inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Level')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Level')
    Age = st.text_input('Age')
    
    # Placeholder for the prediction result
    diagnosis = ''

    # Create a button for prediction
    if st.button('Get Diabetes Test Result'):
        # Validate inputs
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            try:
                # Convert inputs to floats and make predictions
                input_data = [
                    float(Pregnancies), float(Glucose), float(BloodPressure),
                    float(SkinThickness), float(Insulin), float(BMI),
                    float(DiabetesPedigreeFunction), float(Age)
                ]
                diagnosis = predict_disease(input_data, loaded_model)
            except ValueError:
                diagnosis = "Please enter valid numeric values for all fields."
        else:
            diagnosis = "All fields are required for prediction."
    
    # Display the result
    st.success(diagnosis)

# Entry point for the script
if __name__ == '__main__':
    # Load the model at the start
    loaded_model = load_model('diabetes_model.sav')
    main()
