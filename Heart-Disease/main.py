# ======================================================
# 1. Importing Necessary Libraries
# ======================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# ======================================================
# 2. Data Loading and Exploration
# ======================================================

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None

def explore_data(data):
    """
    Perform basic data exploration.

    Args:
        data (DataFrame): Input dataset.

    Prints:
        - Head, tail, shape, info, null counts, duplicate counts, and basic statistics.
    """
    print("First 5 rows of the dataset:\n", data.head())
    print("\nLast 5 rows of the dataset:\n", data.tail())
    print("\nDataset shape (rows, columns):", data.shape)
    print("\nDataset information:\n")
    print(data.info())
    print("\nNull values in each column:\n", data.isnull().sum())
    print("\nNumber of duplicate rows:", data.duplicated().sum())
    print("\nStatistical summary of the data:\n", data.describe())
    print("\nDistribution of target variable:\n", data['target'].value_counts())

# ======================================================
# 3. Data Preprocessing
# ======================================================

def preprocess_data(data):
    """
    Split features (X) and target (Y), then split into train-test sets.

    Args:
        data (DataFrame): Input dataset.

    Returns:
        tuple: (X_train, X_test, Y_train, Y_test) - Train and test splits.
    """
    X = data.drop(columns='target', axis=1)
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test

# ======================================================
# 4. Model Training
# ======================================================

def train_model(X_train, Y_train):
    """
    Train a Logistic Regression model.

    Args:
        X_train (DataFrame): Training features.
        Y_train (Series): Training target.

    Returns:
        model: Trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=500)  # Increased max_iter for potential convergence issues
    model.fit(X_train, Y_train)
    return model

# ======================================================
# 5. Model Evaluation
# ======================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    """
    Evaluate the model on training and test data.

    Args:
        model: Trained model.
        X_train, Y_train: Training data.
        X_test, Y_test: Test data.

    Prints:
        - Accuracy on training and test data.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(Y_train, train_pred)
    test_accuracy = accuracy_score(Y_test, test_pred)
    
    print("\nAccuracy on Training Data:", train_accuracy)
    print("Accuracy on Test Data:", test_accuracy)

# ======================================================
# 6. Predictive System
# ======================================================

def make_prediction(model, input_data):
    """
    Make a prediction for a single data point.

    Args:
        model: Trained model.
        input_data (tuple): Input data for prediction.

    Prints:
        - Prediction result.
    """
    input_data_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_array)
    
    if prediction[0] == 0:
        print("\nPrediction: The person does not have heart disease.")
    else:
        print("\nPrediction: The person has heart disease.")

# ======================================================
# 7. Save and Load Model
# ======================================================

def save_model(model, filename='heart_disease_model.sav'):
    """
    Save the trained model to a file.

    Args:
        model: Trained model.
        filename (str): File name to save the model.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename='heart_disease_model.sav'):
    """
    Load a saved model from a file.

    Args:
        filename (str): File name of the saved model.

    Returns:
        model: Loaded machine learning model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# ======================================================
# 8. Main Execution
# ======================================================

if __name__ == "__main__":
    # Step 1: Load the data
    filepath = 'heart.csv'
    heart_data = load_data(filepath)
    
    if heart_data is not None:
        # Step 2: Explore the data
        explore_data(heart_data)
        
        # Step 3: Preprocess the data
        X_train, X_test, Y_train, Y_test = preprocess_data(heart_data)
        
        # Step 4: Train the model
        model = train_model(X_train, Y_train)
        
        # Step 5: Evaluate the model
        evaluate_model(model, X_train, Y_train, X_test, Y_test)
        
        # Step 6: Make a single prediction
        sample_input = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
        make_prediction(model, sample_input)
        
        # Step 7: Save the model
        save_model(model)
        
        # Step 8: Load the saved model and verify
        loaded_model = load_model()
        print("\nModel loaded successfully. Columns used in prediction:")
        for column in heart_data.columns[:-1]:
            print("-", column)
