#Importing Dependencies
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#Data Collection and Analysis
#PIMA Diabetes Dataset

#loading the dataset to a pandas DataFrame
dataset = pd.read_csv('diabetes.csv')

#checking the first 5 rows of the dataset
print(dataset.head())

#checking the last 5 rows of the dataset
print(dataset.tail())

#number of rows and column in this dataset
print(dataset.shape)

#getting the statistical measures of the data
print(dataset.describe())

#chekcing null values
print(dataset.isna().sum())

#checking total counts of diabetics & non- diabetic patents

#0--> Non- Diabetic
#1--> Diabetic

print(dataset['Outcome'].value_counts())

print(dataset.groupby('Outcome').mean())

# Separating the features and target data
X= dataset.drop(columns='Outcome',axis=1)
Y= dataset['Outcome']

#printing features (X)
print(X)

#printing target (Y)
print(Y)


#Splitting the train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2 , stratify=Y , random_state=2)

print(X.shape, X_train.shape , X_test.shape)

#Training Model

classifier= svm.SVC(kernel='linear')

#training the svm classifier
classifier.fit(X_train, Y_train)

#MODEL EVALUATION

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


# Making Predictive System
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')






# Saving the trained model to current directory

filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')