import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()




@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

with header:
    st.title("welme to the project")
    st.text("In this project I look inot the transaction of ")



with dataset:
    st.header("NYC TAXi dataset")
    st.text("the dataset")
    diabetes = pd.read_csv('diabetes.csv')
    data = load_data('filepath')
    st.write(diabetes.head())

    st.subheader("Glucose level of each and every patient")
    st.bar_chart(diabetes['Glucose'])
    
    st.subheader("Blood Pressure of each patient")
    blood_pressure = pd.DataFrame(diabetes['BloodPressure'].value_counts().head(50))
    st.bar_chart(blood_pressure)






with features:
    st.header('The fetaues created')
    st.markdown("# Welcome to Streamlit")  # Markdown heading
    st.markdown("This is a simple **markdown** example.")


with modelTraining:
    st.header('Model Training')
    st.text("Here you get to chose")

    sel_col , disp_col = st.columns(2)

    max_depth = sel_col.slider("what should be the max_depth of the model ",min_value=10,max_value=100,step=10)

    n_estimators = sel_col.selectbox("How many tress should there be ? ", options=[100,200,300,'no limits'],index=0)

    input_feature = sel_col.text_input("Which feature should be used as the input features",'Insulin')


    regr = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)

    x=diabetes[[input_feature]]
    y=diabetes['Glucose']

    regr.fit(x,y)
    prediction=regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y,prediction))

    disp_col.subheader("Mean squared error of the model is:")
    disp_col.write(mean_squared_error(y,prediction))

    disp_col.subheader("R2 Score")
    disp_col.write(r2_score(y,prediction))


