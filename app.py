# making ui using streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle

#saving the model
with open('heart-disease-model.pkl','rb') as file:
    model,scaler=pickle.load(file)

 #define the columns
categorical_cols=['Gender','ChestPainType','FastingBS','RestingECG','ExerciseAngina','ST_Slope','MajorVessels','Thalassemia']
numerical_cols=['Age','Cholesterol','RestingBp','MaxHR','ST_Depression']

st.set_page_config(page_title="Heart Disease Prediction")
st.write("this app predict the likelihood of heart disease based on patient data")
st.markdown("--")

# collect input data
col1,col2=st.columns(2)
with col1:
        Age=st.number_input("Age",20,100,45)
        Gender=st.selectbox("Gender",["Male","Female"])
        ChestPainType=st.selectbox("Chest Pain Type",[0,1,2,3])
        RestingBp=st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
        Cholesterol=st.number_input("Cholesterol (mg/dL)",100,600,200)
with col2:
      FastingBS=st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])
      RestingECG=st.selectbox("Resting ECG Results",[0,1,2])
      MaxHR=st.number_input("Maximum Heart Rate",60,220,150)
      ExerciseAngina=st.selectbox("Exercise-Induced Angina",[0,1])
      ST_Depression=st.number_input("ST Depression",0.0,6.0,1.0,step=0.1)
      ST_Slope=st.selectbox("ST Slope",[0,1,2])
      MajorVessels=st.selectbox("Major Vessels (0-3)",[0,1,2,3])
      Thalassemia=st.selectbox("Thalassemia(1-3)",[1,2,3])
        
Gender=1 if Gender == 'Male' else 0

#create data frame
input_dict= {
      'Age':Age,
      'Gender':Gender,
      'ChestPainType':ChestPainType,
      'RestingBp':RestingBp,
      'Cholesterol':Cholesterol,
      'FastingBS':FastingBS,
      'RestingECG':RestingECG,
      'MaxHR':MaxHR,
      'ExerciseAngina': ExerciseAngina,
      'ST_Depression': ST_Depression,
      'ST_Slope':ST_Slope,
      'MajorVessels':MajorVessels,
      'Thalassemia':Thalassemia,
      }


input_df=pd.DataFrame([input_dict])
input_encoded=pd.get_dummies(input_df,columns=categorical_cols,drop_first=True)

expected_encoded=model.feature_names_in_
input_encoded=input_encoded.reindex(columns=expected_encoded,fill_value=0)

#scale numeric feature
input_encoded[numerical_cols]=scaler.transform(input_encoded[numerical_cols])

#prediction button

if st.button("predict heart disease"):
      prediction=model.predict(input_encoded)[0]

      if prediction==1:
            st.error("High risk of heart disease")
      else:
            st.success("No sign of heart disease")

st.caption("Develop by Eman Asif | @2026 | Machine Learning Project")
