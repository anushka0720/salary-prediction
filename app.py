import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary Prediction App")

# Inputs
age = st.number_input("Age", min_value=18, max_value=60)
gender = st.selectbox("Gender", encoder["Gender"].classes_)
education = st.selectbox("Education Level", encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title", encoder["Job Title"].classes_)
years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=40)

# Create dataframe (⚠️ Column names must EXACTLY match training data)
df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [years_of_exp]   # ✅ Correct spelling
})

if st.button("Predict"):
    
    # Encode categorical columns
    for col in encoder:
        df[col] = encoder[col].transform(df[col])
    
    # Ensure correct column order (very important)
    df = df[model.feature_names_in_]

    prediction = model.predict(df)
    
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
