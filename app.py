import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("salary_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Prediction App")
st.markdown("Predict an employee's **Salary** based on Age and Experience.")

# Input sliders
age = st.slider("Age", 18, 65, 30)
years_exp = st.slider("Years of Experience", 0, 40, 5)

# Prepare input data for prediction (only required features)
input_data = pd.DataFrame({
    "Age": [age],
    "Years of Experience": [years_exp]
})

# Predict
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted Salary: ₹{int(prediction[0]):,}")
    except Exception as e:
        st.error("⚠️ Prediction failed. The model was likely trained on different features.")
        st.code(str(e))
