import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "https://loanfitai.onrender.com/predict/"

# Streamlit app title
st.title("Loan Prediction App")

# Input fields for the user
st.header("Enter Loan Details:")
applicant_income = st.number_input("Applicant Income (e.g., 5000)", min_value=0.0, step=100.0)
coapplicant_income = st.number_input("Coapplicant Income (e.g., 2000)", min_value=0.0, step=100.0)
loan_amount = st.number_input("Loan Amount (e.g., 150)", min_value=0.0, step=1.0)
loan_amount_term = st.number_input("Loan Amount Term (in days, e.g., 360)", min_value=0.0, step=1.0)
credit_history = st.selectbox("Credit History (e.g., Yes/No)", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"])

# Process inputs for API request
if st.button("Predict Loan Eligibility"):
    # Convert inputs to the required format
    input_data = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": True if credit_history == "Yes" else False,
        "Dependents": dependents,
        "Education": education,
        "Gender": gender,
        "Married": True if married == "Yes" else False,
        "Property_Area": property_area,
        "Self_Employed": self_employed
    }
    
    # Send a POST request to the FastAPI endpoint
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            if prediction == 1:
                st.success("Loan Approved! 🎉")
            else:
                st.error("Loan Rejected. 😔")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
