from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load('../../models/loan_model.pkl')

# Define input schema
class LoanInput(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: bool
    Dependents: str
    Education: str
    Gender: str
    Married: bool
    Property_Area: str
    Self_Employed: str

# Preprocessing functions based on your instructions
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data to match the model's expected features."""
    # Replace 0 in CoapplicantIncome with 1, then take the log
    data.loc[data['CoapplicantIncome'] == 0, 'CoapplicantIncome'] = 1
    data['CoapplicantIncome'] = np.log(data['CoapplicantIncome'])
    # Transform ApplicantIncome
    data['ApplicantIncome'] = np.sqrt(data['ApplicantIncome'])
    # Cube Loan_Amount_Term
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'] ** 3
    
    # Convert Credit_History to integers
    data['Credit_History'] = data['Credit_History'].astype(int)
    
    # Ordinal encoding for Dependents
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    data['Dependents'] = data['Dependents'].map(dependents_map)
    
    # Handle categorical columns manually
    data['Education_Not Graduate'] = 1 if data['Education'].iloc[0] == 'Not Graduate' else 0
    data['Gender_Male'] = 1 if data['Gender'].iloc[0] == 'Male' else 0
    data['Married_Yes'] = 1 if data['Married'].iloc[0] else 0
    data['Property_Area_Semiurban'] = 1 if data['Property_Area'].iloc[0] == 'Semiurban' else 0
    data['Property_Area_Urban'] = 1 if data['Property_Area'].iloc[0] == 'Urban' else 0
    data['Self_Employed_Yes'] = 1 if data['Self_Employed'].iloc[0] == 'Yes' else 0
    
    # Drop unnecessary columns
    data = data.drop(columns=['Education', 'Gender', 'Married', 'Property_Area', 'Self_Employed'], axis=1)
    
    # Ensure correct column order
    expected_columns = [
        "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Education_Not Graduate", 
        "Gender_Male", "Married_Yes", "Property_Area_Semiurban", 
        "Property_Area_Urban", "Self_Employed_Yes"
    ]
    data = data[expected_columns]
    
    return data

# Prediction endpoint
@app.post("/predict/")
async def predict(input: LoanInput):
    # Convert input to DataFrame
    input_data = pd.DataFrame([input.dict()])
    
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Return prediction result
    return {"prediction": int(prediction[0])}
