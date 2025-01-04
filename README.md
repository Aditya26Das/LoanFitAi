# LoanFitAI
A web-based application that predicts loan eligibility based on user-provided information using a Decision Tree Machine Learning model.

# Features
- Accurate Predictions: Achieves ~94% accuracy in loan eligibility predictions.
- Real-Time Analysis: Users receive instant results after entering their details.
- Balanced Dataset: Handles class imbalance using oversampling techniques for reliable predictions.
- Interactive Interface: Built with Streamlit for a seamless user experience.
- Robust Backend: FastAPI ensures high-performance API calls.
- Scalable Deployment: Hosted using Render (backend) and Streamlit Cloud (frontend).

# Dataset Overview
The model is trained on a dataset of approximately 1,100 rows with the following features:

- Dependents: Number of dependents (0, 1, 2, 3+).
- Applicant Income & Co-applicant Income: Income details.
- Loan Amount & Loan Term: Loan specifics.
- Credit History: Credit score information.
- Education: Graduate/Not Graduate.
- Gender, Marital Status, Property Area: Demographics.
- Loan Status: Target variable (Y/N).

# Technologies Used
- Machine Learning: Decision Tree algorithm.
- Frontend: Streamlit for a user-friendly interface.
- Backend: FastAPI for efficient API handling.
- Deployment:
    1. Backend: Render
    2. Frontend: Streamlit Cloud

## Run Locally  
Clone the project  

~~~bash  
  git clone https://github.com/Aditya26Das/LoanFitAi.git
~~~

Go to the project directory  

~~~bash  
  cd LoanFitAi
~~~

Install dependencies  

~~~bash  
pip install -r requirements.txt
~~~

Navigate to ./api/

~~~bash  
cd ./api/
~~~  

Start the backend server  

~~~bash  
uvicorn main:app --reload  
~~~  

Navigate to ./streamlit_app/

~~~bash  
cd ./streamlit_app/
~~~  

Start the server  

~~~bash  
streamlit run app.py  
~~~ 