import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Directories
DATA_DIR = "../data"
MODEL_DIR = "../models"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Numerical and categorical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_columns = ['Credit_History', 'Dependents', 'Education', 'Gender', 'Loan_Status', 'Married', 'Property_Area', 'Self_Employed']

def drop_loan_id(df):
    """Remove the Loan_ID column."""
    df.drop('Loan_ID', axis=1, inplace=True)
    
    # print(f"Rows remaining after 1st: {df.shape}")
    return df

def impute_null_values(df):
    """Impute missing values."""
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            df[column]=df[column].fillna(df[column].mode()[0])
        else:
            df[column]=df[column].fillna(df[column].median())
    
    # print(f"Rows remaining after 2nd: {df.shape}")
    return df

def remove_outliers(df):
    """Remove outliers using the IQR method."""
    for col in numerical_columns:
        Q1 = df['LoanAmount'].quantile(0.25)
        Q3 = df['LoanAmount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[df['LoanAmount'] > lower_limit]
        df = df[df['LoanAmount'] < upper_limit]
    df.to_csv(f"{DATA_DIR}/processed/processed.csv", index=False)
    
    # print(f"Rows remaining after 3rd: {df.shape}")
    return df

def feature_transform(df):
    """Transform features."""
    df.loc[df['CoapplicantIncome'] == 0, 'CoapplicantIncome'] = 1
    df['ApplicantIncome'] = np.sqrt(df['ApplicantIncome'])
    df['CoapplicantIncome'] = np.log(df['CoapplicantIncome'])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'] ** 3
    
    # print(f"Rows remaining after 4th: {df.shape}")
    return df

def feature_scaling(df):
    """Scale numerical features."""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    # print(f"Rows remaining after 5th: {df.shape}")
    return df

def encoding_categorical_data(df):
    """Encode categorical data."""
    label_encoder = LabelEncoder()
    df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])
    
    ordinal_mapping = [['0', '1', '2', '3+']]
    ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)
    df['Dependents'] = ordinal_encoder.fit_transform(df[['Dependents']])
    
    nominal_cols = ['Credit_History', 'Education', 'Gender', 'Married', 'Property_Area', 'Self_Employed']
    df_nominal_encoded = pd.get_dummies(df[nominal_cols], drop_first=True)
    df = pd.concat([df.drop(columns=nominal_cols), df_nominal_encoded], axis=1)
    df.to_csv(f"{DATA_DIR}/final/final.csv", index=False)
    # print(f"Rows remaining after 6th: {df.shape}")
    return df

def split_data(df):
    """Split the data into training and testing sets."""
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"Rows remaining after 7th: {X_train.shape, X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a decision tree classifier."""
    model = DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_leaf=1,min_samples_split=2,random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/loan_model.pkl")
    print(type(model))
    return model

def main():
    # Load data
    df = pd.read_csv(f"{DATA_DIR}/raw/loan_dataset_raw.csv")
    
    # Apply preprocessing pipeline
    df = drop_loan_id(df)
    df = impute_null_values(df)
    df = remove_outliers(df)
    df = feature_transform(df)
    df = feature_scaling(df)
    df = encoding_categorical_data(df)
    
    # Split data
    X_train, _ , y_train, _ = split_data(df)
    
    # Train and save the model
    train_model(X_train, y_train)

if __name__ == "__main__":
    main()
