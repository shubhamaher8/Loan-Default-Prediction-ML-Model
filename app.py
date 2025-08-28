import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and preprocessors
model = joblib.load('trained models/xgboost_model.joblib')
scaler = joblib.load('trained models/scaler.joblib')
encoders = joblib.load('trained models/label_encoders.joblib')
model_columns = joblib.load('trained models/model_columns.joblib')

st.title("Loan Default Risk Prediction")

st.write("Please provide the following details for the loan application:")

with st.form("loan_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    CreditScore = st.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=650)
    MonthsEmployed = st.number_input("Months Employed", min_value=0, value=12)
    NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=3)
    InterestRate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.0)
    LoanTerm = st.number_input("Loan Term (in months)", min_value=1, value=36)
    DTIRatio = st.number_input("Debt-to-Income Ratio (e.g., 0.4)", min_value=0.0, max_value=10.0, value=0.4)
    Education = st.selectbox("Education", ["PhD", "Master's", "Bachelor's", "High School"])
    EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    HasMortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
    HasDependents = st.selectbox("Has Dependents", ["Yes", "No"])
    LoanPurpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Business", "Education", "Other"])
    HasCoSigner = st.selectbox("Has Co-Signer", ["Yes", "No"])
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Age': Age,
        'Income': Income,
        'LoanAmount': LoanAmount,
        'CreditScore': CreditScore,
        'MonthsEmployed': MonthsEmployed,
        'NumCreditLines': NumCreditLines,
        'InterestRate': InterestRate,
        'LoanTerm': LoanTerm,
        'DTIRatio': DTIRatio,
        'Education': Education,
        'EmploymentType': EmploymentType,
        'MaritalStatus': MaritalStatus,
        'HasMortgage': HasMortgage,
        'HasDependents': HasDependents,
        'LoanPurpose': LoanPurpose,
        'HasCoSigner': HasCoSigner
    }
    input_df = pd.DataFrame([input_data])

    # Feature engineering
    input_df['Loan_to_Income_Ratio'] = input_df['LoanAmount'] / (input_df['Income'] + 1e-6)

    # Binary encoding
    for col in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        input_df[col] = 1 if input_df[col].iloc[0].lower() == 'yes' else 0

    # Label encoding
    for col, le in encoders.items():
        try:
            input_df[col] = le.transform(input_df[col])
        except Exception:
            st.error(f"Unknown category '{input_df[col].iloc[0]}' in column '{col}'. Please select a valid option.")
            st.stop()

    # Align columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("Prediction Result: HIGH RISK - Loan is likely to DEFAULT.")
    else:
        st.success("Prediction Result: LOW RISK - Loan is likely to be REPAID.")

    st.info(f"Confidence: Probability of Non-Default: {prediction_proba[0][0]:.2%} | Probability of Default: {prediction_proba[0][1]:.2%}")