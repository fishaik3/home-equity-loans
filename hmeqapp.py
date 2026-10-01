# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("LoanApprovalModel.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# App title
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; "
    "padding: 10px; color: #cc0000;'>"
    "<b>Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Loan Applicant's Details")

# -----------------------------
# USER INPUTS
# -----------------------------

reason = st.selectbox(
    "Reason for Loan",
    ["Home Improvement", "Debt Consolidation", "Other"]
)

requested_loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=0.0,
    step=1000.0
)

fico_score = st.number_input(
    "FICO Score",
    min_value=300,
    max_value=850,
    step=1
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed", "Other"]
)

employment_sector = st.selectbox(
    "Employment Sector",
    ["Private", "Public", "Other"]
)

monthly_gross_income = st.number_input(
    "Monthly Gross Income",
    min_value=0.0,
    step=100.0
)

monthly_housing_payment = st.number_input(
    "Monthly Housing Payment",
    min_value=0.0,
    step=100.0
)

bankruptcy_foreclosure = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    ["Yes", "No"]
)

lender = st.selectbox(
    "Lender",
    ["A", "B", "C"]
)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [bankruptcy_foreclosure],
    "Lender": [lender]
})

# -----------------------------
# ONE-HOT ENCODING
# -----------------------------

input_encoded = pd.get_dummies(
    input_data,
    columns=[
        "Reason",
        "Employment_Status",
        "Employment_Sector",
        "Ever_Bankrupt_or_Foreclose",
        "Lender"
    ],
    drop_first=True
)

# Match model's expected columns
model_columns = model.feature_names_in_

input_encoded = input_encoded.reindex(
    columns=model_columns,
    fill_value=0
)

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Evaluate Loan"):

    # Scale input exactly as training data was scaled
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success("The prediction is: Approved ✅")
    else:
        st.error("The prediction is: Not Approved ❌")

    st.write(
        f"Approval Probability: **{probability:.1%}**"
    )
