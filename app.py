# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="🏦 Smart Loan Approval Predictor", layout="centered")
st.title("🏦 Smart Loan Approval Prediction App")

# --- Load trained model ---
with open("loan_model.pkl", "rb") as f:
    model, scaler, feature_names = pickle.load(f)

st.header("📋 Applicant Information")

# --- User input fields ---
person_age = st.slider("👤 Applicant Age", 18, 70, 25)
person_gender = st.selectbox("👩 Gender", ["male", "female"])
person_education = st.selectbox("🎓 Education Level", ["High School", "Bachelor", "Master", "PhD"])
person_income = st.number_input("💰 Annual Income (USD)", min_value=0.0, value=50000.0)
person_emp_exp = st.slider("🧑‍💼 Years of Work Experience", 0, 50, 2)
person_home_ownership = st.selectbox("🏠 Home Ownership", ["RENT", "OWN", "MORTGAGE"])
loan_amnt = st.number_input("🏦 Loan Amount", min_value=500.0, value=10000.0)
loan_intent = st.selectbox(
    "🎯 Loan Purpose",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
)
loan_int_rate = st.slider("📈 Interest Rate (%)", 5.0, 30.0, 12.0)
loan_percent_income = st.slider("💸 Loan Percent of Income", 0.0, 1.0, 0.3)
cb_person_cred_hist_length = st.slider("📊 Credit History Length (years)", 0, 50, 3)
credit_score = st.slider("⭐ Credit Score", 300, 850, 600)
previous_loan_defaults_on_file = st.selectbox("⚠️ Previous Loan Default?", ["No", "Yes"])

# --- Prepare input dictionary ---
input_dict = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    # One-hot encoded categorical variables
    "person_gender_male": 1 if person_gender == "male" else 0,
    "person_education_Bachelor": 1 if person_education == "Bachelor" else 0,
    "person_education_Master": 1 if person_education == "Master" else 0,
    "person_education_PhD": 1 if person_education == "PhD" else 0,
    "person_home_ownership_OWN": 1 if person_home_ownership == "OWN" else 0,
    "person_home_ownership_RENT": 1 if person_home_ownership == "RENT" else 0,
    "loan_intent_EDUCATION": 1 if loan_intent == "EDUCATION" else 0,
    "loan_intent_MEDICAL": 1 if loan_intent == "MEDICAL" else 0,
    "loan_intent_PERSONAL": 1 if loan_intent == "PERSONAL" else 0,
    "loan_intent_VENTURE": 1 if loan_intent == "VENTURE" else 0,
    "loan_intent_DEBTCONSOLIDATION": 1 if loan_intent == "DEBTCONSOLIDATION" else 0,
    "loan_intent_HOMEIMPROVEMENT": 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
    "previous_loan_defaults_on_file_Yes": 1 if previous_loan_defaults_on_file == "Yes" else 0,
}

# --- Fill missing columns (important) ---
for col in feature_names:
    if col not in input_dict:
        input_dict[col] = 0

# --- Arrange columns correctly ---
input_df = pd.DataFrame([input_dict])[feature_names]
input_scaled = scaler.transform(input_df)

# --- Predict ---
if st.button("🔮 Predict Loan Status"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved.")

st.markdown("---")
st.caption("Model powered by Random Forest Classifier · Built with ❤️ using Streamlit")
