**Streamlit app link**:https://random-forest-cdcb97ymrzcea5bj76bz7v.streamlit.app/

**🏦 Loan Approval Prediction using Random Forest**

📘 Overview

This project predicts whether a loan application will be approved or not using a Random Forest Classifier.
It analyzes applicant details such as income, education, credit history, and employment experience to determine loan eligibility.
The system can also calculate the probability of approval for new applicants based on their inputs.

**⚙️ Features Used**

Feature	Description
person_age	Age of the applicant
person_gender	Gender (male or female)
person_education	Educational qualification (High School, Associate, Bachelor, Master)
person_income	Applicant’s annual income
person_emp_exp	Years of employment experience
person_home_ownership	Home ownership type (RENT, OWN, MORTGAGE)
loan_amnt	Requested loan amount
loan_intent	Purpose of the loan (Personal, Education, Medical, Debt Consolidation, Home Improvement)
loan_int_rate	Interest rate applied to the loan
loan_percent_income	Ratio of loan amount to income
cb_person_cred_hist_length	Credit history length in years
credit_score	Credit score of the applicant
previous_loan_defaults_on_file	Indicates if any previous loan default occurred (Yes or No)
Target Variable – loan_status	Represents whether the loan was approved (1) or rejected (0)

**🧠 Model Used**

Random Forest Classifier – a machine learning ensemble technique that combines multiple decision trees to improve accuracy and reduce overfitting.
It is used here because:

It handles both categorical and numerical data efficiently.

It provides robust predictions even with noisy data.

It allows us to identify which features most influence loan approval.

**🧩 Project Workflow**

Data Preprocessing:

Missing value handling

Encoding categorical variables using Label Encoding

Splitting the dataset into training and testing sets

Model Training:

Training a Random Forest Classifier using the processed data

**Evaluation:**

The model’s performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

A confusion matrix is generated to visualize correct and incorrect predictions.

**User Prediction:**

The system prompts the user to enter details for a new applicant.

Based on the input, the trained model predicts the loan approval status and probability.

**📊 Evaluation Metrics**
Metric	Meaning
Accuracy	Overall percentage of correct predictions
Precision	How many predicted approvals were actually approved
Recall	How many actual approvals were correctly predicted
F1-Score	Balance between Precision and Recall
Confusion Matrix	Table showing True Positives, False Positives, True Negatives, and False Negatives
**🧮 Example Output**

When the user enters applicant details (such as income, education, and credit score), the system outputs:

Approval Probability – the likelihood that the loan will be approved (e.g., 0.86).

Loan Approval Decision – whether the loan is Approved or Rejected.

**💡 Insights**

Higher credit score and credit history length increase approval chances.

Lower loan-to-income ratio (loan_percent_income) is associated with higher approval probability.

Applicants with stable employment and no previous defaults are more likely to be approved.

**🌟 Future Enhancements**

Add a Streamlit web interface for user-friendly interaction.

Include hyperparameter tuning (e.g., GridSearchCV) for better accuracy.

Integrate model saving and loading with Pickle for deployment.

Provide feature importance visualization to show which variables influence decisions the most.
