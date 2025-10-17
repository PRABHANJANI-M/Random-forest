# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import pickle

# Load dataset
df = pd.read_csv("cleaned_loan.csv")

# --- Handle categorical features ---
categorical_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Balance dataset (important!) ---
df_majority = df[df["loan_status"] == 1]
df_minority = df[df["loan_status"] == 0]
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# --- Split data ---
X = df_balanced.drop("loan_status", axis=1)
y = df_balanced["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale numeric features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train model ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save model, scaler, and features ---
with open("loan_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns.tolist()), f)

print("💾 Balanced Random Forest model saved as loan_model.pkl")
