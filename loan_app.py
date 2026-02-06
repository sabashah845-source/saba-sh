import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction App")
def load_data():
    df = pd.read_csv("loan_data.xls")
    return df
# -----------------------------
# Preprocessing
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})

categorical_cols = ["Gender", "Married", "Education", "Self_Employed"]
numerical_cols = ["ApplicantIncome", "LoanAmount", "Credit_History"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# -----------------------------
# Model Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🧾 Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

input_data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [income],
    "LoanAmount": [loan_amount],
    "Credit_History": [credit_history]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Loan Approval"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")