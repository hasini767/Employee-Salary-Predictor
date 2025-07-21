
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv("employee_data.csv")
X = df[["Job Title", "Experience (Years)", "Education Level", "Location"]]
y = df["Salary (USD)"]
categorical_cols = ["Job Title", "Education Level", "Location"]
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    .salary-box {
        background-color: #e0f7fa;
        padding: 1.2rem;
        border-left: 6px solid #007c91;
        font-size: 1.3rem;
        margin-top: 1rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)
st.title("EMPLOYEE SALARY PREDICTOR")
st.subheader("Know your worth before stepping into the interview!")

st.markdown("---")
st.markdown("### Enter your professional details")
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", sorted(df["Job Title"].unique()))
    education = st.selectbox("Education Level", sorted(df["Education Level"].unique()))

with col2:
    experience = st.slider("Experience (Years)", 0, 40, 2)
    location = st.selectbox("Location", sorted(df["Location"].unique()))
sample = pd.DataFrame([[job_title, experience, education, location]], columns=X.columns)
if st.button("Estimate Salary"):
    predicted_salary = model.predict(sample)[0]
    st.markdown(f"<div class='salary-box'>Estimated Salary: <strong>${predicted_salary:,.2f}</strong></div>", unsafe_allow_html=True)
st.markdown("---")
st.caption(f"Wanna check the accuracy? (R2 score measure): {r2:.4f}")
