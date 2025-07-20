import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv("employee_data_synthetic.csv")
X = df[["Job Title", "Experience (Years)", "Education Level", "Location"]]
y = df["Salary (USD)"]
categorical_cols = ["Job Title", "Education Level", "Location"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("EMPLOYEE SALARY PREDICTOR")
st.write("Know your worth -before the interview!")
job_title = st.selectbox("Job Title", sorted(df["Job Title"].unique()))
experience = st.slider("Experience (Years)", 0, 40, 2)
education = st.selectbox("Education Level", sorted(df["Education Level"].unique()))
location = st.selectbox("Location", sorted(df["Location"].unique()))
sample = pd.DataFrame([[job_title, experience, education, location]], columns=X.columns)
if st.button("Estimate Salary"):
    predicted_salary = model.predict(sample)[0]
    st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f}")
st.markdown(f"📊 **Model R² Score (on training data):** `{r2:.4f}`")
