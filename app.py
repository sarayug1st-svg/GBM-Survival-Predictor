import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("gbm_synthetic_clinical_data.csv")
    return df

# Train model and encoders
@st.cache_resource
def train_model(df):
    label_encoders = {}
    for col in ["Sex", "Treatment_Type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[["Age", "Sex", "IDH1_Status", "MGMT_Status", "Tumor_Size_cm", "KPS_Score", "Treatment_Type"]]
    y = df["Overall_Survival_Months"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, label_encoders

# App interface
st.title("GBM Survival Predictor")
st.write("This tool uses simulated clinical data to estimate overall survival for glioblastoma patients based on biomarker and clinical inputs.")

data = load_data()
model, encoders = train_model(data)

# User input
age = st.slider("Age", 18, 90, 60)
sex = st.selectbox("Sex", ["Male", "Female"])
tumor_size = st.slider("Tumor Size (cm)", 1.0, 6.0, 3.5)
kps = st.slider("Karnofsky Performance Score (KPS)", 0, 100, 80)
mgmt = st.selectbox("MGMT Promoter Methylation", ["Methylated", "Unmethylated"])
idh1 = st.selectbox("IDH1 Mutation", ["Mutant", "Wildtype"])
treatment = st.selectbox("Treatment Type", ["Surgery only", "RT", "RT+TMZ", "TMZ only"])

# Transform input
sex_encoded = encoders["Sex"].transform([sex])[0]
treatment_encoded = encoders["Treatment_Type"].transform([treatment])[0]
mgmt_encoded = 1 if mgmt == "Methylated" else 0
idh1_encoded = 1 if idh1 == "Mutant" else 0

features = [[age, sex_encoded, idh1_encoded, mgmt_encoded, tumor_size, kps, treatment_encoded]]
prediction = model.predict(features)[0]

# Output
st.subheader("Predicted Overall Survival")
st.write(f"**{prediction:.1f} months**")
