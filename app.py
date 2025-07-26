import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
mmodel = joblib.load("rf_gbm_model_py13.pkl")
label_encoders = joblib.load("label_encoders_py13.pkl")

st.title("GBM Survival Prediction Tool")
st.write("Estimate survival time based on patient features and biomarkers.")

# User inputs
age = st.slider("Age", 18, 85, 60)
sex = st.selectbox("Sex", ["Male", "Female"])
idh1 = st.selectbox("IDH1 Mutation", ["Wild-type (0)", "Mutated (1)"])
mgmt = st.selectbox("MGMT Methylation", ["Unmethylated (0)", "Methylated (1)"])
tumor_size = st.slider("Tumor Size (cm)", 1.0, 6.0, 3.5)
kps = st.slider("KPS Score", 0, 100, 80)
treatment = st.selectbox("Treatment Type", ["Surgery only", "RT", "RT+TMZ", "TMZ only"])

# Encode categorical variables
encoded_sex = label_encoders["Sex"].transform([sex])[0]
encoded_treatment = label_encoders["Treatment_Type"].transform([treatment])[0]
idh1_val = int(idh1[-2])
mgmt_val = int(mgmt[-2])

# Create DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": encoded_sex,
    "IDH1_Status": idh1_val,
    "MGMT_Status": mgmt_val,
    "Tumor_Size_cm": tumor_size,
    "KPS_Score": kps,
    "Treatment_Type": encoded_treatment
}])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Survival: {prediction:.1f} months")
