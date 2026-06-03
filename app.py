import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Load model safely ---
model_path = os.path.join(os.path.dirname(__file__), "gradient_boosting_model.pkl")
model = joblib.load(model_path)

# Expected features (from preprocessing pipeline)
model_expected_features = [
    'height','weight','ap_hi','ap_lo','age_years',
    'gender_2','cholesterol_2','cholesterol_3',
    'gluc_2','gluc_3','smoke_1','alco_1','active_1'
]

# --- Preprocessing function ---
def preprocess_input(user_inputs_dict):
    df = pd.DataFrame([user_inputs_dict])
    # Convert age to age_years
    df['age_years'] = df['age']
    df.drop(columns=['age'], inplace=True)
    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        columns=['gender','cholesterol','gluc','smoke','alco','active'],
        drop_first=True
    )
    # Reindex to expected features
    df = df.reindex(columns=model_expected_features, fill_value=0)
    return df

# --- Streamlit UI ---
st.title("🩺 Cardiovascular Risk Prediction App")
st.write("Enter patient details in the sidebar to generate a risk report.")

with st.sidebar:
    st.header("Patient Information")
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    ap_hi = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol", [1,2,3])
    gluc = st.selectbox("Glucose", [1,2,3])
    gender = st.selectbox("Gender", [1,2])  # 1=female, 2=male
    smoke = st.selectbox("Smoke", [0,1])
    alco = st.selectbox("Alcohol", [0,1])
    active = st.selectbox("Active", [0,1])

# Collect inputs
user_inputs = {
    'age': age, 'height': height, 'weight': weight,
    'ap_hi': ap_hi, 'ap_lo': ap_lo,
    'gender': gender, 'cholesterol': cholesterol, 'gluc': gluc,
    'smoke': smoke, 'alco': alco, 'active': active
}

# Process and predict
processed_input = preprocess_input(user_inputs)
prediction = model.predict(processed_input)[0]
probability = model.predict_proba(processed_input)[0][1] * 100

# --- Display results ---
st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High risk of cardiovascular disease\n\nEstimated probability: {probability:.1f}%")
else:
    st.success(f"✅ Low risk of cardiovascular disease\n\nEstimated probability: {probability:.1f}%")
