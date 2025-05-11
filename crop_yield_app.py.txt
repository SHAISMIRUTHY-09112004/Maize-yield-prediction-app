import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load your saved model

# Load the trained model (make sure the .pkl file is in the same folder)
model = joblib.load("maize_yield_model.pkl")  # change the name if different

st.title("🌽 Maize Yield Prediction App")

st.markdown("Enter the values below to predict maize yield:")

# Input fields for features (you can change based on your model)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=1000.0, step=0.1)
soil_moisture = st.number_input("🌱 Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)
sunlight = st.number_input("☀️ Sunlight (hours/day)", min_value=0.0, max_value=24.0, step=0.1)
ph = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, step=0.1)
fertilizer = st.number_input("🧴 Fertilizer used (kg/hectare)", min_value=0.0, max_value=500.0, step=0.1)
pesticide = st.number_input("🐛 Pesticide used (kg/hectare)", min_value=0.0, max_value=100.0, step=0.1)

# Predict button
if st.button("Predict Yield"):
    features = np.array([[temperature, rainfall, soil_moisture, sunlight, ph, fertilizer, pesticide]])
    prediction = model.predict(features)
    st.success(f"🌽 Predicted Maize Yield: {prediction[0]:.2f} tons/hectare")

st.markdown("---")
st.caption("Created with ❤️ using Streamlit")
