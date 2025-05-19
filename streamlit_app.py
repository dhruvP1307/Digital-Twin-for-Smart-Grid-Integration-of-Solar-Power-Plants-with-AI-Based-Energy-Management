import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor

# Load pre-trained models (this example retrains them from scratch for simplicity)
@st.cache_resource
def load_models():
    # Dummy training data (replace with your CSV or real data)
    df = pd.read_csv("synthetic_solar_power_dataset.csv")
    X = df.drop(columns=['Solar Power Output (kW)'])
    y = df['Solar Power Output (kW)']

    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    svm = SVR()
    rf = RandomForestRegressor(random_state=42)
    catboost = CatBoostRegressor(verbose=0, random_state=42)

    svm.fit(X_scaled, y)
    rf.fit(X_scaled, y)
    catboost.fit(X, y)

    ensemble = VotingRegressor([('svm', svm), ('rf', rf), ('catboost', catboost)])
    ensemble.fit(X, y)

    return scaler, svm, rf, catboost, ensemble

# UI
st.title("Smart Grid: Solar Energy Prediction")
st.markdown("Predict solar power output (kW) using smart grid input parameters.")

# User Inputs
irradiance = st.slider("Solar Irradiance (W/m²)", 0, 1000, 600)
temperature = st.slider("Temperature (°C)", -10, 50, 25)
voltage = st.slider("Voltage (V)", 100, 250, 220)
current = st.slider("Current (A)", 0, 20, 10)
wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)

# Prediction
input_features = np.array([[irradiance, temperature, voltage, current, wind_speed]])
scaler, svm, rf, catboost, ensemble = load_models()

input_scaled = scaler.transform(input_features)

svm_pred = svm.predict(input_scaled)[0]
rf_pred = rf.predict(input_scaled)[0]
catboost_pred = catboost.predict(input_features)[0]
ensemble_pred = ensemble.predict(input_features)[0]

# Results
st.subheader("Predicted Energy Output (kW)")
st.write(f"🔹 SVM: `{svm_pred:.2f}` kW")
st.write(f"🔹 Random Forest: `{rf_pred:.2f}` kW")
st.write(f"🔹 CatBoost: `{catboost_pred:.2f}` kW")
st.write(f"🔹 Ensemble: `{ensemble_pred:.2f}` kW")

# Visualization
st.subheader("Comparison Chart")
models = ["SVM", "Random Forest", "CatBoost", "Ensemble"]
predictions = [svm_pred, rf_pred, catboost_pred, ensemble_pred]

fig, ax = plt.subplots()
ax.bar(models, predictions, color=['skyblue', 'lightgreen', 'orange', 'purple'])
ax.set_ylabel("Predicted Energy Output (kW)")
ax.set_ylim(0, max(predictions) + 10)
st.pyplot(fig)
