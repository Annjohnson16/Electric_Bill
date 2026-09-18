import streamlit as st
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("Electric Bill Predictor")

ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

if st.button("Predict"):

    # Prediction
    prediction = model.predict([[ac_units]])

    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")
