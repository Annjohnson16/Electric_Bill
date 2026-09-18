import streamlit as st
import joblib

# Load model and polynomial transformer
model = joblib.load("model.pkl")
poly = joblib.load("poly.pkl")

st.title("Electric Bill Predictor")

ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

if st.button("Predict"):

    # Transform input using the same polynomial transformation
    ac_units_poly = poly.transform([[ac_units]])

    # Predict electric bill
    prediction = model.predict(ac_units_poly)

    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")
