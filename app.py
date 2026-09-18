
import streamlit as st
import pandas as pd
import joblib

# Load the trained Polynomial Regression pipeline
model = joblib.load("Fan.pkl")

st.title("Electric Bill Prediction")


ac_units = st.number_input(
    "AC Electricity Consumption (Units)",
    min_value=0.0,
    max_value=150.0,
    value=10.0
)

fan_units = st.number_input(
    "Fan Electricity Consumption (Units)",
    min_value=0.0,
    max_value=150.0,
    value=10.0
)

if st.button("Predict Electric Bill"):

    input_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Electric Bill: ₹{prediction[0]:.2f}"
    )
