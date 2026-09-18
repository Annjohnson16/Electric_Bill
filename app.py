
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Load trained Linear Regression model
model = joblib.load("Fan.pkl")

# Create PolynomialFeatures with the same degree used during training
poly = PolynomialFeatures(degree=2)

st.title("Electric Bill Prediction")


# AC Units input
ac_units = st.number_input(
    "AC Electricity Consumption (Units)",
    min_value=0.0,
    max_value=150.0,
    value=10.0
)

# Fan Units input
fan_units = st.number_input(
    "Fan Electricity Consumption (Units)",
    min_value=0.0,
    max_value=150.0,
    value=10.0
)

if st.button("Predict Electric Bill"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })

    # Transform input into polynomial features
    input_poly = poly.fit_transform(input_data)

    # Predict using trained Linear Regression model
    prediction = model.predict(input_poly)

    st.success(
        f"Predicted Electric Bill: ₹{prediction[0]:.2f}"
    )
