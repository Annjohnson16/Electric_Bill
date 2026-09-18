import streamlit as st
import joblib
from sklearn.preprocessing import PolynomialFeatures

model = joblib.load("model(1).pkl")

poly = PolynomialFeatures(degree=2)

st.title("Electric Bill Predictor")

ac_units = st.number_input(
    "Enter AC Units(Between 1 nad 150)",
    min_value=1.0,
    max_value=149.0,
    step=1.0
)

if st.button("Predict"):

    data = [[ac_units]]

    data_poly = poly.fit_transform(data)

    prediction = model.predict(data_poly)

    st.success(f"Expected Electric Bill: ₹{prediction[0]:.2f}")
