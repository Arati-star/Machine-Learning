import streamlit as st
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter House Details")

overall_qual = st.slider(
    "Overall Quality",
    1,
    10,
    5
)

gr_liv_area = st.number_input(
    "Living Area (sq ft)",
    min_value=500,
    max_value=5000,
    value=1500
)

garage_cars = st.slider(
    "Garage Capacity",
    0,
    4,
    2
)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars]
    })

    prediction = model.predict(input_df)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )