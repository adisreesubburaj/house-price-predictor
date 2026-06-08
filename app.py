
import streamlit as st
import numpy as np
import joblib

model = joblib.load('house_price_model.pkl')

st.title("🏠 House Price Predictor")
st.write("Fill in the details below to predict the house price!")

area      = st.number_input("Area (sqft)",        min_value=500,  max_value=5000, value=1500)
bedrooms  = st.number_input("Bedrooms",            min_value=1,    max_value=6,    value=3)
bathrooms = st.number_input("Bathrooms",           min_value=1,    max_value=4,    value=2)
age       = st.number_input("Age of House (yrs)",  min_value=1,    max_value=50,   value=10)
garage    = st.selectbox("Garage",   ["Yes", "No"])
location  = st.selectbox("Location", ["Urban", "Suburb", "Rural"])

garage_enc   = 1 if garage == "Yes" else 0
location_map = {"Urban": 2, "Suburb": 1, "Rural": 0}
location_enc = location_map[location]

if st.button("🔮 Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, age, garage_enc, location_enc]])
    price    = model.predict(features)[0]
    st.success(f"💰 Predicted House Price: ${price:,.0f}")
    st.balloons()
