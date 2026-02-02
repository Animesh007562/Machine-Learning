# Save the Streamlit app code to app.py
# with Dropdown meanu use
import streamlit as st
import pandas as pd
import pickle 

# ----------------------
# Load Model & Dropdown Data
# ----------------------
pipe = pickle.load(open("model.pkl", "rb"))
locations = pickle.load(open("locations.pkl", "rb"))
area_types = pickle.load(open("area_type.pkl", "rb"))

# ----------------------

# Streamlit App
# ----------------------
st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction App")
st.write("Enter property details below to get an estimated price (in Lakhs).")

# User inputs
location = st.selectbox("Select Location", locations)
area_type = st.selectbox("Select Area Type", area_types)
total_sqft = st.number_input("Enter Total Square Feet", min_value=100.0, step=10.0)
bath = st.number_input("Enter Number of Bathrooms", min_value=1.0, step=1.0)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1, step=1)
balcony = st.number_input("Enter number of Balcony: ", min_value=0)


# Prediction button
if st.button("Predict Price"):
    # Create DataFrame from inputs
    user_input = pd.DataFrame([[location, area_type, total_sqft, bath, bhk,balcony]],
                          columns=['location', 'area_type', 'total_sqft', 'bath', 'bhk','balcony'])

    # Make prediction
    predicted_price = pipe.predict(user_input)[0]

    st.success(f"💰 Estimated Price: **{predicted_price:.2f} Lakhs**")

