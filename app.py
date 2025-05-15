import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

model_url = 'https://www.dropbox.com/scl/fi/0rmj62y9z3767ou6aubn4/random_forest_model-1.pkl?rlkey=1una1on12h7hv6iegxt1rufhi&st=hndqo42n&dl=1'

if not os.path.exists('random_forest_model.pkl'):
    urllib.request.urlretrieve(model_url, 'random_forest_model.pkl')

model = joblib.load('random_forest_model.pkl')

st.title("California House Price Prediction")

# Input fields for the model features
MedInc = st.number_input('Median Income (MedInc)', 0.0, 20.0, 3.5)
HouseAge = st.number_input('House Age (HouseAge)', 1, 100, 30)
AveRooms = st.number_input('Average Rooms (AveRooms)', 0.0, 20.0, 5.0)
AveBedrms = st.number_input('Average Bedrooms (AveBedrms)', 0.0, 10.0, 1.0)
Population = st.number_input('Population', 0, 50000, 1000)
AveOccup = st.number_input('Average Occupancy (AveOccup)', 0.0, 10.0, 3.0)
Latitude = st.number_input('Latitude', 30.0, 42.0, 34.0)
Longitude = st.number_input('Longitude', -125.0, -114.0, -118.0)

# Prepare data for prediction
input_data = {
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude
}

# Reorder input_df columns exactly as model expects
input_df = pd.DataFrame([input_data])
input_df = input_df[model.feature_names_in_]

# Button to make prediction
if st.button('Predict'):
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Median House Value: ${pred * 100000:.2f}")
