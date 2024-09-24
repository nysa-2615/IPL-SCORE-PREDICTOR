import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

# Load the pre-trained model, scaler, and encoders
model = load_model('ipl_score_model.h5')
scaler = load('scaler.joblib')

venue_encoder = load('venue_encoder.joblib')
batting_team_encoder = load('batting_team_encoder.joblib')
bowling_team_encoder = load('bowling_team_encoder.joblib')
striker_encoder = load('striker_encoder.joblib')
bowler_encoder = load('bowler_encoder.joblib')

# Load the dataset (for displaying options in dropdowns)
df = pd.read_csv('ipl_data.csv')

# Streamlit UI components
st.title("IPL Score Prediction")

venue = st.selectbox("Select Venue:", df['venue'].unique())
batting_team = st.selectbox("Select Batting Team:", df['bat_team'].unique())
bowling_team = st.selectbox("Select Bowling Team:", df['bowl_team'].unique())
striker = st.selectbox("Select Striker:", df['batsman'].unique())
bowler = st.selectbox("Select Bowler:", df['bowler'].unique())

# Prediction button
if st.button("Predict Score"):
    # Encode the input values using encoders
    encoded_venue = venue_encoder.transform([venue])[0]
    encoded_batting_team = batting_team_encoder.transform([batting_team])[0]
    encoded_bowling_team = bowling_team_encoder.transform([bowling_team])[0]
    encoded_striker = striker_encoder.transform([striker])[0]
    encoded_bowler = bowler_encoder.transform([bowler])[0]

    # Prepare the input for the model
    input_data = np.array([encoded_venue, encoded_batting_team, encoded_bowling_team, encoded_striker, encoded_bowler])
    input_data = input_data.reshape(1, -1)  # Reshape to 2D
    input_data = scaler.transform(input_data)  # Apply scaling

    # Make the prediction
    predicted_score = model.predict(input_data)
    predicted_score = int(predicted_score[0, 0])  # Get the score as an integer

    # Display the predicted score
    st.success(f"The predicted score is: {predicted_score}")
