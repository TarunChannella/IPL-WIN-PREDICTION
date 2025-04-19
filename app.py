import streamlit as st
import pickle
import pandas as pd

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals','Locknow super gaints','Gujarat titans']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the machine learning model (assuming it's already trained and saved in 'ml project' folder)
pipe = pickle.load(open(r'C:\Users\tarun\OneDrive\Desktop\ml project\pipe.pkl', 'rb'))

# Streamlit app setup
st.title('IPL Win Predictor')

# Layout for selecting teams and cities
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select host city and input for target score
selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=0)

# Layout for score, overs, and wickets
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, format="%.1f")
with col5:
    wickets = st.number_input('Wickets out', min_value=0)

# Button to predict the probability
if st.button('Predict Probability'):
    # Calculate remaining runs, balls, wickets, current run rate (CRR) and required run rate (RRR)
    runs_left = target - score
    balls_left = 120 - (overs * 6)  # 120 balls in a T20 match
    wickets_left = 10 - wickets
    crr = score / overs if overs != 0 else 0  # Avoid division by zero
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0  # Avoid division by zero

    # Prepare input data for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make prediction using the loaded model pipeline
    result = pipe.predict_proba(input_df)

    # Extracting win and loss probabilities
    loss = result[0][0]
    win = result[0][1]

    # Displaying the results
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
