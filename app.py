import streamlit as st
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Function to get live games using today's date
def get_live_games():
    today = datetime.now().strftime('%Y-%m-%d')
    scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    
    # Filter for games that are currently live
    live_games = games[games['GAME_STATUS_TEXT'].str.contains('LIVE', case=False)]
    return live_games

# Function to get player stats for a live game
def get_live_player_stats(game_id):
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    player_stats = boxscore.get_data_frames()[0]
    return player_stats

# Function to recommend players based on current game stats
def recommend_live_players(game_id, player_name, num_recommendations=5):
    player_stats = get_live_player_stats(game_id)
    
    # Preprocess player data for similarity
    stats_df = player_stats[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']]
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_df[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']])
    
    # Calculate similarity between players
    similarity_matrix = cosine_similarity(scaled_stats)
    similarity_df = pd.DataFrame(similarity_matrix, index=stats_df['PLAYER_NAME'], columns=stats_df['PLAYER_NAME'])
    
    if player_name not in similarity_df.index:
        return "Player not found in live game data."
    
    # Get top similar players
    similar_players = similarity_df[player_name].sort_values(ascending=False)[1:num_recommendations+1]
    return similar_players.index.tolist()

# Predict win chances based on live stats
def predict_team_win(game_id):
    player_stats = get_live_player_stats(game_id)
    team_stats = player_stats.groupby('TEAM_ABBREVIATION').sum()
    
    # Add more features: Shooting Efficiency, Turnovers, Rebounds
    team_stats['POINT_DIFFERENCE'] = team_stats['PTS'].diff().iloc[-1]
    features = ['FG_PCT', 'REB', 'AST', 'STL', 'BLK', 'TO', 'POINT_DIFFERENCE']
    
    # Prepare data
    X = team_stats[features].fillna(0)
    y = (X['POINT_DIFFERENCE'] > 0).astype(int)  # Simple win/loss based on point difference
    
    # Logistic Regression for prediction
    model = LogisticRegression()
    model.fit(X, y)
    prediction = model.predict(X)
    
    predicted_winner = team_stats.index[prediction.argmax()]
    return predicted_winner, X

# Streamlit app structure
st.title("NBA Betting Insights App")
st.header("Live NBA Games Overview")

# Get live games
live_games = get_live_games()

# Display live games in a table
if not live_games.empty:
    st.write("Current Live Games:")
    st.dataframe(live_games[['GAME_ID', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'LIVE_PERIOD', 'LIVE_PC_TIME']])
else:
    st.write("No live games are available right now.")

# User interaction to select game for recommendations
if not live_games.empty:
    selected_game_id = st.selectbox("Select a Game ID for Player Insights", live_games['GAME_ID'])
    selected_player = st.text_input("Enter a player's name from the selected game:")
    
    if selected_player:
        recommendations = recommend_live_players(selected_game_id, selected_player)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write(f"Players similar to {selected_player}:")
            st.write(recommendations)
            
            # Visualization: Display player performance using radar chart
            player_stats = get_live_player_stats(selected_game_id)
            radar_data = player_stats[player_stats['PLAYER_NAME'].isin([selected_player] + recommendations)]
            radar_data = radar_data[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']]
            
            fig = px.line_polar(radar_data.melt(id_vars='PLAYER_NAME'), r='value', theta='variable', 
                                color='PLAYER_NAME', line_close=True)
            fig.update_layout(title="Player Performance Comparison (Radar Chart)")
            st.plotly_chart(fig)

# Prediction section for betting insights
if not live_games.empty:
    selected_game_id = st.selectbox("Select a Game ID for Prediction", live_games['GAME_ID'], key='prediction')
    
    if selected_game_id:
        predicted_winner, feature_data = predict_team_win(selected_game_id)
        st.write(f"Predicted Winner: {predicted_winner}")
        
        # Visualization: Show feature importance for prediction
        fig = px.bar(feature_data.T, title="Key Features Influencing Prediction", labels={'index': 'Features', 'value': 'Importance'})
        st.plotly_chart(fig)

# Filter games based on user preferences
st.sidebar.title("Game Filters")
team_filter = st.sidebar.text_input("Filter games by team name (e.g., Lakers, Warriors):")

if team_filter:
    filtered_games = live_games[
        live_games['HOME_TEAM_NAME'].str.contains(team_filter, case=False) | 
        live_games['VISITOR_TEAM_NAME'].str.contains(team_filter, case=False)
    ]
    
    if not filtered_games.empty:
        st.write(f"Filtered Games for '{team_filter}':")
        st.dataframe(filtered_games[['GAME_ID', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 'LIVE_PERIOD', 'LIVE_PC_TIME']])
    else:
        st.write(f"No live games found for '{team_filter}'.")

st.sidebar.write("Use the filters above to find specific teams or players!")
