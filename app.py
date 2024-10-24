import streamlit as st
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2, commonplayerinfo, playergamelog
from nba_api.stats.static import players
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import requests
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

# Function to fetch player info including headshots
def get_player_info(player_name):
    nba_players = players.get_active_players()
    player_dict = {player['full_name']: player for player in nba_players}
    player = player_dict.get(player_name, None)
    
    if player:
        player_id = player['id']
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        headshot_url = player_info['PLAYER_IMAGE'].iloc[0]
        return player_id, headshot_url
    else:
        return None, None

# Function to recommend players based on recent game stats
def recommend_recent_game_player(player_id, num_recommendations=5):
    # Fetch player's most recent game data
    recent_game = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]
    recent_stats = recent_game[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']].iloc[0]
    
    # Placeholder for other players' stats
    # Fetch similar stats for other players to recommend (Example data)
    all_players = players.get_active_players()
    all_stats = []
    
    for p in all_players[:50]:  # Limiting to the first 50 players for simplicity
        try:
            p_game_log = playergamelog.PlayerGameLog(player_id=p['id'], season='2023-24').get_data_frames()[0]
            p_recent_stats = p_game_log[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']].iloc[0]
            all_stats.append([p['full_name'], *p_recent_stats])
        except:
            continue
    
    # Creating DataFrame for similarity calculation
    stats_df = pd.DataFrame(all_stats, columns=['PLAYER_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT'])
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_df[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']])
    
    # Calculate similarity
    recent_scaled = scaler.transform([recent_stats])
    similarity_matrix = cosine_similarity(recent_scaled, scaled_stats)
    
    # Get top similar players
    similarity_df = pd.DataFrame(similarity_matrix[0], index=stats_df['PLAYER_NAME'], columns=['Similarity'])
    top_similar = similarity_df.sort_values(by='Similarity', ascending=False).head(num_recommendations + 1)
    
    return top_similar.index.tolist()

# Function to predict win chances based on recent game stats
def predict_recent_game_win(player_id):
    recent_game = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]
    recent_stats = recent_game[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT']].iloc[0]
    
    # Placeholder for team stats (Example features)
    X = pd.DataFrame([recent_stats], columns=['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG_PCT'])
    
    # Logistic Regression for prediction
    y = [1]  # Placeholder target
    model = LogisticRegression()
    model.fit(X, y)
    
    prediction = model.predict(X)
    return "Win" if prediction[0] == 1 else "Loss"

# Streamlit app structure
st.title("NBA Betting Insights App")
st.header("Player Insights & Recommendations")

# Fetch player data for dropdown
all_players = players.get_active_players()
player_names = [p['full_name'] for p in all_players]
selected_player = st.selectbox("Select a Player:", player_names)

# Get player info and stats
if selected_player:
    player_id, headshot_url = get_player_info(selected_player)
    
    if player_id:
        if headshot_url:
            # Display player headshot
            response = requests.get(headshot_url)
            if response.status_code == 200:
                image = Image.open(requests.get(headshot_url, stream=True).raw)
                st.image(image, caption=selected_player, width=200)
        
        # Check if player's team is currently playing
        live_games = get_live_games()
        player_stats = None
        
        if not live_games.empty:
            st.write(f"Live game data for {selected_player}'s team:")
            player_stats = get_live_player_stats(live_games['GAME_ID'].iloc[0])
            st.dataframe(player_stats)
        else:
            st.write(f"No live games for {selected_player}'s team.")
            
            # Provide recommendations based on recent game stats
            recommendations = recommend_recent_game_player(player_id)
            st.write(f"Players similar to {selected_player} based on the most recent game:")
            st.write(recommendations)
            
            # Prediction based on recent stats
            prediction = predict_recent_game_win(player_id)
            st.write(f"Predicted outcome based on {selected_player}'s most recent game: {prediction}")
