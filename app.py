import streamlit as st
import plotly.graph_objects as go
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players, teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Function to get player ID by name
def get_player_id(player_name):
    nba_players = players.get_active_players()
    player_dict = {player['full_name']: player for player in nba_players}
    return player_dict.get(player_name, {}).get('id')

# Improved function to fetch player stats with consistent return
@st.cache_data
def get_player_stats(player_id, season='2023-24'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        gamelog_df = gamelog.get_data_frames()[0]
        if gamelog_df.empty:
            return None, "No games played in the selected season."
        return gamelog_df, None  # Return DataFrame and None for error
    except Exception as e:
        return None, str(e)  # Return None and the error message if an exception occurs

# Function to train a Random Forest model
def train_random_forest(data):
    # Prepare data
    data['GameNumber'] = np.arange(len(data)) + 1
    features = ['GameNumber', 'FGM', 'FGA', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    target = 'PTS'
    
    # Split data
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Function to make predictions with the trained model
def predict_next_game(model, data):
    next_game = pd.DataFrame({
        'GameNumber': [data['GameNumber'].max() + 1],
        'FGM': [data['FGM'].mean()],
        'FGA': [data['FGA'].mean()],
        'REB': [data['REB'].mean()],
        'AST': [data['AST'].mean()],
        'STL': [data['STL'].mean()],
        'BLK': [data['BLK'].mean()],
        'TOV': [data['TOV'].mean()]
    })
    predicted_points = model.predict(next_game)
    return predicted_points[0]

# Function to display player headshot
def display_player_headshot(player_id, player_name):
    url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=player_name, use_column_width=False)
    else:
        st.write(f"Could not retrieve headshot for {player_name}.")

# Function to get a player's current team
def get_player_team(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        if not player_info.empty:
            team_id = player_info['TEAM_ID'].values[0]
            team_name = teams.find_team_name_by_id(team_id)
            return team_name['full_name'] if team_name else "Unknown Team"
        return "Unknown Team"
    except Exception as e:
        return "Unknown Team"

# Streamlit app
def main():
    st.title("NBA Player Performance Prediction - 2023-2024 Season")

    # Dropdown for player selection
    nba_players = players.get_active_players()
    player_names = [player['full_name'] for player in nba_players]
    selected_player = st.selectbox("Select a player to predict performance:", player_names)

    if selected_player:
        player_id = get_player_id(selected_player.strip())
        if player_id:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            if not player_info.empty:
                gamelog_df, error = get_player_stats(player_id, season='2023-24')
                if gamelog_df is not None:
                    if len(gamelog_df) >= 10:  # Ensure enough data for training
                        # Get the team for the 2024-2025 season
                        team_name = get_player_team(player_id)
                        st.write(f"**{selected_player} will be playing for {team_name} in the 2024-2025 season.**")

                        # Train the Random Forest model
                        model, mse = train_random_forest(gamelog_df)
                        st.write(f"Model Mean Squared Error: {mse:.2f}")

                        # Predict next game's points
                        predicted_pts = predict_next_game(model, gamelog_df)
                        st.write(f"Predicted Points for Next Game: {predicted_pts:.2f}")

                        # Display player headshot
                        display_player_headshot(player_id, selected_player)

                        # Visualization of recent game data with labeled axes
                        st.write("### Recent Game Performance")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=gamelog_df['GameNumber'], y=gamelog_df['PTS'], mode='lines+markers', name='Points'))
                        fig.update_layout(
                            title='Recent Game Performance',
                            xaxis_title='Game Number',
                            yaxis_title='Points Scored',
                            showlegend=True
                        )
                        st.plotly_chart(fig)

                        # Additional visuals: Field Goal Percentage over time
                        st.write("### Field Goal Percentage Over Time")
                        fig_fg = go.Figure()
                        gamelog_df['FG%'] = (gamelog_df['FGM'] / gamelog_df['FGA']) * 100
                        fig_fg.add_trace(go.Scatter(x=gamelog_df['GameNumber'], y=gamelog_df['FG%'], mode='lines+markers', name='FG%'))
                        fig_fg.update_layout(
                            title='Field Goal Percentage Over Time',
                            xaxis_title='Game Number',
                            yaxis_title='Field Goal Percentage (%)',
                            showlegend=True
                        )
                        st.plotly_chart(fig_fg)

                        # Visualization: Comparison of Points, Assists, and Rebounds
                        st.write("### Points, Assists, and Rebounds Comparison")
                        comparison_fig = go.Figure()
                        comparison_fig.add_trace(go.Bar(x=['Points', 'Assists', 'Rebounds'], y=[
                            gamelog_df['PTS'].mean(),
                            gamelog_df['AST'].mean(),
                            gamelog_df['REB'].mean()
                        ], name='Average Stats', marker_color='blue'))
                        comparison_fig.update_layout(
                            title='Average Points, Assists, and Rebounds',
                            yaxis_title='Average per Game',
                            xaxis_title='Stat Category'
                        )
                        st.plotly_chart(comparison_fig)

                    else:
                        st.write("Not enough game data to train a predictive model.")
                else:
                    st.write(f"Error fetching player stats: {error}")
            else:
                st.write(f"Error fetching player info for {selected_player}.")
        else:
            st.write(f"Player {selected_player} not found.")

if __name__ == "__main__":
    main()
