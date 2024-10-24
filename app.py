import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, shotchartdetail, leagueleaders
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

# Function to plot shot chart
def plot_shot_chart(player_id, player_name, season='2023-24'):
    try:
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0, 
            player_id=player_id, 
            season_nullable=season, 
            season_type_all_star='Regular Season', 
            context_measure_simple='FGA'
        )
        shot_df = shotchart.get_data_frames()[0]

        fig = go.Figure()
        for made in [1, 0]:
            shot_data = shot_df[shot_df['SHOT_MADE_FLAG'] == made]
            fig.add_trace(go.Scatter(
                x=shot_data['LOC_X'], 
                y=shot_data['LOC_Y'], 
                mode='markers',
                marker=dict(color='green' if made else 'red'),
                name='Made' if made else 'Missed'
            ))

        fig.update_layout(
            title=f'Shot Chart for {player_name} ({season})',
            xaxis_title='Court Length',
            yaxis_title='Court Width'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Error fetching shot chart data: {e}")

# Function to display a spider chart of key stats
def plot_spider_chart(stats, player_name):
    categories = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers']
    values = [
        stats['PTS'].mean(),
        stats['AST'].mean(),
        stats['REB'].mean(),
        stats['STL'].mean(),
        stats['BLK'].mean(),
        stats['TOV'].mean()
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + values[:1],  # Close the loop
        theta=categories + categories[:1],
        fill='toself',
        name=player_name
    ))
    fig.update_layout(
        title=f"Spider Chart of Key Stats for {player_name}",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    st.plotly_chart(fig)

# Function to get top league players
def get_top_league_players(category, season='2023-24'):
    try:
        leaders = leagueleaders.LeagueLeaders(
            stat_category_abbreviation=category, 
            season=season, 
            season_type_all_star='Regular Season'
        )
        leaders_df = leaders.get_data_frames()[0].head(10)  # Top 10 players
        return leaders_df[['PLAYER', 'TEAM', category]]
    except Exception as e:
        return None

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

                        # Plot the shot chart
                        st.write("### Shot Chart")
                        plot_shot_chart(player_id, selected_player, season='2023-24')

                        # Spider chart of key stats
                        st.write("### Key Stats Spider Chart")
                        plot_spider_chart(gamelog_df, selected_player)

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

                        # Top league players in various categories
                        st.write("### League Player Rankings")
                        categories = {
                            "PTS": "Points Per Game (PPG)",
                            "AST": "Assists Per Game (APG)",
                            "REB": "Rebounds Per Game (RPG)"
                        }
                        for category, label in categories.items():
                            st.subheader(label)
                            top_players = get_top_league_players(category)
                            if top_players is not None:
                                st.table(top_players)
                            else:
                                st.write(f"Error fetching top players for {label}")
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
