import streamlit as st
import plotly.graph_objects as go
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, teamgamelog, shotchartdetail, leaguegamelog, leagueleaders
from nba_api.stats.static import players, teams
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px

# Function to get player ID by name
def get_player_id(player_name):
    nba_players = players.get_active_players()
    player_dict = {player['full_name']: player for player in nba_players}
    return player_dict.get(player_name, {}).get('id')

# Function to fetch player stats
@st.cache_data
def get_player_stats(player_id, season='2023-24'):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        gamelog_df = gamelog.get_data_frames()[0]
        if gamelog_df.empty:
            return None, "No games played in the selected season."
        ppg = gamelog_df['PTS'].mean()
        fg_percentage = (gamelog_df['FGM'].sum() / gamelog_df['FGA'].sum()) * 100
        apg = gamelog_df['AST'].mean()
        rpg = gamelog_df['REB'].mean()
        return ppg, fg_percentage, apg, rpg
    except Exception as e:
        return None, str(e)

# Function to fetch and display player headshot
def display_player_headshot(player_id, player_name):
    url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=player_name, use_column_width=False)
    else:
        st.write(f"Could not retrieve headshot for {player_name}.")

# Function to get team stats
@st.cache_data
def get_team_stats(team_id, season='2023-24'):
    try:
        team_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
        team_log_df = team_log.get_data_frames()[0]
        if team_log_df.empty:
            return None, "No games played in the selected season."

        ppg = team_log_df['PTS'].mean()
        fg_percentage = (team_log_df['FGM'].sum() / team_log_df['FGA'].sum()) * 100
        apg = team_log_df['AST'].mean()
        rpg = team_log_df['REB'].mean()
        return ppg, fg_percentage, apg, rpg
    except Exception as e:
        return None, str(e)

# Function to get league averages for the specified season
@st.cache_data
def get_league_averages(season='2023-24'):
    try:
        league_log = leaguegamelog.LeagueGameLog(season=season)
        league_log_df = league_log.get_data_frames()[0]
        if league_log_df.empty:
            return None, "No games played in the selected season."

        ppg = league_log_df['PTS'].mean()
        fg_percentage = (league_log_df['FGM'].sum() / league_log_df['FGA'].sum()) * 100
        apg = league_log_df['AST'].mean()
        rpg = league_log_df['REB'].mean()
        return ppg, fg_percentage, apg, rpg
    except Exception as e:
        return None, str(e)

# Function to get top 5 players in various categories
@st.cache_data
def get_top_players(category, season='2023-24'):
    try:
        leaders = leagueleaders.LeagueLeaders(stat_category_abbreviation=category, season=season, season_type_all_star='Regular Season')
        leaders_df = leaders.get_data_frames()[0].head(5)
        # Ensure that per-game values are displayed correctly
        if category in ["PTS", "AST", "REB", "STL", "BLK"]:
            leaders_df[category] = leaders_df[category] / leaders_df['GP']
        elif category == "FG_PCT":
            leaders_df[category] = leaders_df[category] * 100
        return leaders_df[['PLAYER', 'TEAM', category]]
    except Exception as e:
        return None, str(e)

# Function to plot bar charts for comparison
def plot_stats(player_stats, comparison_stats, player_name, comparison_label, team_name=None):
    categories = ['PTS', 'FG%', 'AST', 'REB']
    player_values = player_stats
    comparison_values = comparison_stats

    fig = go.Figure(data=[
        go.Bar(name=player_name, x=categories, y=player_values),
        go.Bar(name=comparison_label, x=categories, y=comparison_values)
    ])
    fig.update_layout(barmode='group')
    
    # Add title based on comparison type
    if team_name:
        fig.update_layout(title=f'{player_name} and {team_name}')
    else:
        fig.update_layout(title=f'{player_name} vs {comparison_label}')
        
    st.plotly_chart(fig)

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
            fig.add_trace(go.Scatter(x=shot_data['LOC_X'], y=shot_data['LOC_Y'], mode='markers',
                                     marker=dict(color='green' if made else 'red'), name='Made' if made else 'Missed'))

        fig.update_layout(title=f'Shot Chart for {player_name} ({season})', xaxis_title='Court Length', yaxis_title='Court Width')
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Error fetching shot chart data: {e}")

def plot_box_whisker(player_id, player_name, season='2023-24'):
    try:
        # Fetch shot chart data
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star='Regular Season',
            context_measure_simple='FGA'
        )
        shot_df = shotchart.get_data_frames()[0]

        # Filter data to include only successful shots
        made_shots_df = shot_df[shot_df['SHOT_MADE_FLAG'] == 1]

        if made_shots_df.empty:
            st.write(f"No made shots data available for {player_name} in {season}.")
            return

        # Create a box-and-whisker plot for the x and y coordinates of made shots
        fig = px.box(made_shots_df, x='LOC_X', y='LOC_Y', points='all',
                    title=f'Shot Distribution for Made Shots of {player_name} ({season})')
        fig.update_layout(xaxis_title='Court Length', yaxis_title='Court Width')

        # Display the plot
        st.plotly_chart(fig)

    except Exception as e:
        st.write(f"Error fetching shot chart data: {e}")


# Streamlit app
def main():
    st.title("NBA Player Statistics Viewer")

    # Dropdown for player selection
    nba_players = players.get_active_players()
    player_names = [player['full_name'] for player in nba_players]
    selected_player1 = st.selectbox("Select first player:", player_names)
    show_second_player = st.checkbox("Show second player's information")

    if selected_player1:
        player_id1 = get_player_id(selected_player1.strip())
        if player_id1:
            player_info1 = commonplayerinfo.CommonPlayerInfo(player_id=player_id1).get_data_frames()[0]
            if not player_info1.empty:
                player_stats1 = get_player_stats(player_id1)
                league_averages = get_league_averages()

                if len(player_stats1) == 4:
                    ppg1, fg_percentage1, apg1, rpg1 = player_stats1

                    team_id1 = player_info1['TEAM_ID'].values[0]
                    team_name1 = teams.find_team_name_by_id(team_id1)
                    if team_name1:
                        team_name1 = team_name1['full_name']  # Get the full name from the dictionary
                    else:
                        team_name1 = "Unknown Team"

                    team_stats1 = get_team_stats(team_id1)
                    if len(team_stats1) == 4:
                        team_ppg1, team_fg1, team_apg1, team_rpg1 = team_stats1

                        # Display player 1 headshot and comparison with team average
                        display_player_headshot(player_id1, selected_player1)
                        plot_stats([ppg1, fg_percentage1, apg1, rpg1], [team_ppg1, team_fg1, team_apg1, team_rpg1], selected_player1, team_name1)
                        # Display player 1 comparison with league average
                        if league_averages:
                            plot_stats([ppg1, fg_percentage1, apg1, rpg1], list(league_averages), selected_player1, "League Average")
                        else:
                            st.write("Error fetching league averages.")
                        # Display player 1 shot chart
                        plot_shot_chart(player_id1, selected_player1)
                        # Display box-and-whisker plot for player 1
                        plot_box_whisker(player_id1, selected_player1)

                        if show_second_player:
                            selected_player2 = st.selectbox("Select second player:", player_names)
                            if selected_player2:
                                player_id2 = get_player_id(selected_player2.strip())
                                if player_id2:
                                    player_info2 = commonplayerinfo.CommonPlayerInfo(player_id=player_id2).get_data_frames()[0]
                                    if not player_info2.empty:
                                        player_stats2 = get_player_stats(player_id2)
                                        if len(player_stats2) == 4:
                                            ppg2, fg_percentage2, apg2, rpg2 = player_stats2

                                            team_id2 = player_info2['TEAM_ID'].values[0]
                                            team_name2 = teams.find_team_name_by_id(team_id2)
                                            if team_name2:
                                                team_name2 = team_name2['full_name']  # Get the full name from the dictionary
                                            else:
                                                team_name2 = "Unknown Team"

                                            team_stats2 = get_team_stats(team_id2)
                                            if len(team_stats2) == 4:
                                                team_ppg2, team_fg2, team_apg2, team_rpg2 = team_stats2

                                                # Display player 2 headshot and comparison with team average
                                                display_player_headshot(player_id2, selected_player2)
                                                plot_stats([ppg2, fg_percentage2, apg2, rpg2], [team_ppg2, team_fg2, team_apg2, team_rpg2], selected_player2, team_name2)
                                                # Display player 2 comparison with league average
                                                if league_averages:
                                                    plot_stats([ppg2, fg_percentage2, apg2, rpg2], list(league_averages), selected_player2, "League Average")
                                                else:
                                                    st.write("Error fetching league averages.")
                                                # Display player 2 shot chart
                                                plot_shot_chart(player_id2, selected_player2)
                                                # Display box-and-whisker plot for player 2
                                                plot_box_whisker(player_id2, selected_player2)
                                                
                                                # Player vs Player comparison
                                                plot_stats([ppg1, fg_percentage1, apg1, rpg1], [ppg2, fg_percentage2, apg2, rpg2], selected_player1, selected_player2)
                                            else:
                                                st.write(f"Error fetching team stats for {selected_player2}: {team_stats2[1]}")
                                        else:
                                            st.write(f"Error fetching player stats for {selected_player2}: {player_stats2[1]}")
                                    else:
                                        st.write(f"Error fetching player info for {selected_player2}.")
                                else:
                                    st.write(f"Player {selected_player2} not found.")
                            else:
                                st.write("Please select a second player.")
                    else:
                        st.write(f"Error fetching team stats for {selected_player1}: {team_stats1[1]}")
                else:
                    st.write(f"Error fetching player stats for {selected_player1}: {player_stats1[1]}")
            else:
                st.write(f"Error fetching player info for {selected_player1}.")
        else:
            st.write(f"Player {selected_player1} not found.")
    
    # Add section for top 5 season leaders
    st.header("Top 5 Leaders for 2023-24")
    categories = {
        "PTS": "Points Per Game (PPG)",
        "AST": "Assists Per Game (APG)",
        "REB": "Rebounds Per Game (RPG)",
        "STL": "Steals Per Game (SPG)",
        "BLK": "Blocks Per Game (BPG)",
        "FG_PCT": "Field Goal Percentage (FG%)",
        "FG3M": "3-Pointers Made"
    }
    
    for category, label in categories.items():
        st.subheader(label)
        top_players = get_top_players(category)
        if top_players is not None:
            st.table(top_players[['PLAYER', 'TEAM', category]])
        else:
            st.write(f"Error fetching top players for {label}")

if __name__ == "__main__":
    main()
