import streamlit as st
import pandas as pd
import numpy as np
from data_manager import DataManager

def main():
    st.set_page_config(page_title="Character Match Statistics", layout="wide")

    st.title("Character Match Statistics Tracker") 

    # Initialize data manager
    dm = DataManager("C:/INeedThis/UnmatchedDataSync/matches.csv")

    # Sidebar for adding new matches
    st.sidebar.header("Add New Match")

    # Match type selection
    match_type = st.sidebar.radio("Select Match Type", ["1v1", "2v2"])

    # Team 1 selection
    st.sidebar.subheader("Team 1")
    team1 = []
    p1 = []
    team1.append(st.sidebar.selectbox("Персонаж 1", dm.characters, key="t1p1"))
    p1.append(st.sidebar.selectbox("Игрок 1", dm.players, key="p1p1"))
    if match_type == "2v2":
        team1.append(st.sidebar.selectbox("Персонаж 2", dm.characters, key="t1p2"))
        p1.append(st.sidebar.selectbox("Игрок 2", dm.players, key="p1p2"))
        

    # Team 2 selection
    st.sidebar.subheader("Team 2")
    team2 = []
    p2 = []
    team2.append(st.sidebar.selectbox("Персонаж 1", dm.characters, key="t2p1"))
    p2.append(st.sidebar.selectbox("Игрок 1", dm.players, key="p2p1"))
    if match_type == "2v2":
        team2.append(st.sidebar.selectbox("Персонаж 2", dm.characters, key="t2p2"))
        p2.append(st.sidebar.selectbox("Игрок 2", dm.players, key="p2p2"))
        
    st.sidebar.subheader("Score")
    # Score input
    score1 = st.sidebar.number_input("Team 1 Score", min_value=0, value=0, key="score1")
    score2 = st.sidebar.number_input("Team 2 Score", min_value=0, value=0, key="score2")

    # Submit button
    if st.sidebar.button("Add Match Result"):
        if dm.validate_teams(team1, team2):
            if score1 == score2:
                st.sidebar.error("Scores cannot be equal!")
            else:
                success = dm.add_match(team1, team2, score1, score2, p1, p2)
                if success:
                    st.sidebar.success("Match result added successfully!")
                else:
                    st.sidebar.error("Failed to add match result!")
        else:
            st.sidebar.error("Invalid team composition!")

    # Main content area - using tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Character Statistics", 
                                            "Win Rate Trends", "Winrate Matrix", "Match History", 
                                            "Matchup Matrix", "Player Stats"])

    # Add match type filter in the main area
    stats_match_type = st.radio(
        "Filter statistics by match type:",
        ["All Matches", "1v1 Only", "2v2 Only"],
        horizontal=True
    )

    # Convert selection to filter value
    filter_type = {"All Matches": "all", "1v1 Only": "1v1", "2v2 Only": "2v2"}[stats_match_type]

    with tab1:
        st.header("Character Statistics")
        stats = dm.get_character_stats(match_type=filter_type)

        # Create a DataFrame for better display
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        # Calculate total games
        stats_df['total_games'] = stats_df['wins'] + stats_df['losses']
        stats_df['win_rate'] = (stats_df['wins'] / stats_df['total_games'] * 100).round(2)
        stats_df['win_rate'] = stats_df['win_rate'].fillna(0)
        stats_df = stats_df.sort_values('win_rate', ascending=False)

        # Rename columns for better readability
        stats_df.columns = ['Wins', 'Losses', 'Total Games', 'Win Rate (%)']

        # Display statistics with styling and full width
        st.dataframe(
            stats_df.style.format({
                'Win Rate (%)': '{:.2f}%',
                'Total Games': '{:,.0f}',  # Format as whole number
                'Wins': '{:,.0f}',
                'Losses': '{:,.0f}'
            }).set_properties(**{
                'text-align': 'center',
                'width': '100%'
            }),
            width='stretch',
            height=600
        )

    with tab2:
        st.header("Win Rate Trends")
        win_rates = dm.get_win_rates_over_time(match_type=filter_type)

        if win_rates:
            # Character selection for the graph
            selected_chars = st.multiselect(
                "Select characters to display",
                dm.characters,
                default=dm.characters[:3]  # Default to first 3 characters
            )

            if selected_chars:
                # Create a line chart
                chart_data = pd.DataFrame()

                for char in selected_chars:
                    if win_rates[char]['timestamps']:  # Only add if character has matches
                        df = pd.DataFrame({
                            'timestamp': win_rates[char]['timestamps'],
                            char: win_rates[char]['win_rates']
                        }).set_index('timestamp')

                        if chart_data.empty:
                            chart_data = df
                        else:
                            chart_data = chart_data.join(df, how='outer')

                if not chart_data.empty:
                    st.line_chart(chart_data)
                    st.info("💡 The graph shows how each character's win rate has changed over time. " 
                           "Select different characters to compare their performance trends.")
            else:
                st.info("Please select at least one character to display their win rate trend.")
        else:
            st.info("No match data available yet!")

    with tab3:
        st.header("Character Winrate Matrix")
        matchup_matrix = dm.get_matchup_matrix(match_type=filter_type)

        # Convert to percentage
        # matchup_matrix = matchup_matrix * 100

        # Display the heatmap
        st.write("Win rates (%) for row character vs column character")

        # Format the matrix for display
        formatted_matrix = matchup_matrix.copy()
        # formatted_matrix = formatted_matrix.round(1)

        # Create a color-coded display using custom HTML
        def color_scale(val):
            # Create a color scale from red (0%) to green (100%)
            if val == '':
                return 'background-color: #101414'
            val = float(val.split('%')[0])
            normalized = val / 100
            red = int(255 * (1 - normalized))
            green = int(255 * normalized)
            return f'background-color: rgb({red}, {green}, 0); color: white'

        st.dataframe(
            formatted_matrix.style.map(color_scale)
            .set_properties(**{
                'text-align': 'center',
                'width': '100%'
            }),
            width='stretch',
            height=800
        )

        st.info("💡 Reading the matrix: Each cell shows the win rate (%) of the row character against the column character. " 
                "Green indicates a favorable matchup, red indicates an unfavorable one.")

    with tab4:
        st.header("Match History")
        history = dm.get_match_history(match_type=filter_type)
        if not history.empty:
            history['result'] = history.apply(
                lambda x: f"{x['team1']} vs {x['team2']} ({x['score1']}-{x['score2']})",
                axis=1
            )
            history['players_team1'] = history.apply(
                lambda x: f"{str(str(x['p1']).split('!')[0])} и {str(str(x['p1']).split('!')[-1])}",
                axis=1
            )
            history['players_team2'] = history.apply(
                lambda x: f"{str(str(x['p2']).split('!')[0])} и {str(str(x['p2']).split('!')[-1])}",
                axis=1
            )
            st.dataframe(
                history[['result', 'players_team1', 'players_team2', 'timestamp']].sort_values('timestamp', ascending=False),
                width='stretch'
            )
        else:
            st.info("No matches recorded yet!")

    with tab5:
        st.header("Character Matchup Matrix")
        matchup_matrix = dm.get_score_matrix(match_type=filter_type)

        # Display the heatmap
        st.write("Score rate (%) for row character vs column character")

        # Format the matrix for display
        formatted_matrix = matchup_matrix.copy()
        # formatted_matrix = formatted_matrix.round(1)

        # Create a color-coded display using custom HTML
        def color_scale(val):
            # Create a color scale from red (0%) to green (100%)
            if val == '':
                return 'background-color: #101414'
            val = float(val.split('%')[0])
            normalized = val / 100
            red = int(255 * (1 - normalized))
            green = int(255 * normalized)
            return f'background-color: rgb({red}, {green}, 0); color: white'

        st.dataframe(
            formatted_matrix.style.map(color_scale)
            .set_properties(**{
                'text-align': 'center',
                'width': '100%'
            }),
            width='stretch',
            height=800
        )

        st.info("💡 Reading the matrix: Each cell shows the score rate (%) of the row character against the column character. " 
                "Green indicates a favorable matchup, red indicates an unfavorable one.")
        
    with tab6:
        tab6_1, tab6_2, tab6_3 = st.tabs(["Players winrate", "Players winrate by character", "Players winrate matrix"])
        with tab6_1:
            st.header("Players winrate")
            players_winrate = dm.get_players_winrate(match_type=filter_type)
            players_winrate.columns = ['Total games', 'Wins', 'Win Rate (%)']
            # Display statistics with styling and full width
            st.dataframe(
                players_winrate.style.format({
                    'Win Rate (%)': '{:.2f}%',
                    'Total Games': '{:,.0f}',
                    'Wins': '{:,.0f}'
                }).set_properties(**{
                    'text-align': 'center',
                    'width': '100%'
                }),
                width='stretch',
                height=600
            )
        with tab6_2:
            st.header("Players winrate by character")
            pl = (st.selectbox("Игрок", dm.players, key="p"))
            players_winrate_by_character = dm.get_players_winrate_by_character(player=pl, match_type=filter_type)
            # Display statistics with styling and full width
            players_winrate_by_character.columns = ['Total games', 'Wins', 'Win Rate (%)']
            st.dataframe(
                players_winrate_by_character.style.format({
                    'Win Rate (%)': '{:.2f}%',
                    'Total Games': '{:,.0f}',
                    'Wins': '{:,.0f}'
                }).set_properties(**{
                    'text-align': 'center',
                    'width': '100%'
                }),
                width='stretch',
                height=600
            )
        with tab6_3:
            st.header("Players winrate matrix")
            matchup_matrix = dm.get_players_score_matrix(match_type=filter_type)
            # Format the matrix for display
            formatted_matrix = matchup_matrix.copy()
            # formatted_matrix = formatted_matrix.round(1)

            # Create a color-coded display using custom HTML
            def color_scale(val):
                # Create a color scale from red (0%) to green (100%)
                if val == '-':
                    return 'background-color: #101414'
                val = float(val.split('%')[0])
                normalized = val / 100
                red = int(255 * (1 - normalized))
                green = int(255 * normalized)
                return f'background-color: rgb({red}, {green}, 0); color: white'

            st.dataframe(
                formatted_matrix.style.map(color_scale)
                .set_properties(**{
                    'text-align': 'center',
                    'width': '100%'
                }),
                width='stretch',
                height=800
            )



if __name__ == "__main__":
    main()