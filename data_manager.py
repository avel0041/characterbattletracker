import pandas as pd
import os
import numpy as np
from typing import List, Tuple, Dict

class DataManager:
    def __init__(self, csv_path: str = "matches.csv"):
        self.csv_path = csv_path
        self.characters = [
            "Сунь-укун", "Йененга", "Кровавая Мэри", "Ахиллес", 
            "Ода Набунага", "Томоэ", "Джин", "Гудини", 
            "Три сестры", "Гамлет", "Шекспир", "Титания"
        ]
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['team1', 'team2', 'score1', 'score2', 'timestamp'])
            df.to_csv(self.csv_path, index=False)

    def add_match(self, team1: List[str], team2: List[str], score1: int, score2: int) -> bool:
        """Add a new match result to the CSV file."""
        try:
            new_match = pd.DataFrame([{
                'team1': '+'.join(team1),
                'team2': '+'.join(team2),
                'score1': score1,
                'score2': score2,
                'timestamp': pd.Timestamp.now()
            }])

            new_match.to_csv(self.csv_path, mode='a', header=False, index=False)
            return True
        except Exception:
            return False

    def _get_match_type(self, team1: str, team2: str) -> str:
        """Determine if a match is 1v1 or 2v2 based on team composition."""
        return "2v2" if "+" in team1 or "+" in team2 else "1v1"

    def get_character_stats(self, match_type: str = "all") -> Dict[str, Dict[str, int]]:
        """Calculate statistics for each character with optional match type filter."""
        df = pd.read_csv(self.csv_path)
        stats = {char: {'wins': 0, 'losses': 0} for char in self.characters}

        for _, row in df.iterrows():
            current_match_type = self._get_match_type(row['team1'], row['team2'])

            # Skip if match type doesn't match filter
            if match_type != "all" and current_match_type != match_type:
                continue

            team1 = row['team1'].split('+')
            team2 = row['team2'].split('+')

            # Determine winners and losers
            if row['score1'] > row['score2']:
                winners = team1
                losers = team2
            else:
                winners = team2
                losers = team1

            # Update statistics
            for char in winners:
                stats[char]['wins'] += 1
            for char in losers:
                stats[char]['losses'] += 1

        return stats

    def get_win_rates_over_time(self, match_type: str = "all") -> pd.DataFrame:
        """Calculate win rates for each character over time with optional match type filter."""
        df = pd.read_csv(self.csv_path)
        if df.empty:
            return pd.DataFrame()

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Initialize results dictionary
        results = {char: {'timestamps': [], 'win_rates': []} for char in self.characters}

        # Process matches chronologically
        df = df.sort_values('timestamp')

        for char in self.characters:
            wins = 0
            matches = 0
            win_rates = []
            timestamps = []

            for _, row in df.iterrows():
                current_match_type = self._get_match_type(row['team1'], row['team2'])

                # Skip if match type doesn't match filter
                if match_type != "all" and current_match_type != match_type:
                    continue

                team1 = row['team1'].split('+')
                team2 = row['team2'].split('+')

                # Check if character participated in the match
                if char in team1 or char in team2:
                    matches += 1
                    # Check if character won
                    if (char in team1 and row['score1'] > row['score2']) or \
                       (char in team2 and row['score2'] > row['score1']):
                        wins += 1

                    win_rate = (wins / matches) * 100 if matches > 0 else 0
                    win_rates.append(win_rate)
                    timestamps.append(row['timestamp'])

            if timestamps:  # Only add if character has matches
                results[char]['timestamps'] = timestamps
                results[char]['win_rates'] = win_rates

        return results

    def get_matchup_matrix(self, match_type: str = "all") -> pd.DataFrame:
        """Calculate the matchup correlation matrix between characters with optional match type filter."""
        df = pd.read_csv(self.csv_path)

        # Initialize matchup matrix with zeros
        matchup_matrix = pd.DataFrame(0, 
                                    index=self.characters,
                                    columns=self.characters,
                                    dtype=float)

        # Count wins and total matches for each matchup
        for _, row in df.iterrows():
            current_match_type = self._get_match_type(row['team1'], row['team2'])

            # Skip if match type doesn't match filter
            if match_type != "all" and current_match_type != match_type:
                continue

            team1 = row['team1'].split('+')
            team2 = row['team2'].split('+')

            # Determine winner and loser teams
            team1_won = row['score1'] > row['score2']

            # Update matchup statistics
            for char1 in team1:
                for char2 in team2:
                    if team1_won:
                        matchup_matrix.loc[char1, char2] += 1
                        total_matches = matchup_matrix.loc[char1, char2] + matchup_matrix.loc[char2, char1]
                        win_rate = matchup_matrix.loc[char1, char2] / total_matches if total_matches > 0 else 0.5
                        matchup_matrix.loc[char1, char2] = win_rate
                        matchup_matrix.loc[char2, char1] = 1 - win_rate
                    else:
                        matchup_matrix.loc[char2, char1] += 1
                        total_matches = matchup_matrix.loc[char1, char2] + matchup_matrix.loc[char2, char1]
                        win_rate = matchup_matrix.loc[char2, char1] / total_matches if total_matches > 0 else 0.5
                        matchup_matrix.loc[char2, char1] = win_rate
                        matchup_matrix.loc[char1, char2] = 1 - win_rate

        return matchup_matrix

    def get_match_history(self, match_type: str = "all") -> pd.DataFrame:
        """Return the match history with optional match type filter."""
        df = pd.read_csv(self.csv_path)
        if match_type != "all":
            df = df[df.apply(lambda row: self._get_match_type(row['team1'], row['team2']) == match_type, axis=1)]
        return df

    def validate_teams(self, team1: List[str], team2: List[str]) -> bool:
        """Validate team composition."""
        # Check if all characters exist in the character list
        all_chars = team1 + team2
        if not all(char in self.characters for char in all_chars):
            return False

        # Check for duplicates across teams
        if len(set(all_chars)) != len(all_chars):
            return False

        # Check team sizes (must be either 1v1 or 2v2)
        if len(team1) != len(team2) or len(team1) not in [1, 2]:
            return False

        return True