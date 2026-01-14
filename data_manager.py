import pandas as pd
import os
import numpy as np
from typing import List, Tuple, Dict

class DataManager:
    def __init__(self, csv_path: str = "matches.csv"):
        self.csv_path = csv_path
        self.characters = sorted([
            "Сунь-укун", "Йененга", "Кровавая Мэри", "Ахиллес", 
            "Ода Набунага", "Томоэ", "Джин", "Гудини", 
            "Три сестры", "Гамлет", "Шекспир", "Титания",
            "Красная шапочка","Беовульф","Дракула","Шерлок","Невидимка","Джекил и Хайд", "Брюс Ли", "Смерть", "ИИ"
        ])
        self.players = sorted([
            "Артем", "Кристина", "Денис", "Татьяна",
            "Танюша", "ОлегВ", "Вадим", "Олежка", "Виталик",
            "СашаВ", "АняВ", "Даня", "Карина"
        ])
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['team1', 'team2', 'score1', 'score2', 'timestamp'])
            df.to_csv(self.csv_path, index=False)

    def add_match(self, team1: List[str], team2: List[str], score1: int, score2: int, p1: List[str], p2: List[str]) -> bool:
        """Add a new match result to the CSV file."""
        try:
            new_match = pd.DataFrame([{
                'team1': '+'.join(team1),
                'team2': '+'.join(team2),
                'score1': score1,
                'score2': score2,
                'timestamp': pd.Timestamp.now(),
                'p1': '!'.join(p1),
                'p2': '!'.join(p2),
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
                    else:
                        matchup_matrix.loc[char2, char1] += 1
        winrate_matrix = pd.DataFrame(matchup_matrix, dtype=str)
        for col in winrate_matrix:
            for i in winrate_matrix:
                total_matches = matchup_matrix.loc[col, i] + matchup_matrix.loc[i, col]
                winrate_matrix.loc[col, i] = str(matchup_matrix.loc[col, i] / total_matches) if total_matches > 0 else ''
                if total_matches > 0:
                    winrate_matrix.loc[col, i] = str(round(float(winrate_matrix.loc[col, i]) * 100, 2)) + '% (' + str(int(matchup_matrix.loc[col, i])) + '/' + str(int(total_matches)) + ')'
        return winrate_matrix

    def get_score_matrix(self, match_type: str = "all") -> pd.DataFrame:
        """Calculate the score correlation matrix between characters with optional match type filter."""
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
                        matchup_matrix.loc[char1, char2] += float(row['score1'])
                    else:
                        matchup_matrix.loc[char2, char1] += float(row['score2'])
        winrate_matrix = pd.DataFrame(matchup_matrix, dtype=str)
        for col in winrate_matrix:
            for i in winrate_matrix:
                total_matches = matchup_matrix.loc[col, i] + matchup_matrix.loc[i, col]
                winrate_matrix.loc[col, i] = str(matchup_matrix.loc[col, i] / total_matches) if total_matches > 0 else ''
                if total_matches > 0:
                    winrate_matrix.loc[col, i] = str(round(float(winrate_matrix.loc[col, i]) * 100, 2)) + '% (' + str(int(matchup_matrix.loc[col, i])) + '/' + str(int(total_matches)) + ')'
        return winrate_matrix
        
    def get_match_history(self, match_type: str = "all") -> pd.DataFrame:
        """Return the match history with optional match type filter."""
        df = pd.read_csv(self.csv_path)
        if match_type != "all":
            df = df[df.apply(lambda row: self._get_match_type(row['team1'], row['team2']) == match_type, axis=1)]
        return df

    def get_players_winrate(self, match_type: str = "all") -> pd.DataFrame:
        """Return players winrate."""
        df = pd.read_csv(self.csv_path)
        if match_type != "all":
            df = df[df.apply(lambda row: self._get_match_type(row['team1'], row['team2']) == match_type, axis=1)]
        players_data = []
    
        for _, row in df.iterrows():
            # Определяем победившую команду
            if row['score1'] > row['score2']:
                winning_team = 1
            elif row['score1'] < row['score2']:
                winning_team = 2
            else:
                winning_team = 0  # ничья
            
            # Обрабатываем игроков команды 1 (убираем дубликаты в рамках одной команды)
            p1_players = [row['p1']] if '!' not in str(row['p1']) else str(row['p1']).split('!')
            unique_p1_players = list(set(p1_players))  # убираем дубликаты в команде
            for player in unique_p1_players:
                players_data.append({
                    'Player': player,
                    'game_result': 1 if winning_team == 1 else 0
                })
            
            # Обрабатываем игроков команды 2 (убираем дубликаты в рамках одной команды)
            p2_players = [row['p2']] if '!' not in str(row['p2']) else str(row['p2']).split('!')
            unique_p2_players = list(set(p2_players))  # убираем дубликаты в команде
            for player in unique_p2_players:
                players_data.append({
                    'Player': player,
                    'game_result': 1 if winning_team == 2 else 0
                })
        
        # Создаем DataFrame и вычисляем статистику
        players_df = pd.DataFrame(players_data)
        
        # Для каждого игрока считаем общее количество уникальных игр и количество побед
        result_df = players_df.groupby('Player').agg(
            total_games=('game_result', 'count'),
            wins=('game_result', 'sum')
        ).reset_index()
        
        # Вычисляем процент побед
        result_df['win_percentage'] = (result_df['wins'] / result_df['total_games']) * 100
        result_df['win_percentage'] = result_df['win_percentage'].round(2)
        result_df.set_index('Player', inplace=True)
        return result_df
    
    def get_players_winrate_by_character(self, player: str, match_type: str = "all") -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if match_type != "all":
            df = df[df.apply(lambda row: self._get_match_type(row['team1'], row['team2']) == match_type, axis=1)]
        # Создаем список для хранения данных
        character_data = []
        
        # Проходим по всем строкам DataFrame
        for _, row in df.iterrows():
            # Получаем список персонажей и игроков для каждой команды
            team1_characters = str(row['team1']).split('+')
            team1_players = [row['p1']] if '!' not in str(row['p1']) else str(row['p1']).split('!')
            team2_characters = row['team2'].split('+')
            team2_players = [row['p2']] if '!' not in str(row['p2']) else str(row['p2']).split('!')
            
            # Проверяем, был ли наш игрок в команде 1
            if player in team1_players:
                # Находим индекс игрока в команде
                player_index = team1_players.index(player)
                # Получаем соответствующего персонажа
                character = team1_characters[player_index]
                # Определяем результат игры (1 - победа, 0 - поражение или ничья)
                result = 1 if row['score1'] > row['score2'] else 0
                character_data.append({'character': character, 'result': result})
                if team1_players.count(player) > 1:
                    character = team1_characters[1]
                    result = 1 if row['score1'] > row['score2'] else 0
                    character_data.append({'character': character, 'result': result})
            
            # Проверяем, был ли наш игрок в команде 2
            elif player in team2_players:
                # Находим индекс игрока в команде
                player_index = team2_players.index(player)
                # Получаем соответствующего персонажа
                character = team2_characters[player_index]
                # Определяем результат игры (1 - победа, 0 - поражение или ничья)
                result = 1 if row['score2'] > row['score1'] else 0
                character_data.append({'character': character, 'result': result})
                if team2_players.count(player) > 1:
                    character = team2_characters[1]
                    result = 1 if row['score2'] > row['score1'] else 0
                    character_data.append({'character': character, 'result': result})
        
        # Если игрок не найден ни в одной игре
        if not character_data:
            return pd.DataFrame(columns=['character', 'games_played', 'wins', 'win_rate'])
        
        # Создаем DataFrame из собранных данных
        character_df = pd.DataFrame(character_data)
        
        # Группируем по персонажу и считаем статистику
        result_df = character_df.groupby('character').agg(
            games_played=('result', 'count'),
            wins=('result', 'sum')
        ).reset_index()
        
        # Вычисляем процент побед
        result_df['win_rate'] = (result_df['wins'] / result_df['games_played']) * 100
        result_df['win_rate'] = result_df['win_rate'].round(2)

        result_df.set_index('character', inplace=True)
        
        return result_df

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
    
    def get_players_score_matrix(self, match_type: str = "all") -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if match_type != "all":
            df = df[df.apply(lambda row: self._get_match_type(row['team1'], row['team2']) == match_type, axis=1)]
        # Получаем список всех уникальных игроков
        all_players = set()
        
        for _, row in df.iterrows():
            # Обрабатываем игроков, удаляя дубликаты в рамках одной команды
            p1_players = list(set([str(row['p1'])] if '!' not in str(row['p1']) else str(row['p1']).split('!')))
            p2_players = list(set([str(row['p2'])] if '!' not in str(row['p2']) else str(row['p2']).split('!')))
            
            all_players.update(p1_players)
            all_players.update(p2_players)
        
        all_players = sorted(list(all_players))
        player_count = len(all_players)
        
        # Создаем матрицы для хранения результатов
        win_matrix = pd.DataFrame(
            np.zeros((player_count, player_count), dtype=int),
            index=all_players,
            columns=all_players
        )
        games_matrix = pd.DataFrame(
            np.zeros((player_count, player_count), dtype=int),
            index=all_players,
            columns=all_players
        )
        
        # Заполняем матрицы
        for _, row in df.iterrows():
            # Удаляем дубликаты игроков в командах
            p1_players = list(set([str(row['p1'])] if '!' not in str(row['p1']) else str(row['p1']).split('!')))
            p2_players = list(set([str(row['p2'])] if '!' not in str(row['p2']) else str(row['p2']).split('!')))
            
            # Определяем результат матча
            if row['score1'] > row['score2']:
                winners = p1_players
                losers = p2_players
            elif row['score1'] < row['score2']:
                winners = p2_players
                losers = p1_players
            else:
                continue  # Ничья не учитывается
            
            # Учитываем по одной победе для каждой уникальной пары игроков
            for winner in winners:
                for loser in losers:
                    win_matrix.at[winner, loser] += 1
            
            # Учитываем по одной игре для каждой уникальной пары между командами
            for p1 in p1_players:
                for p2 in p2_players:
                    games_matrix.at[p1, p2] += 1
                    games_matrix.at[p2, p1] += 1
        
        # Создаем результирующую матрицу
        result_matrix = pd.DataFrame(
            np.empty((player_count, player_count), dtype=object),
            index=all_players,
            columns=all_players
        )
        
        for i in all_players:
            for j in all_players:
                if i == j:
                    result_matrix.at[i, j] = "-"
                elif games_matrix.at[i, j] > 0:
                    wins = win_matrix.at[i, j]
                    total = games_matrix.at[i, j]
                    percentage = round((wins / total) * 100, 2)
                    result_matrix.at[i, j] = f"{percentage}% ({wins}/{total})"
                else:
                    result_matrix.at[i, j] = "-"
        
        return result_matrix