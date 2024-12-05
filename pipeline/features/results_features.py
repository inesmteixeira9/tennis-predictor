import logging
import pandas as pd
from typing import Tuple
from datetime import datetime

class ResultsFeatures():
    @staticmethod
    def add_consecutive_wins_and_losses(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add columns for the consecutives wins and losses for each player.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding consecutives wins and losses...")

        # Initialize dictionary to store consecutive wins and losses
        consecutive_wins = {}
        consecutive_losses = {}

        # Initialize 'consecutive_wins' and 'consecutive_losses' columns with None
        cols = ['consecutive_wins_p1', 'consecutive_losses_p1', 'consecutive_wins_p2', 'consecutive_losses_p2']
        for col in cols:
            df[col] = None

        # Iterate over rows
        for index, row in df.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            # Update or initialize consecutive wins and losses for the winner and loser
            consecutive_wins[winner] = consecutive_wins.get(winner, 0) + 1
            consecutive_losses[loser] = consecutive_losses.get(loser, 0) + 1

            # Reset consecutive wins for the loser and consecutive losses for the winner
            consecutive_wins[loser] = 0
            consecutive_losses[winner] = 0

            # Update DataFrame with consecutive wins and losses
            if row['winner_is_p1'] == 1:
                df.at[index, 'consecutive_wins_p1'] = consecutive_wins[winner] - 1
                df.at[index, 'consecutive_losses_p2'] = consecutive_losses[loser] - 1
                df.at[index, 'consecutive_losses_p1'] = 0
                df.at[index, 'consecutive_wins_p2'] = 0
            else:
                df.at[index, 'consecutive_wins_p2'] = consecutive_wins[winner] - 1
                df.at[index, 'consecutive_losses_p1'] = consecutive_losses[loser] - 1
                df.at[index, 'consecutive_wins_p1'] = 0
                df.at[index, 'consecutive_losses_p2'] = 0

        # Convert the columns to numeric
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        features.extend(cols)
        end_time = datetime.now()
        logging.info(f" -> Added consecutives wins and losses. ({(end_time-begin_time).total_seconds()})")
        
        return df, features


    @staticmethod
    def add_consecutive_results(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add columns for the consecutives results for each couple of players.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding consecutives results...")

        df['consecutive_results'] = df['consecutive_wins_p1'] - df['consecutive_wins_p2'] - df['consecutive_losses_p1'] + df['consecutive_losses_p2']

        features.extend(['consecutive_results'])
        end_time = datetime.now()
        logging.info(f" -> Added consecutives results. ({(end_time-begin_time).total_seconds()})")
        
        return df, features


    @staticmethod
    def add_records(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add columns for players' records.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding players' records...")

        # Initialize dictionary to store player's records
        records = {}

        # Initialize 'rank_evol' with 0
        df['record_p1'] = 0
        df['record_p2'] = 0

        # Iterate over rows
        for index, row in df.iterrows():
            winner = row['winner']
            loser = row['loser']

            # Update DataFrame with consecutive wins and losses
            if row['winner_is_p1'] == 1:   

                # if player in records
                if records.get(winner) != None:
    
                    # Update df
                    df.at[index, 'record_p1'] = records[winner]

                    # Update p1 records
                    records[winner] = records[winner] + 1

                else:
                    # Update p1 records
                    records[winner] = 1

                # if player in records
                if records.get(loser) != None:

                    # Update df
                    df.at[index, 'record_p2'] = records[loser]

                    # Update p2 records
                    records[loser] = records[loser] - 1

                else:
                    # Update p1 records
                    records[loser] = -1

            else:  

                # if player in records
                if records.get(winner) != None:
    
                    # Update df
                    df.at[index, 'record_p2'] = records[winner]

                    # Update p1 records
                    records[winner] = records[winner] + 1

                else:
                    # Update p1 records
                    records[winner] = 1

                # if player in records
                if records.get(loser) != None:

                    # Update df
                    df.at[index, 'record_p1'] = records[loser]

                    # Update p2 records
                    records[loser] = records[loser] - 1

                else:
                    # Update p1 records
                    records[loser] = -1
    
        # Convert the columns to numeric
        cols = ['record_p1', 'record_p2']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        features.extend(cols)
        end_time = datetime.now()
        logging.info(f" -> Added players' records. ({(end_time-begin_time).total_seconds()})")
        
        return df, features