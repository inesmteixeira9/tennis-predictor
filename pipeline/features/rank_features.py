import logging
import pandas as pd
from typing import Tuple
from datetime import datetime

class RankFeatures():
    @staticmethod
    def add_ranks(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """For each row, get the ranking of P1 and P2.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added rank columns and updated features list.
        """
        begin_time = datetime.now()
        logging.info("Adding player's rankings...")
        
        df['rank_p1'] = df.apply(lambda row: row['wrank'] if row['winner_is_p1'] == 1 else row['lrank'], axis=1)
        df['rank_p2'] = df.apply(lambda row: row['wrank'] if row['winner_is_p1'] == 0 else row['lrank'], axis=1)

        features.extend(['rank_p1', 'rank_p2'])

        end_time = datetime.now()
        logging.info(f" -> Added player's rankings. ({(end_time-begin_time).total_seconds()})")
        return df, features

    @staticmethod
    def add_rank_dif(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add column 'rank_diff' to df with rank_p2 - rank_p1.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added rank_diff column and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding ranking's differences...")

        df['rank_diff'] = df['rank_p2'] - df['rank_p1']
        
        features.extend(['rank_diff'])
        end_time = datetime.now()
        logging.info(f" -> Added ranking's differences. ({(end_time-begin_time).total_seconds()})")
        
        return df, features


    @staticmethod
    def add_rank_ratio(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add column 'rank_ratio' to df with rank_p2 / rank_p1.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added rank_ratio column and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding ranking's ratio...")

        df['rank_ratio'] = df['rank_p2'] / df['rank_p1']

        features.extend(['rank_ratio'])
        end_time = datetime.now()
        logging.info(f" -> Added ranking's ratio. ({(end_time-begin_time).total_seconds()})")
        
        return df, features


    @staticmethod
    def add_rank_evolution(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Apply update_rank_evol_and_df for each player.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding ranking evolution...")

        # Initialize dictionary to store player's rankings
        rank_evol = {}

        # Initialize 'rank_evol' with 0
        df['rank_evol_p1'] = 0
        df['rank_evol_p2'] = 0

        # Iterate over rows
        for index, row in df.iterrows():
            date = row['date']
            winner = row['winner']
            loser = row['loser']
            wrank = row['wrank']
            lrank = row['lrank']

            # Update DataFrame with consecutive wins and losses
            if row['winner_is_p1'] == 1:      
                RankFeatures.update_rank_evol_and_df(df, index, rank_evol, date, winner, wrank, 1)  
                RankFeatures.update_rank_evol_and_df(df, index, rank_evol, date, loser, lrank, 2)  
            else:  
                RankFeatures.update_rank_evol_and_df(df, index, rank_evol, date, winner, wrank, 2)  
                RankFeatures.update_rank_evol_and_df(df, index, rank_evol, date, loser, lrank, 1) 
    
        # Convert the columns to numeric
        cols = ['rank_evol_p1', 'rank_evol_p2']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        features.extend(cols)
        end_time = datetime.now()
        logging.info(f" -> Added ranking evolution. ({(end_time-begin_time).total_seconds()})")
        
        return df, features

    @staticmethod
    def add_rank_combined(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add columns for the combined rankings and rankings evolution.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added combined rankings and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding combined rankings...")

        df['rank_combined'] = - df['rank_p1'] + df['rank_p2'] + df['rank_evol_p1'] - df['rank_evol_p2']
        
        features.extend(['rank_combined'])
        end_time = datetime.now()
        logging.info(f" -> Added combined rankings. ({(end_time-begin_time).total_seconds()})")
        
        return df, features

    @staticmethod
    def update_rank_evol_and_df(df, index, rank_evol, date, player_name, rank, player):
        """Add columns for the ranking evolution.
        
        Args:
            df (pd.DataFrame): DataFrame containing match data.
            index: df index.
            rank_evol (dict): {'player' : (yyyy-mm-dd, ranking, last_rank_evol)}
            date (str): matche's date
            player_name (str)
            rank (int)
            player (int): 1 if it's the player with the lowest odd else 2
        """

        # if player in rank_evol
        if rank_evol.get(player_name) != None:

            # if date is more recent
            if pd.to_datetime(date) > pd.to_datetime(rank_evol[player_name][0]):

                # Update df
                df.at[index, f'rank_evol_p{player}'] = rank_evol[player_name][1] - rank 

                # Update rank_evol
                rank_evol[player_name] = (date, rank, rank_evol[player_name][1] - rank)

            else:
                # Update df with last_rank_evol
                df.at[index, f'rank_evol_p{player}'] = rank_evol[player_name][2] 

                # Update or add new item to rank_evol
                rank_evol[player_name] = (date, rank, rank_evol[player_name][1] - rank)

        else:
            # Add new item to rank_evol
            rank_evol[player_name] = (date, rank, 0)
