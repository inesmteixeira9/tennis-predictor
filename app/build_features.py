import os
import sys
import pandas as pd
from datetime import datetime
from app.libs.monitoring import log, LogLevel
from typing import Tuple

# Import custom libraries
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.config import SCHEMA
from app.libs import data_utils

def transform_data() -> pd.DataFrame:
    """Transform ATP and WTA raw data into a single cleaned DataFrame.

    Returns:
        pd.DataFrame: Transformed data.
    """
    # Read data
    atp_df = data_utils.read_data(os.path.join(parent_dir, 'data/raw/atp'))
    wta_df = data_utils.read_data(os.path.join(parent_dir, 'data/raw/wta'))

    # Rename columns to lowercase
    atp_df.rename(columns=lambda x: x.lower(), inplace=True)

    # Drop matches that aren't completed
    atp_df = atp_df[atp_df['comment'] == "Completed"]

    # Correct typos: Replace occurrences of '..' with '.0'
    wta_df['b365l'] = wta_df['b365l'].apply(lambda x: str(x).replace('..', '.0') if isinstance(x, str) and '..' in x else x)

    # Select relevant columns
    atp_df = atp_df[SCHEMA['raw'].keys()]
    wta_df = wta_df[SCHEMA['raw'].keys()]

    # Format data according to data schemas toml file
    atp_df = data_utils.format_data(atp_df, SCHEMA['raw'])
    wta_df = data_utils.format_data(wta_df, SCHEMA['raw'])

    # Concat datasets
    df = pd.concat([atp_df, wta_df])

    # Drop null values
    df = df.dropna()

    df['winner_is_p1'] = df.apply(winner_is_p1, axis=1)
    df['odd_p1'] = df.apply(lambda row: row['b365w'] if row['winner_is_p1'] == 1 else row['b365l'], axis=1)
    df['odd_p2'] = df.apply(lambda row: row['b365w'] if row['winner_is_p1'] == 0 else row['b365l'], axis=1)

    return df

def add_ranks(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """For each row, get the ranking of P1 and P2.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added rank columns and updated features list.
    """
    begin_time = datetime.now()
    log.print_log(LogLevel.INFO, "Adding player's rankings...")
    
    df['rank_p1'] = df.apply(lambda row: row['wrank'] if row['winner_is_p1'] == 1 else row['lrank'], axis=1)
    df['rank_p2'] = df.apply(lambda row: row['wrank'] if row['winner_is_p1'] == 0 else row['lrank'], axis=1)

    features.extend(['rank_p1', 'rank_p2'])

    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added player's rankings. ({(end_time-begin_time).total_seconds()})")
    return df, features

def add_rank_dif(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add column 'rank_diff' to df with rank_p2 - rank_p1.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added rank_diff column and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding ranking's differences...")

    df['rank_diff'] = df['rank_p2'] - df['rank_p1']
    
    features.extend(['rank_diff'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added ranking's differences. ({(end_time-begin_time).total_seconds()})")
    
    return df, features

def add_odd_dif(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add column 'odd_diff' to df with odd_p2 - odd_p1.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added odd_diff column and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding odd's differences...")

    df['odd_diff'] = df['odd_p2'] - df['odd_p1']

    features.extend(['odd_diff'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added odd's differences. ({(end_time-begin_time).total_seconds()})")

    return df, features

def add_rank_ratio(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add column 'rank_ratio' to df with rank_p2 / rank_p1.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added rank_ratio column and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding ranking's ratio...")

    df['rank_ratio'] = df['rank_p2'] / df['rank_p1']

    features.extend(['rank_ratio'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added ranking's ratio. ({(end_time-begin_time).total_seconds()})")
    
    return df, features

def add_odd_ratio(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add column 'Odd_ratio' to df with odd_p2 / odd_p1.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added odd_ratio column and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding odd's ratio...")

    df['odd_ratio'] = df['odd_p2'] / df['odd_p1']
    
    features.extend(['odd_ratio'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added odd's ratio. ({(end_time-begin_time).total_seconds()})")

    return df, features

def add_h2h(dataset, features):
    """Add column 'h2h' to df.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added odd_ratio column and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding h2h...")

    # Add feature h2h
    cumulative_match_counts = {} # Create an empty DataFrame for the combination of winners and losers
    dataset = dataset.copy()
    dataset['h2h'] = 0

    # Loop over each combination of winners and losers and save the history between them (H2H)
    for index, row in dataset.iterrows():
        winner = row['winner']
        loser = row['loser']
        wins_count = cumulative_match_counts.get((winner, loser), 0) + 1
        losses_count =  cumulative_match_counts.get((loser, winner), 0) + 1
        H2H = wins_count - losses_count
        cumulative_match_counts[(winner, loser)] = wins_count
        # if p1 is the winner
        if (row['wrank'] == row['rank_p1']):
            dataset.loc[index, 'h2h'] = H2H
        else:
            dataset.loc[index, 'h2h'] = H2H * -1

    features.extend(['h2h'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added h2h. ({(end_time-begin_time).total_seconds()})")

    return dataset, features

def OHE_surface(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Replace column 'Surface' with one-hot encoded columns.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with one-hot encoded surface columns and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "One-hot-encoding surface...")

    try:
        # Replace Carpet per Grass (same surface)
        df['surface'] = df['surface'].replace({'Carpet': 'Grass'})
        df['surface'] = df['surface'].replace({'Greenset': 'Grass'})

        # One hot encoding 'Surface column
        df = pd.get_dummies(df, columns=['surface'], prefix='surface')
        # Convert the columns containing 'surface_' to 0 and 1
        df[df.filter(like='surface_').columns] = df.filter(like='surface_').astype(int)

        features.extend(df.filter(like='surface_').columns)
        end_time = datetime.now()
        log.print_log(LogLevel.INFO, f" -> One-hot-encoded surface. ({(end_time-begin_time).total_seconds()})")

    except Exception as e:
        log.print_log(LogLevel.ERROR, f"Error in one-hot-encoding surface: {e}")
    
    return df, features

def winner_is_p1(row: pd.Series) -> int:
    """Determine if P1 is the winner based on odds and ranking.

    Args:
        row (pd.Series): Row of match data.

    Returns:
        int: 1 if P1 is the winner, 0 otherwise.
    """
    if row['b365w'] < row['b365l']:  # winner is P1
        return 1
    elif row['b365w'] == row['b365l']:
        return 1 if row['wrank'] < row['lrank'] else 0  # winner is P1 if wrank < lrank
    else:  # winner is P2
        return 0

def add_consecutive_wins_and_losses(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add columns for the consecutives wins and losses for each player.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding consecutives wins and losses...")

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
    log.print_log(LogLevel.INFO, f" -> Added consecutives wins and losses. ({(end_time-begin_time).total_seconds()})")
    
    return df, features


def add_consecutive_results(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add columns for the consecutives results for each couple of players.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding consecutives results...")

    df['consecutive_results'] = df['consecutive_wins_p1'] - df['consecutive_wins_p2'] - df['consecutive_losses_p1'] + df['consecutive_losses_p2']

    features.extend(['consecutive_results'])
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added consecutives results. ({(end_time-begin_time).total_seconds()})")
    
    return df, features

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

def add_rank_evolution(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Apply update_rank_evol_and_df for each player.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding ranking evolution...")

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
            update_rank_evol_and_df(df, index, rank_evol, date, winner, wrank, 1)  
            update_rank_evol_and_df(df, index, rank_evol, date, loser, lrank, 2)  
        else:  
            update_rank_evol_and_df(df, index, rank_evol, date, winner, wrank, 2)  
            update_rank_evol_and_df(df, index, rank_evol, date, loser, lrank, 1) 
 
    # Convert the columns to numeric
    cols = ['rank_evol_p1', 'rank_evol_p2']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    features.extend(cols)
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added ranking evolution. ({(end_time-begin_time).total_seconds()})")
    
    return df, features



def add_records(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
    """Add columns for players' records.

    Args:
        df (pd.DataFrame): DataFrame containing match data.
        features (list): List to store feature names.

    Returns:
        Tuple[pd.DataFrame, list]: DataFrame with added consecutive wins/losses columns and updated features list.
    """
    begin_time = datetime.now()

    log.print_log(LogLevel.INFO, "Adding players' records...")

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
    cols = ['rank_evol_p1', 'rank_evol_p2']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    features.extend(cols)
    end_time = datetime.now()
    log.print_log(LogLevel.INFO, f" -> Added players' records. ({(end_time-begin_time).total_seconds()})")
    
    return df, features

