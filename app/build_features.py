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
    """Calculate the consecutives wins and losses for each player.

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
