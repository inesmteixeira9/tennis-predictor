import logging
import pandas as pd
from typing import Tuple
from datetime import datetime

class OddsFeatures():
    @staticmethod
    def add_odd_dif(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add column 'odd_diff' to df with odd_p2 - odd_p1.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added odd_diff column and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding odd's differences...")

        df['odd_diff'] = df['odd_p2'] - df['odd_p1']

        features.extend(['odd_diff'])
        end_time = datetime.now()
        logging.info(f" -> Added odd's differences. ({(end_time-begin_time).total_seconds()})")

        return df, features

    @staticmethod
    def add_odd_ratio(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Add column 'Odd_ratio' to df with odd_p2 / odd_p1.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added odd_ratio column and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding odd's ratio...")

        df['odd_ratio'] = df['odd_p2'] / df['odd_p1']
        
        features.extend(['odd_ratio'])
        end_time = datetime.now()
        logging.info(f" -> Added odd's ratio. ({(end_time-begin_time).total_seconds()})")

        return df, features