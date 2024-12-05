
import logging
import pandas as pd
from datetime import datetime
from typing import Tuple

class SurfaceFeatures():
    @staticmethod
    def OHE_surface(df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, list]:
        """Replace column 'Surface' with one-hot encoded columns.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with one-hot encoded surface columns and updated features list.
        """
        begin_time = datetime.now()

        logging.info("One-hot-encoding surface...")

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
            logging.info(f" -> One-hot-encoded surface. ({(end_time-begin_time).total_seconds()})")

        except Exception as e:
            logging.error(f"Error in one-hot-encoding surface: {e}")
        
        return df, features