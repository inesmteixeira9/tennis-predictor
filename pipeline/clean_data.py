# Usage: python -m pipeline.clean_data
import logging
import pandas as pd
from libs import data_utils
from . import PARAMS

class DataCleaner():
    @staticmethod
    def main() -> pd.DataFrame:
        """Transform ATP and WTA raw data into a single cleaned DataFrame.
        Saves data in cleaned_data.csv located in interim_data_path.

        Returns:
            pd.DataFrame: Transformed data.
        """
        # Read data
        raw_data_path = PARAMS.data_path.raw.root_dir
        atp_file_path = raw_data_path + PARAMS.data_path.raw.atp
        wta_file_path = raw_data_path + PARAMS.data_path.raw.wta
        atp_df = data_utils.read_data(atp_file_path)
        wta_df = data_utils.read_data(wta_file_path)

        # Rename columns to lowercase
        atp_df.rename(columns=lambda x: x.lower(), inplace=True)

        # Drop matches that aren't completed
        atp_df = atp_df[atp_df['comment'] == "Completed"]

        # Correct typos: Replace occurrences of '..' with '.0'
        wta_df['b365l'] = wta_df['b365l'].apply(lambda x: str(x).replace('..', '.0') if isinstance(x, str) and '..' in x else x)

        # Select relevant columns
        schema = PARAMS.data_schemas.raw
        atp_df = atp_df[schema.keys()]
        wta_df = wta_df[schema.keys()]

        # Format data according to data schemas toml file
        logging.info('Formating atp data according to the imported schema...')
        atp_df = data_utils.format_data(atp_df, schema)

        logging.info('Formating wta data according to the imported schema...')
        wta_df = data_utils.format_data(wta_df, schema)

        # Concat datasets
        df = pd.concat([atp_df, wta_df])

        # Drop null values
        df = df.dropna()

        df['winner_is_p1'] = df.apply(DataCleaner.winner_is_p1, axis=1)
        df['odd_p1'] = df.apply(lambda row: row['b365w'] if row['winner_is_p1'] == 1 else row['b365l'], axis=1)
        df['odd_p2'] = df.apply(lambda row: row['b365w'] if row['winner_is_p1'] == 0 else row['b365l'], axis=1)

        interim_data_path = PARAMS.data_path.interim.root_dir
        df.to_csv(interim_data_path + 'cleaned_data.csv')

        return df
    
    @staticmethod
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
        
    @staticmethod
    def remove_outliers(df: pd.DataFrame):
        # Define the conditions
        conditions = [
            (df['odd_p1'] < 1.35) & (df['rank_p1'] > 1000),
            (df['odd_p1'] < 1.35) & (df['rank_diff'] < -700),
            (df['odd_p1'] > 1.6) & (df['rank_diff'] > 700),
            (df['odd_p1'] < 1.1) & (df['odd_diff'] < 5),
            (df['odd_p1'] < 1.1) & (df['odd_ratio'] < 5),
            (df['odd_p1'] > 1.5) & (df['consecutive_results'] > 20),
            (df['odd_p1'] < 1.3) & (df['consecutive_results'] < -10),
            (df['rank_evol_p1'] < -500),
            (df['rank_evol_p1'] > 500),
            (df['rank_evol_p2'] < -500),
            (df['rank_evol_p2'] > 500),
            (df['odd_p2'] > 20) & (df['rank_p1'] > 200),
        ]

        # Combine conditions with logical OR
        outliers = conditions[0]
        for condition in conditions[1:]:
            outliers |= condition

        # Filter out the outliers
        return df[~outliers]

if __name__ == "__main__": # Won't be executed when module is imported
    cleaned_data = DataCleaner.main()