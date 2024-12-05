import logging
from datetime import datetime

class H2HFeatures():
    @staticmethod
    def add_h2h(dataset, features):
        """Add column 'h2h' to df.

        Args:
            df (pd.DataFrame): DataFrame containing match data.
            features (list): List to store feature names.

        Returns:
            Tuple[pd.DataFrame, list]: DataFrame with added odd_ratio column and updated features list.
        """
        begin_time = datetime.now()

        logging.info("Adding h2h...")

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
        logging.info(f" -> Added h2h. ({(end_time-begin_time).total_seconds()})")

        return dataset, features