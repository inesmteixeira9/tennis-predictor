# Usage: python -m pipeline.features.build_features
import pandas as pd
from ..clean_data import DataCleaner
from pipeline.features.h2h_features import H2HFeatures
from pipeline.features.odds_features import OddsFeatures
from pipeline.features.rank_features import RankFeatures
from pipeline.features.results_features import ResultsFeatures
from pipeline.features.surface_features import SurfaceFeatures
from pipeline.features.transformations import Transformations
from .. import PARAMS
from libs import data_utils


class FeaturesBuilder():
    @staticmethod
    def main(df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        # save features list for analysis
        features = []
        interim_data_path = PARAMS.data_path.interim.root_dir

        if df.empty:
            df = data_utils.read_data(interim_data_path + 'cleaned_data.csv')

        # Ensure df is in chronological order
        df = df.sort_values(by="date")

        # Add match_id column
        df['match_id'] = df.index

        # Add rankings
        df, features = RankFeatures.add_ranks(df, features)

        # Add the difference and ratio between players
        df, features = RankFeatures.add_rank_dif(df, features)
        df, features = RankFeatures.add_rank_ratio(df, features)

        # Add player's odds
        df['odd_p1'] = df.apply(lambda row: row['b365w'] if row['b365w'] < row['b365l'] else row['b365l'], axis=1)
        df['odd_p2'] = df.apply(lambda row: row['b365w'] if row['b365w'] > row['b365l'] else row['b365l'], axis=1)

        # Add the difference and ratio between players
        df, features = OddsFeatures.add_odd_dif(df, features)
        df, features = OddsFeatures.add_odd_ratio(df, features)

        # Add a binary column for each surface
        df, features = SurfaceFeatures.OHE_surface(df, features)  

        # Add player's head-to-head
        df, features = H2HFeatures.add_h2h(df, features)  

        df, features = ResultsFeatures.add_consecutive_wins_and_losses(df, features)

        # Add combined consecutive results (consecutive_wins_p1 - consecutive_wins_p2 - consecutive_losses_p1 + consecutive_losses_p2)
        df, features = ResultsFeatures.add_consecutive_results(df, features)

        # Add ranking's evolution for each player
        df, features = RankFeatures.add_rank_evolution(df, features)

        # Add combined rankings and rankings evolution (- rank_p1 + rank_p2 + rank_evol_p1 - rank_evol_p2)
        df, features = RankFeatures.add_rank_combined(df, features) 

        # Add players' records (total wins - total losses)
        df, features = ResultsFeatures.add_records(df, features)

        df = DataCleaner.remove_outliers(df)

        # Performed logaritmic transformations to the skewed features 
        df, features = Transformations.logaritmic_trasformation(df, features)
        
        # Bin features that have consistent distributions but uneven frequencies among their values.
        df, features = Transformations.bin_features(df, features)

        # Invert features with a negative impact on the target
        df, features = Transformations.invert_features(df, features)

        # Save data
        print(interim_data_path)
        df.to_csv(interim_data_path + 'features.csv')

        return df


def build_features_by_steps(df: pd.DataFrame, steps: list = None) -> pd.DataFrame:
    """
    Orchestrates the feature-building process.

    Args:
        df (pd.DataFrame): Input DataFrame.
        steps (list): List of steps to execute (e.g., ['add_ranks', 'add_odd_ratio']).
                      If None, all steps are executed.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    features = []
    steps = steps or [
        "add_ranks",
        "add_rank_dif",
        "add_rank_ratio",
        "add_odd_dif",
        "add_odd_ratio",
        "OHE_surface",
        "add_h2h",
        "add_consecutive_wins_and_losses",
        "add_consecutive_results",
        "add_rank_evolution",
        "add_rank_combined",
        "add_records",
        "logarithmic_transformation",
        "bin_features",
        "invert_features",
    ]

    for step in steps:
        step_func = getattr(FeaturesBuilder, step)
        df, features = step_func(df, features)

    return df


if __name__ == "__main__": # Won't be executed when module is imported
    features = FeaturesBuilder.main()