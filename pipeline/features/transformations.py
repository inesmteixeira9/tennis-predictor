import logging
import pandas as pd
import numpy as np

class Transformations():
    @staticmethod
    def logaritmic_trasformation(df: pd.DataFrame, features: list):

        skewed_features = ['rank_p1', 'rank_p2', 'rank_ratio', 'odd_diff', 'odd_ratio', 'record_p1', 'record_p2']

        # Apply logaritmic trasformation to skewed features
        for col in skewed_features:
            if df[col].min() == 0:
                df[col] = df[col].apply(lambda x: x + 1)
            else:
                df[col] = df[col].apply(lambda x: x + 1 + abs(df[col].min()))
            df[f'log_{col}'] = df[col].apply(lambda x: np.log(x))

            # Updating features list
            i = features.index(col)
            features[i] = f'log_{col}'

        return df.drop(columns=skewed_features), features
    
    @staticmethod
    def bin_features(df: pd.DataFrame, features: list):
        
        binning_features = {'rank_diff': 14,
                            'h2h': 2,
                            'rank_evol_p1': 2,
                            'rank_evol_p2': 2,
                            'rank_combined': 14}

        # Iterate over each feature in binning_features
        for col, bin in binning_features.items():
            # Calculate quantile boundaries
            quantiles = np.linspace(0, 100, bin+1)
            bin_boundaries = np.percentile(df[col], quantiles)

            # Apply binning transformation
            df[f'{col}_binned'] = pd.cut(df[col], bins=bin_boundaries, labels=False, include_lowest=True)

            binning_features[col] = (bin, (bin_boundaries))

            i = features.index(col) 
            features[i] = f'{col}_binned'
        
        df = df.drop(columns=binning_features)

        # Convert into a binned feature based on the previous vizualisation
        df['consecutive_wins_p1'] = df['consecutive_wins_p1'].apply(lambda x: 0 if x < 0 else (0.5 if 0 <= x <= 1 else 1))          
        df['consecutive_losses_p2'] = df['consecutive_losses_p2'].apply(lambda x: 0 if x < 0.9 else (0.5 if 0.9 <= x <= 1.1 else 1))
        df['consecutive_results'] = df['consecutive_results'].apply(lambda x: 0 if x < 1 else 1)
        df['consecutive_wins_p2'] = df['consecutive_wins_p2'].apply(lambda x: 0 if x > 1 else 1)
        df['consecutive_losses_p1'] = df['consecutive_losses_p1'].apply(lambda x: 0 if x >= 1 else 1)

        unclear_association_features =  ['log_rank_p2', 'log_rank_ratio', 'log_odd_diff', 'log_record_p1', 'log_record_p2']
        
        return df.drop(columns=unclear_association_features), features
    
    @staticmethod
    def invert_features(df: pd.DataFrame, features: list):
        # Invert features that have a negative impact 
        inverse_features = ['log_rank_p1']

        for col in inverse_features:
            df[f'inverted_{col}'] = 1 / df[col]
            i = features.index(col) 
            features[i] = f'inverted_{col}'

        return df.drop(columns=inverse_features), features