"""
This script contains all calculations for potencial features.
It builds two datasets:
 - features.pkl
 - features_classified.pkl

python src/features/build_features.py C:/Users/inesm/projectos/tennis-predictor/data/interim/ C:/Users/inesm/projectos/tennis-predictor/data/processed/
"""
import click
import logging
import sys
sys.path.append('src/data/')
from make_dataset import load_data
from make_dataset import save_data
import pandas as pd

from pathlib import Path


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn dataset from (../interim) into
        insightful data ready to be trained (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    
    
    # Load data using the load_data function
    logger.info('loading raw data')
    dataset = load_data(input_filepath, file_name='dataset.pkl')
    
    # Add rankings of each player
    dataset['RankP1'] = dataset.apply(calculate_p1_rank, axis=1)
    dataset['RankP2'] = dataset.apply(calculate_p2_rank, axis=1)
    

    # Add a binary column for each surface
    dataset = OHE_surface(dataset)
    
    # Add players odds
    dataset = add_players_odds(dataset)
    
    # Add head-to-head
    dataset = calculate_h2h(dataset)
    
    # Add probability of P1 being the winner based on the odds 
    dataset = add_prob_p1_wins(dataset)
    
    # Define target: P1 is the winner
    dataset = define_target(dataset)

    # build another dataset with classified features
    dataset_classified = dataset

    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'RankP1')

    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'RankP2')

    # Add difference of rankings
    dataset = calculate_rank_dif(dataset)
    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'Rank_dif', num_classes=15)
    
    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'OddP1', num_classes=15)
    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'OddP2', num_classes=15)
    
    # Add difference of odds
    dataset = calculate_odd_dif(dataset)
    dataset, dataset_classified = classify_feature(dataset, dataset_classified, 'Odd_dif', num_classes=15)
    
    handling_ranks_outliers(dataset)
    
    print(dataset.describe())
    print(dataset_classified.describe())
    
    # Drop Leaky features
    features = dataset.drop(['Winner', 'Loser', 'WRank', 'LRank', 'B365W', 'B365L'], axis = 1)
    features_classified = dataset_classified.drop(['Winner', 'Loser', 'WRank', 'LRank', 'B365W', 'B365L'], axis = 1)

    logger.info('saving features')
    save_data(features, 'features.pkl', output_filepath)
    save_data(features_classified, 'features_classified.pkl', output_filepath)

def classify_feature(dataset, dataset_classified, col_name, num_classes=25):
    """Define intervals (classes) with the same frequency 
    and classify numeric feature.
    Add classify features to dataset_classified.
    """
    feature = dataset[col_name]
    
    # Calculate the bin edges to have approximately the same frequency in each bin
    bin_edges = pd.qcut(feature, q=num_classes, labels=False, retbins=True)[1]

    new_col_name = col_name + '_classified'
    
    # Split the column into classes based on the bin edges
    dataset_classified[new_col_name] = pd.cut(feature, bins=bin_edges, labels=False, include_lowest=True)
    
    try:
      dataset_classified = dataset_classified.drop(col_name, axis=1)
      print(f'dropping {col_name} from dataset_classified')
    except:
      pass
    
    try:
      dataset = dataset.drop(new_col_name, axis=1)
      print(f'dropping {new_col_name} from dataset_classified')
    except:
      pass
    # Rename the 'old_name' column to 'new_name'
    dataset_classified.rename(columns={new_col_name: col_name}, inplace=True)
    
    return dataset, dataset_classified

# Add features RankP1 and RankP2

def calculate_p1_rank(row):
    # Check if the winner is p1
    if row['B365W'] < row['B365L']: # Winner = P1 (player with the lowest ranking)
      return row['WRank']
    # if players have the same odd consider P1 to be the player with the highest ranking
    elif row['B365W'] == row['B365L']:
      if row['WRank'] < row['LRank']:
        return row['WRank']
      else:
        return row['LRank']
    else: # Winner = P2 (player with the highest ranking)
      return row['LRank']

def calculate_p2_rank(row):
    # Check if the winner is p2
    if row['B365W'] > row['B365L']: # Winner = P2 (player with the highest ranking)
      return row['WRank']
    # if players have the same odd consider P1 to be the player with the highest ranking
    elif row['B365W'] == row['B365L']:
      if row['WRank'] > row['LRank']:
        return row['WRank']
      else:
        return row['LRank']
    else: # Winner = P1 (player with the lowest ranking)
      return row['LRank']
    
def handling_ranks_outliers(dataset):
    # handling outliers setting max limit to 250
    dataset['RankP1'] = dataset.apply(lambda row: 250 if row['RankP1'] > 250 else row['RankP1'], axis =1)
    dataset['RankP2'] = dataset.apply(lambda row: 250 if row['RankP2'] > 250 else row['RankP2'], axis =1)

def calculate_h2h(dataset):
   # Add feature h2h
    cumulative_match_counts = {} # Create an empty DataFrame for the combination of winners and losers
    dataset = dataset.copy()
    dataset['H2H'] = 0

    # Loop over each combination of winners and losers and save the history between them (H2H)
    for index, row in dataset.iterrows():
        winner = row['Winner']
        loser = row['Loser']
        wins_count = cumulative_match_counts.get((winner, loser), 0) + 1
        losses_count =  cumulative_match_counts.get((loser, winner), 0) + 1
        H2H = wins_count - losses_count
        cumulative_match_counts[(winner, loser)] = wins_count
        # if p1 is the winner
        if (row['WRank'] == row['RankP1']):
            dataset.loc[index, 'H2H'] = H2H
        else:
            dataset.loc[index, 'H2H'] = H2H * -1
  
    return dataset

def calculate_rank_dif(dataset):
    # Add feature Rank_dif
    dataset['Rank_dif'] = dataset['RankP1'] - dataset['RankP2']
    return dataset

def OHE_surface(dataset):
    try:
        # One hot encoding 'Surface column
        dataset = pd.get_dummies(dataset, columns=['Surface'], prefix='Surface')
        # Convert the columns containing 'Surface_' to 0 and 1
        dataset[dataset.filter(like='Surface_').columns] = dataset.filter(like='Surface_').astype(int)
    except:
       print("df doesn't contain surface.")
    return dataset

def add_players_odds(dataset):
    # Add OddP1 and OddP2 columns to dataset
    dataset['OddP1'] = dataset.apply(lambda row: row['B365W'] if row['B365W'] < row['B365L'] else row['B365L'], axis =1)
    dataset['OddP2'] = dataset.apply(lambda row: row['B365W'] if row['B365W'] > row['B365L'] else row['B365L'], axis =1)
    
    # handling outliers setting max limit to 10
    dataset['OddP2'] = dataset.apply(lambda row: 10 if row['OddP2'] > 10 else row['OddP2'], axis =1)
    
    return dataset

def calculate_odd_dif(dataset):
    # Add feature Rank_dif
    dataset['Odd_dif'] = dataset['OddP1'] - dataset['OddP2']
    return dataset
  
def add_prob_p1_wins(dataset):
    # Add Y_B365 column with probability of player 1 being the winner through betting odds to dataset
    dataset['Y_B365'] = dataset.apply(lambda row: row['OddP2'] / (row['B365W'] + row['B365L']), axis =1)
    return dataset


def winner_is_p1(row):
    if row['B365W'] < row['B365L']: # Winner = P1 (player with the lowest ranking)
      return 1
    elif row['B365W'] == row['B365L']:
      if row['WRank'] < row['LRank']: # Winner = P1
        return 1
      else:
        return 0
    else:                           # Winner = P2 (player with the highest ranking)
      return 0

def define_target(dataset):
    """ The variable we want to predict in this algorithm is 
    the match outcome (Winner is P1)."""

    # Define Target
    dataset['winner_is_p1'] = dataset.apply(winner_is_p1, axis=1)

    return dataset

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
    
    
