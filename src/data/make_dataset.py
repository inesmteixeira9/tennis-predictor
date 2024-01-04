"""
python src/data/make_dataset.py C:/Users/inesm/projectos/tennis-predictor/data/raw/ C:/Users/inesm/projectos/tennis-predictor/data/interim/
"""
import click
import logging
from pathlib import Path
import os
from dotenv import find_dotenv, load_dotenv
import glob
import pandas as pd
import warnings

logger = logging.getLogger(__name__)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    
    # Load data 
    logger.info('loading raw data')
    dataset = load_data(input_filepath)
    print(dataset)


    logger.info('pre-processing raw data')
    # Remove uncompleted matches, select relevant columns 
    # and drop rows with missing values.
    dataset = filter_data(dataset)
    
    # Save data 
    logger.info('saving pre-processed data')
    save_data(dataset, 'dataset.pkl', output_filepath)


def load_data(data_dir, file_name=None):
    # Load every file if file_name is not specify
    if file_name == None:
        logger.info(f"loading all files in {data_dir}")
        # Use glob to get a list of all matching file paths
        data_files = glob.glob(data_dir + '*.csv')

        # Initialize an empty DataFrame to store the concatenated data
        dataset = pd.DataFrame()

        # Loop through the file paths and read each CSV file into a DataFrame
        for data_dir in data_files:
            try:
                df = pd.read_pickle(data_dir)
            except:
                df = pd.read_csv(data_dir)
            dataset = pd.concat([dataset, df], ignore_index=True)

    # Load the specified file_name 
    else:
        logger.info(f"loading {file_name}")
        try:
            dataset = pd.read_pickle(os.path.join(data_dir, file_name))
        except:
            dataset = pd.read_csv(os.path.join(data_dir, file_name))
    
    return dataset

def filter_data(dataset):
    """Remove uncompleted matches, select relevant columns and and drop rows with missing values.
    """

    # Drop matches that aren't completed
    if 'Comment' in dataset.columns:
        dataset = dataset[dataset['Comment'] == "Completed"]
        logger.info("selecting completed matches from db.")
    else:
        logger.info("the 'Comment' column does not exist in the DataFrame.")
        
    dataset = dataset[dataset['Comment'] == "Completed"]

    # Select relevant columns
    columns_to_select = [
        'Date',
        'Winner',
        'Loser',
        'WRank',  # winner's ranking
        'LRank',  # loser's ranking
        'B365W',  # odd that was given for the winner by B365
        'B365L',  # odd that was given for the loser by B365
        'Surface'  # type of court
    ]

    dataset = dataset[columns_to_select]
    logger.info(f"Selecting columns: {columns_to_select}.")

    # Drop rows with any missing values
    dataset = dataset.dropna()

    # Define datatypes
    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%d/%m/%Y")

    dataset = dataset.astype({
        'Winner': str,
        'Loser': str,
        'WRank': int,
        'LRank': int,
        'B365W': float,
        'B365L': float,
        'Surface': str,
    })

    # Ensure df is in chronological order
    dataset = dataset.sort_values(by="Date")

    # Add match_id column
    dataset['match_id'] = dataset.index

    dataset = dataset.drop('Date', axis=1)
    
    return dataset

def save_data(data, file_name, data_dir):
    """ Takes an object, transforms in a pd dataframe
    and saves it in a pickle file in the given directory."""

    # Check if data form at is DataFrame
    if not isinstance(data, pd.DataFrame):
        # Transform data into a DataFrame
        data = pd.DataFrame(data)

    # Check if the file already exists
    if os.path.isfile(file_name):
        # If the file exists, remove it
        os.remove(data_dir + file_name)

    # Save DataFrame as a pickle
    data.to_pickle(data_dir + file_name)
    logger.info(f"saved {file_name} to {data_dir}")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()



