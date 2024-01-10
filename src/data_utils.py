import os
import logging
import glob
import pandas as pd
import pickle

logger = logging.getLogger(__name__)

def load_csv(file_path: str):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Raises:
        ValueError: Raised if an error occurs while reading empty data from the CSV file.
        ValueError: Raised if an error occurs while reading the CSV file.
        
    Returns:
        pandas.DataFrame: The loaded DataFrame from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as empty_data_error:
        raise ValueError(f"Error reading empty data from file {file_path}: {empty_data_error}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")

    return df

def load_pkl(file_path: str):
    """
    Load data from a pickle file.
    
    Args:
        file_path (str): The path to the pickle file.
        
    Raises:
        ValueError: Raised if an error occurs while unpickling the file.
        ValueError: Raised if an error occurs while reading empty data from the pickled file.
        
    Returns:
        pandas.DataFrame: The loaded DataFrame from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            df = pickle.load(file)
    except pickle.PickleError as unpickling_error:
        raise ValueError(f"Error unpickling file {file_path}: {unpickling_error}")
    except Exception as e:
        raise ValueError(f"Error reading pickle file {file_path}: {str(e)}")

    return df

def load_data(data_path: str, file_name=None):
    """
    Load data from CSV or pickle files in the specified directory or load a specific file.
    
    Args:
        data_path (str): The path to the directory containing CSV and pickle files.
        file_name (str, optional): The specific file to load. Defaults to None.
        
    Raises:
        FileNotFoundError: Raised if the specified data_path does not exist.
        FileNotFoundError: Raised if no CSV or pickle files are found in the specified directory.
        ValueError: Raised if an error occurs while reading data from a file.
        
    Returns:
        pandas.DataFrame: The concatenated DataFrame if loading all files, or the loaded DataFrame if loading a specific file.
    """
    # Check if the specified data_path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data_path '{data_path}' does not exist.")

    # Load every file if file_name is not specified
    if file_name is None:
        logger.info(f"Loading all files in {data_path}")
        # Use glob to get a list of all matching file paths for both CSV and pickle files
        data_files = glob.glob(os.path.join(data_path, '*.csv')) + glob.glob(os.path.join(data_path, '*.pkl'))

        # Check if any files were found
        if not data_files:
            raise FileNotFoundError(f"No CSV or pickle files found in {data_path}")
        
        # Initialize an empty DataFrame to store the concatenated data
        dataset = []

        # Loop through the file paths and read each file into a DataFrame
        for file_path in data_files:
            if file_path.endswith('.csv'):
                df = load_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = load_pkl(file_path)
            else:
                logger.warning(f"Unsupported file type for {file_path}")
                continue

            dataset.append(df)

        # Check if any valid dataset were loaded
        if not dataset:
            raise ValueError("No valid dataset loaded from the specified files.")

        # Concatenate all dataset into a single DataFrame
        dataset = pd.concat(dataset, ignore_index=True)

    # Load the specified file_name
    else:
        file_path = os.path.join(data_path, file_name)
        logger.info(f"Loading {file_path}")

        # Determine the file type based on the extension
        if file_path.endswith('.csv'):
            dataset = load_csv(file_path)
        elif file_path.endswith('.pkl'):
            dataset = load_pkl(file_path)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

    return dataset


def save_data(data, file_name, data_path):
    """ Takes an object, transforms in a pd dataframe
    and saves it in a pickle file in the given directory."""

    # Check if data form at is DataFrame
    if not isinstance(data, pd.DataFrame):
        # Transform data into a DataFrame
        data = pd.DataFrame(data)

    # Check if the file already exists
    if os.path.isfile(file_name):
        # If the file exists, remove it
        os.remove(data_path + file_name)

    # Save DataFrame as a pickle
    data.to_pickle(data_path + file_name)
    logger.info(f"saved {file_name} to {data_path}")