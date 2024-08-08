import os
import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from app.libs.monitoring import log, LogLevel
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from app.config import APP_NAME


def upload_csvfile(csv_file, local_path: str) -> pd.DataFrame:
    """Get a CSV file and store it in a local folder.

    Args:
        csv_file: CSV file object to be uploaded.
        local_path (str): Path where the CSV file should be stored.

    Raises:
        ERROR log: If an error occurs while reading the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the uploaded data.
    """
    try:
        try:
            raw_data = pd.read_csv(csv_file.file, sep=None, engine='python')
        except:
            raw_data = pd.read_csv(csv_file, sep=None, engine='python')

        # Remove empty spaces from strings in the raw_data
        raw_data = raw_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Save pd DataFrame in local path
        raw_data.to_csv(local_path)

        return raw_data

    except Exception as e:
        log.print_log(LogLevel.ERROR, f'Error reading raw_data as df using read_raw_data: {e}')


def read_data(data_path: str, file_name: str = None) -> pd.DataFrame:
    """
    Read data from CSV or pickle files in the specified directory or read a specific file.

    Args:
        data_path (str): The path to the directory containing CSV and pickle files.
        file_name (str, optional): The specific file to read. Defaults to None.

    Raises:
        FileNotFoundError: Raised if the specified data_path does not exist.
        FileNotFoundError: Raised if no CSV or pickle files are found in the specified directory.
        ERROR log: If an unsupported file type is encountered.

    Returns:
        pd.DataFrame: The concatenated DataFrame if reading all files, or the read DataFrame if reading a specific file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data_path '{data_path[data_path.rfind(APP_NAME):]}' does not exist.")

    if data_path.endswith('.csv'):
        file_name = data_path.split('/')[-1]

    if file_name is None:
        log.print_log(LogLevel.INFO, f"Reading all files in {data_path[data_path.rfind(APP_NAME):]}...")
        data_files = glob.glob(os.path.join(data_path, '*.csv')) + glob.glob(os.path.join(data_path, '*.pkl'))

        if not data_files:
            raise FileNotFoundError(f"No CSV or pickle files found in {data_path}")

        dataset = []

        for file_path in data_files:
            if file_path.endswith('.csv'):
                df = read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = read_pkl(file_path)
            else:
                log.print_log(LogLevel.ERROR, f"Unsupported file type for {file_path}")
                continue

            dataset.append(df)

        dataset = pd.concat(dataset, ignore_index=True)

    else:
        file_path = os.path.join(data_path, file_name)
        log.print_log(LogLevel.INFO, f"Reading {file_path}...")

        if file_path.endswith('.csv'):
            dataset = read_csv(file_path)
        elif file_path.endswith('.pkl'):
            dataset = read_pkl(file_path)
        else:
            log.print_log(LogLevel.ERROR, f"Unsupported file type for {file_path}")
            return pd.DataFrame()

    log.print_log(LogLevel.INFO, f"Shape of the dataset: {dataset.shape}")

    return dataset


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Raises:
        ERROR log: If an error occurs while reading empty data from the CSV file.
        ERROR log: If an error occurs while reading the CSV file.

    Returns:
        pd.DataFrame: The DataFrame read from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.EmptyDataError as empty_data_error:
        log.print_log(LogLevel.ERROR, f"Error reading empty data from file {file_path}: {empty_data_error}")
    except Exception as e:
        log.print_log(LogLevel.ERROR, f"Error reading CSV file {file_path}: {str(e)}")

    return pd.DataFrame()


def read_pkl(file_path: str) -> pd.DataFrame:
    """
    Read data from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Raises:
        ERROR log: If an error occurs while unpickling the file.
        ERROR log: If an error occurs while reading empty data from the pickled file.

    Returns:
        pd.DataFrame: The DataFrame read from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            df = pickle.load(file)
            return df
    except pickle.PickleError as unpickling_error:
        log.print_log(LogLevel.ERROR, f"Error unpickling file {file_path}: {unpickling_error}")
    except Exception as e:
        log.print_log(LogLevel.ERROR, f"Error reading pickle file {file_path}: {str(e)}")

    return pd.DataFrame()


def format_data(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Convert DataFrame columns according to a specified schema.

    Args:
        df (pd.DataFrame): DataFrame to be formatted.
        schema (dict): Dictionary defining the desired data types for the columns.

    Raises:
        WARNING log: If columns with object type were not formatted.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """
    try:
        df = df.astype('string')

        cols_to_format = {col: dtype for col, dtype in schema.items() if col in df.columns}
        for col, dtype in cols_to_format.items():
            if 'datetime' in dtype:
                # Determine the date format
                if df[col].iloc[0][4] == '-':
                    format = "%Y-%m-%d"
                elif df[col].iloc[0][2] == '-':
                    format = "%d-%m-%Y"
                elif df[col].iloc[0][4] == '/':
                    format = "%Y/%m/%d"
                elif df[col].iloc[0][2] == '/':
                    format = "%d/%m/%Y"

                # Convert datetime columns
                df[col] = pd.to_datetime(df[col], format=format)

            elif 'int' in dtype:
                df[col] = df[col].str.replace('.0', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

            else:
                # Convert other columns
                df[col] = df[col].astype(dtype)

        # Check for columns with object type and give warnings
        object_cols = df.select_dtypes(include=['object']).columns
        if not object_cols.empty:
            log.print_log(LogLevel.WARNING, f"The following columns are of type object and were not formatted: {', '.join(object_cols)}")
        else:
            log.print_log(LogLevel.INFO, 'Data formatted according to the imported schema...')

        return df
    except Exception as e:
        log.print_log(LogLevel.ERROR, f"Error formatting data: {e}")
        return df


def split_data(X, y, train_size, test_size, cv=False):
    """Split data into training, testing, and optionally validation sets.

    Args:
        X (array-like): Feature data.
        y (array-like): Labels.
        train_size (float): Proportion of the data to be used for training (between 0 and 1).
        test_size (float): Proportion of the data to be used for testing (between 0 and 1).
        cv (bool): If True, split data into training and testing sets only. 
                   If False, split data into training, validation, and testing sets.

    Raises:
        ValueError: If train_size + test_size > 1 or if train_size or test_size are not in (0,1).

    Returns:
        tuple: If cv is True, returns (X_train, X_test, y_train, y_test).
               If cv is False, returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if cv == True: 
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        log.print_log(LogLevel.INFO, f"Splitting data randomly into a training ({train_size * 100}%) and a testing ({test_size * 100}%) datasets.")
        return X_train, X_test, y_train, y_test
    else:
        # define validation size
        temp_size = 1 - train_size
        test_val_ratio = test_size / temp_size
        
        # Split data into training, validation and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=temp_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_val_ratio, random_state=42)
        log.print_log(LogLevel.INFO, f"Splitting data randomly into a training ({train_size * 100}%), a validation ({round(1 - train_size - test_size, 2) * 100}%) and a testing ({test_size * 100}%) datasets.")
        return X_train, X_val, X_test, y_train, y_val, y_test



def scaler_data(X_train, X_val, X_test):
    # Initialize the scaler and fit it to the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler to transform val and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def save_data(data: pd.DataFrame, file_name: str, path: str):
    """Save an object as a DataFrame in a pickle or CSV file in the given directory.

    Args:
        data (pd.DataFrame): DataFrame to be saved.
        file_name (str): Must end with '.pkl' or '.csv'.
        path (str): Directory where the file should be saved.

    Raises:
        ERROR log: If an unrecognized file type is provided.

    Returns:
        None
    """
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        file_path = os.path.join(path, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)

        if file_name.endswith('.csv'):
            data.to_csv(file_path)
        elif file_name.endswith('.pkl'):
            data.to_pickle(file_path)
        else:
            log.print_log(LogLevel.ERROR, f"Unrecognized file type for {file_name}.")
            return

        log.print_log(LogLevel.INFO, f"Saved {file_name} to {path}")
    except Exception as e:
        log.print_log(LogLevel.ERROR, f"Error saving data: {e}")
