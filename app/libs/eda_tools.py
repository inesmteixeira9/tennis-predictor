import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from app.libs.monitoring import log, LogLevel

def check_missing_data(df: pd.DataFrame):
    """Check for missing data in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Raises:
        WARNING log: Prints the percentage of missing data for each column.
    """
    try:
        if df.isnull().any().any():
            print('Missing Data Percentage List')
            for col in df.columns:
                pct_missing = np.mean(df[col].isnull())
                if pct_missing > 0:
                    print('{} - {}%'.format(col, round(pct_missing * 100, 2)))
        else:
            print('No missing values.')
    except Exception as e:
        log.print_log(LogLevel.WARNING,f"An error occurred while checking for missing data: {e}")

def visualize_pd_series(df: pd.DataFrame, features: list, skew_threshold: float = 4, percentile_threshold: float = 98):
    """Visualizes the features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features (list): List of feature names to visualize.
        skew_threshold (float): Threshold for skewness to apply percentile filtering.
        percentile_threshold (float): Percentile threshold for filtering outliers.

    Raises:
        WARNING log: In case of issues during visualization.
    """
    try:
        num_features = len(features)
        cols = 3  # Number of columns for subplots
        rows = (num_features + cols - 1) // cols  # Calculate rows needed

        fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 5))

        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        for i, col in enumerate(features):
            if col.startswith('H2H_'):
                df_ = df[df[col[len('H2H_'):]] == 1]
            else:
                df_ = df

            unique_values = df_[col].nunique()
            if unique_values < 30:
                # Categorical column: use a count plot
                sns.countplot(x=col, data=df_, palette='Set2', hue=col, ax=axs[i], legend=False)
            else:
                # Continuous column: check skewness and visualize
                col_skew = skew(df_[col].dropna())
                if abs(col_skew) > skew_threshold:
                    threshold = np.percentile(df_[col].dropna(), percentile_threshold)
                    data_to_plot = df_[df_[col] < threshold][col]
                    if len(df_[col].unique()) < 10:
                        data_to_plot = df_[col]
                else:
                    data_to_plot = df_[col]

                sns.histplot(data_to_plot, bins=round(10 + len(df_[col].unique())**0.5), kde=True, color='skyblue', alpha=0.6, ax=axs[i])

            axs[i].set_xlabel(col.replace('_', ' '))
            axs[i].set_title(f"Distribution of {col.replace('_', ' ')}")

        # If there are any unused subplots, turn them off
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
    except Exception as e:
        log.print_log(LogLevel.WARNING,f"An error occurred during visualization: {e}")

def scatter(df: pd.DataFrame, x: str, y: str, func: str = None, title: str = None, xlabel: str = None, ylabel: str = None, sqrt: float = 2):
    """Creates scatter plots with optional regression or square root fitting.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x (str): Name of the feature to plot on the x-axis.
        y (str): Name of the feature to plot on the y-axis.
        func (str, optional): Function to apply ('linear_reg' or 'sqrt'). Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        sqrt (float, optional): Exponent for square root fitting. Defaults to 2.

    Raises:
        WARNING log: In case of issues during plotting.
    """
    try:
        if title is None:
            title = f'{y} vs. {x}'
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        if isinstance(x, list) and isinstance(y, list):
            x = pd.concat(x)
            y = pd.concat(y)
        else:
            x = df[x]
            y = df[y]

        plt.figure(figsize=(12, 4))
        plt.title(title)

        if func == 'linear_reg':
            sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.3, 'marker': '.', 's': 10},
                        line_kws={'color': 'darkblue', 'linestyle': '--'})
            plt.ylim(bottom=y.min(), top=y.max())
        elif func == 'sqrt':
            plt.scatter(x, y, marker='.')
            start_point = (x.min(), y.min())
            end_point = (x.max(), y.max())
            x_start, y_start = start_point
            x_end, y_end = end_point
            x_values = np.linspace(x_start, x_end, 100)
            a = (y_end - y_start) / (x_end - x_start) ** (1 / sqrt)
            y_values = a * (x_values - x_start) ** (1 / sqrt) + y_start
            plt.plot(x_values, y_values, color='blue', linestyle='--')
        else:
            plt.scatter(x, y, marker='.')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
    except Exception as e:
        log.print_log(LogLevel.WARNING,f"An error occurred during scatter plot creation: {e}")
