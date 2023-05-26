import numpy as np
import pandas as pd
from enum import Enum
import seaborn as sns


class Frequency(Enum):
    DAILY = '1440'
    WEEKLY = '10080'
    MONTHLY = '43200'


def filter_dataframe(df: pd.DataFrame):
    threshold_of_nan = df.shape[0] * 0.4
    filtered_df = df.loc[:, df.isnull().sum() <= threshold_of_nan]
    return filtered_df


def pivot_dataframe(df: pd.DataFrame, freq: Frequency):
    if freq == Frequency.DAILY:
        df['timestamp_freq'] = df["timestamp"].astype("datetime64[D]")
        df['timestamp_freq'] = df['timestamp_freq'].dt.strftime('%Y-%m-%d')
    elif freq == Frequency.WEEKLY:
        df['timestamp_freq'] = df["timestamp"].astype("datetime64[W]")
        df['timestamp_freq'] = df['timestamp_freq'].dt.strftime('%Y-%w')
    elif freq == Frequency.MONTHLY:
        # consolidates the date to first day of the month for easier plotting
        df['timestamp_freq'] = df["timestamp"].astype("datetime64[M]")
        # converts the timestamp to a datetime object with showing only year and month
        df['timestamp_freq'] = df['timestamp_freq'].dt.strftime('%Y-%m')
    pivoted_df = df.pivot_table(index=["timestamp_freq"], columns="symbol", values="open")
    return pivoted_df


def sort_best_correlations(arr: np.array):
    sorted_arr = np.sort(arr, order='corr')[::-1]
    rolled_sorted_arr = np.roll(sorted_arr, -np.count_nonzero(np.isnan(arr['corr'])))
    return rolled_sorted_arr


def preprocess(df: pd.DataFrame, freq: Frequency):
    pivoted_df = pivot_dataframe(df, freq)
    corr = pivoted_df.corr()

    print(corr)
    # Sort the columns and rows of the correlation matrix to show clusters
    clustered_corr = sns.clustermap(corr, cmap="YlGnBu", row_cluster=True, col_cluster=True)

    # Get the sorted correlation matrix
    sorted_corr = corr.iloc[
        clustered_corr.dendrogram_row.reordered_ind,
        clustered_corr.dendrogram_col.reordered_ind
    ]
    return sorted_corr

