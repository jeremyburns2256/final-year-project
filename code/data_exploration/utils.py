"""
Utility functions for data exploration.
"""

import os
import pandas as pd


def merge_nem_data(directory_source, directory_destination):
    """
    Merge all NEM data files in the specified directory into a single DataFrame and saves as csv.
    """

    all_files = [
        os.path.join(directory_source, f)
        for f in os.listdir(directory_source)
        if f.endswith(".csv")
    ]
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    # merged_df.to_csv(
    #     os.path.join(directory_destination, "merged_nem_data.csv"), index=False
    # )

    return merged_df


def remove_outliers(price_df, column="RRP", lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = price_df[column].quantile(lower_quantile)
    upper_bound = price_df[column].quantile(upper_quantile)
    return price_df[
        (price_df[column] >= lower_bound) & (price_df[column] <= upper_bound)
    ]


if __name__ == "__main__":
    df = merge_nem_data("data_exploration/data/monthly_NEM", "data_exploration/data")
    outlier_free_df = remove_outliers(df)
    print(outlier_free_df.head())
    outlier_free_df.to_csv(
        "data_exploration/data/merged_nem_data_outlier_removed.csv", index=False
    )

    # source_dir = "data_exploration/data/monthly_NEM"
    # destination_dir = "data_exploration/data"
    # merged_data = merge_nem_data(source_dir, destination_dir)
    # print(
    #     f"Merged data saved to {os.path.join(destination_dir, 'merged_nem_data.csv')}"
    # )
