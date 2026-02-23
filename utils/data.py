"""
Module used for loading and normalising data
"""

import pandas as pd

def remove_outliers(price_df, column="RRP", lower_quantile=0.15, upper_quantile=0.85):
    lower_bound = price_df[column].quantile(lower_quantile)
    upper_bound = price_df[column].quantile(upper_quantile)
    return price_df[(price_df[column] >= lower_bound) & (price_df[column] <= upper_bound)]