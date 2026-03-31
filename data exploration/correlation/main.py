import pandas as pd
import glob
from pathlib import Path
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.models import HoverTool, Span
import os
from statsmodels.tsa.stattools import acf, pacf
import numpy as np


def load_all_nem_data(data_folder):
    """
    Loads all NEM CSV files from the monthly folder and combines them into a single DataFrame.

    Parameters:
    data_folder (str): Path to the folder containing monthly CSV files.

    Returns:
    pd.DataFrame: Combined DataFrame with all data.
    """
    # Get all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    print(f"Found {len(csv_files)} CSV files")

    # Read and combine all files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Parse datetime
    combined_df["SETTLEMENTDATE"] = pd.to_datetime(
        combined_df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    )

    # Sort by date
    combined_df = combined_df.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    print(f"Total records: {len(combined_df)}")
    print(
        f"Date range: {combined_df['SETTLEMENTDATE'].min()} to {combined_df['SETTLEMENTDATE'].max()}"
    )

    return combined_df


def plot_rrp_and_load(df):
    """
    Creates Bokeh plots for RRP and TOTALDEMAND over time.

    Parameters:
    df (pd.DataFrame): DataFrame containing SETTLEMENTDATE, RRP, and TOTALDEMAND columns.
    """
    output_file("nem_rrp_and_load.html")

    # Create figure for RRP
    p1 = figure(
        title="Regional Reference Price (RRP) - NSW1",
        x_axis_label="Date",
        y_axis_label="RRP ($/MWh)",
        x_axis_type="datetime",
        width=1200,
        height=400,
    )

    p1.line(df["SETTLEMENTDATE"], df["RRP"], line_width=1, color="navy", alpha=0.8)

    # Add hover tool
    hover1 = HoverTool(
        tooltips=[("Date", "@x{%F %H:%M}"), ("RRP", "@y{$0.00}")],
        formatters={"@x": "datetime"},
    )
    p1.add_tools(hover1)

    # Create figure for Total Demand
    p2 = figure(
        title="Total Demand (Load) - NSW1",
        x_axis_label="Date",
        y_axis_label="Total Demand (MW)",
        x_axis_type="datetime",
        width=1200,
        height=400,
    )

    p2.line(
        df["SETTLEMENTDATE"], df["TOTALDEMAND"], line_width=1, color="green", alpha=0.8
    )

    # Add hover tool
    hover2 = HoverTool(
        tooltips=[("Date", "@x{%F %H:%M}"), ("Demand", "@y{0,0} MW")],
        formatters={"@x": "datetime"},
    )
    p2.add_tools(hover2)

    # Combine plots
    layout = column(p1, p2)

    show(layout)
    print("Plot saved to nem_rrp_and_load.html")


def remove_outliers(price_df, column="RRP", lower_quantile=0.15, upper_quantile=0.85):
    lower_bound = price_df[column].quantile(lower_quantile)
    upper_bound = price_df[column].quantile(upper_quantile)
    return price_df[
        (price_df[column] >= lower_bound) & (price_df[column] <= upper_bound)
    ]


def correlation_analysis(df, column1="RRP", column2="TOTALDEMAND"):
    correlation = df[column1].corr(df[column2])
    print(f"Correlation between {column1} and {column2}: {correlation:.4f}")
    return correlation


def autocorrelation_analysis(df, column="RRP", lags=24):
    autocorrelation = df[column].autocorr(lag=lags)
    print(f"Autocorrelation of {column} at lag {lags}: {autocorrelation:.4f}")
    return autocorrelation


def plot_acf_pacf(df, col_name="RRP", max_lag=96, output_html="acf_pacf.html"):
    """
    Creates ACF and PACF plots using Bokeh.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    col_name (str): Column name to analyze.
    max_lag (int): Maximum number of lags to display (default 96 = 48 hours for 30-min data).
    output_html (str): Output HTML filename.
    """
    # Calculate ACF and PACF
    acf_values = acf(df[col_name].dropna(), nlags=max_lag, fft=False)
    pacf_values = pacf(df[col_name].dropna(), nlags=max_lag, method="ywm")

    # Calculate confidence interval (95%)
    confidence_interval = 1.96 / np.sqrt(len(df[col_name].dropna()))

    output_file(output_html)

    # Create ACF plot
    p_acf = figure(
        title=f"Autocorrelation Function (ACF) - {col_name}",
        x_axis_label="Lag (30-min intervals)",
        y_axis_label="ACF",
        width=1200,
        height=400,
    )

    # Plot ACF bars
    lags = list(range(len(acf_values)))
    p_acf.vbar(x=lags, top=acf_values, width=0.8, color="navy", alpha=0.7)

    # Add zero line
    p_acf.line(lags, [0] * len(lags), color="black", line_width=1)

    # Create PACF plot
    p_pacf = figure(
        title=f"Partial Autocorrelation Function (PACF) - {col_name}",
        x_axis_label="Lag (30-min intervals)",
        y_axis_label="PACF",
        width=1200,
        height=400,
    )

    # Plot PACF bars
    p_pacf.vbar(x=lags, top=pacf_values, width=0.8, color="green", alpha=0.7)

    # Add zero line
    p_pacf.line(lags, [0] * len(lags), color="black", line_width=1)

    # Combine plots
    layout = column(p_acf, p_pacf)

    show(layout)
    print(f"ACF/PACF plot saved to {output_html}")


if __name__ == "__main__":
    # Load all NEM data
    data_folder = "data/monthly"
    df = load_all_nem_data(data_folder)

    # Display basic statistics
    print("\n=== Data Summary ===")
    print(df.describe())

    print("\n=== Sample Data ===")
    print(df.head())

    # Create plots
    # plot_rrp_and_load(df)

    # Remove outliers from RRP for better visualization
    cleaned_df = remove_outliers(
        df, column="RRP", lower_quantile=0.15, upper_quantile=0.85
    )
    print(f"\nRemoved outliers from RRP. Remaining records: {len(cleaned_df)}")
    # plot_rrp_and_load(cleaned_df)

    # Correlation analysis
    correlation_analysis(df, column1="RRP", column2="TOTALDEMAND")
    # 0.1742 -> weak positive correlation, but not strong enough to be a reliable predictor on its own

    # Autocorrelation analysis
    autocorrelation_analysis(df, column="RRP", lags=24)
    # 0.1565 -> weak autocorrelation at 24 hours, suggesting some daily patterns but not very strong
    autocorrelation_analysis(df, column="TOTALDEMAND", lags=24)
    # 0.6351 -> strong autocorrelation at 24 hours, indicating a strong daily pattern in demand

    # ACF and PACF plots
    print("\n=== Plotting ACF and PACF for RRP ===")
    plot_acf_pacf(df, col_name="RRP", max_lag=48, output_html="rrp_acf_pacf.html")

    print("\n=== Plotting ACF and PACF for TOTALDEMAND ===")
    plot_acf_pacf(
        df, col_name="TOTALDEMAND", max_lag=48, output_html="demand_acf_pacf.html"
    )
