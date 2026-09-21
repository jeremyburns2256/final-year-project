"""
A Python module for generating basic plots uses Bokeh
"""

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import HoverTool, DatetimeTickFormatter, DatetimeTicker


def plot_rrp_and_load(source_path, destination_path):
    """
    Creates Bokeh plots for RRP and TOTALDEMAND over time with 30-day trend lines.

    Parameters:
    source_path (str): Path to the merged NEM data CSV file.
    destination_path (str): Path to save the output HTML file.
    """
    df = pd.read_csv(source_path)
    df["SETTLEMENTDATE"] = pd.to_datetime(
        df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    )
    print(df.head())
    output_file(destination_path)

    # Resample to daily mean first to avoid 5-min vs 30-min interval bias,
    # then compute 30-day rolling mean as the trend line
    daily = df.set_index("SETTLEMENTDATE")[["RRP", "TOTALDEMAND"]].resample("D").mean()
    trend = daily.rolling(window=30, center=True, min_periods=1).mean()

    # Create figure for RRP
    p1 = figure(
        title="Regional Reference Price (RRP) - NSW",
        x_axis_label="Date",
        y_axis_label="RRP ($/MWh)",
        x_axis_type="datetime",
        width=1200,
        height=400,
    )
    p1.line(df["SETTLEMENTDATE"], df["RRP"], line_width=1, color="navy", alpha=0.4)
    p1.line(
        trend.index,
        trend["RRP"],
        line_width=2,
        color="red",
        legend_label="30-day Rolling Average",
    )
    p1.legend.location = "top_right"
    p1.xaxis.ticker = DatetimeTicker(desired_num_ticks=20)
    p1.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M %d %b %Y",
        days="%d %b %Y",
        months="%b %Y",
        years="%Y",
    )
    p1.add_tools(
        HoverTool(
            tooltips=[("Date", "@x{%F %H:%M}"), ("RRP", "@y{$0.00}")],
            formatters={"@x": "datetime"},
        )
    )

    # Create figure for Total Demand (linked x-axis)
    p2 = figure(
        title="Total Demand (Load) - NSW",
        x_axis_label="Date",
        y_axis_label="Total Demand (MW)",
        x_axis_type="datetime",
        width=1200,
        height=400,
        x_range=p1.x_range,
    )
    p2.line(
        df["SETTLEMENTDATE"], df["TOTALDEMAND"], line_width=1, color="green", alpha=0.4
    )
    p2.line(
        trend.index,
        trend["TOTALDEMAND"],
        line_width=2,
        color="red",
        legend_label="30-day Rolling Average",
    )
    p2.legend.location = "top_right"
    p2.xaxis.ticker = DatetimeTicker(desired_num_ticks=20)
    p2.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M %d %b %Y",
        days="%d %b %Y",
        months="%b %Y",
        years="%Y",
    )
    p2.add_tools(
        HoverTool(
            tooltips=[("Date", "@x{%F %H:%M}"), ("Demand", "@y{0,0} MW")],
            formatters={"@x": "datetime"},
        )
    )

    show(column(p1, p2))
    print(f"Plot saved to {destination_path}")


if __name__ == "__main__":
    source_path = "data_exploration/data/merged_nem_data_outlier_removed.csv"
    destination_path = "data_exploration/plots/nem_rrp_and_load_outlier_removed.html"
    # source_path = "data_exploration/data/merged_nem_data.csv"
    # destination_path = "data_exploration/plots/nem_rrp_and_load.html"
    plot_rrp_and_load(source_path, destination_path)
