"""
Analysis of state machine simulation results.
Plots export revenue generated vs RRP percentile bucket.
"""

import numpy as np
import pandas as pd
from bokeh.models import HoverTool, Label
from bokeh.plotting import figure, output_file, show


def load_results(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    # Per-interval export revenue ($): kWh exported * $/MWh / 1000
    df["export_revenue"] = df["grid_export_kwh"] * df["rrp"] / 1000
    return df


def plot_revenue_by_rrp_percentile(
    df, top_pct=30, output_html="revenue_by_rrp_percentile.html"
):
    """
    Bins intervals into 1-percentile buckets and plots total export
    revenue for the top N percentiles of RRP.

    Parameters:
    df (pd.DataFrame): Simulation results with rrp and export_revenue columns.
    top_pct (int): How many top percentile buckets to show (default 30 = top 30%).
    output_html (str): Output HTML file path.
    """
    # Compute all 101 quantile edges at 1% increments across the full dataset
    all_edges = df["rrp"].quantile(np.linspace(0, 1, 101)).values

    # Slice to only the top N% edges and deduplicate
    start_pct = 100 - top_pct  # e.g. 70 for top 30%
    top_edges = np.unique(all_edges[start_pct:])

    # Generate labels to match however many bins survive after deduplication
    n_actual = len(top_edges) - 1
    labels = [f"{start_pct + i}-{start_pct + i + 1}%" for i in range(n_actual)]

    df["bin"] = pd.cut(df["rrp"], bins=top_edges, labels=labels, include_lowest=True)

    buckets = (
        df.groupby("bin", observed=True)
        .agg(
            total_revenue=("export_revenue", "sum"),
            rrp_min=("rrp", "min"),
            rrp_max=("rrp", "max"),
            rrp_mean=("rrp", "mean"),
            interval_count=("rrp", "count"),
            export_intervals=("grid_export_kwh", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .rename(columns={"bin": "bucket_label"})
    )
    buckets["bucket_label"] = buckets["bucket_label"].astype(str)

    output_file(output_html)

    p = figure(
        title="Export Revenue by RRP Percentile",
        x_range=buckets["bucket_label"].tolist(),
        x_axis_label="RRP Percentile Bucket",
        y_axis_label="Total Export Revenue ($)",
        width=1000,
        height=500,
    )

    p.vbar(
        x=buckets["bucket_label"],
        top=buckets["total_revenue"],
        width=0.7,
        color="steelblue",
        alpha=0.8,
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Percentile", "@x"),
                ("Revenue", "@top{$0.00}"),
                ("RRP range", "@rrp_min{$0.00} – @rrp_max{$0.00}"),
                ("Mean RRP", "@rrp_mean{$0.00}"),
                ("Intervals with export", "@export_intervals / @interval_count"),
            ]
        )
    )

    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    show(p)
    print(f"Plot saved to {output_html}")

    # Print summary table
    print("\n=== Revenue by RRP Percentile ===")
    print(
        buckets[
            [
                "bucket_label",
                "rrp_min",
                "rrp_max",
                "rrp_mean",
                "total_revenue",
                "export_intervals",
            ]
        ].to_string(index=False)
    )


def plot_revenue_99th_percentile(df, output_html="revenue_99th_percentile.html"):
    """
    Breaks the 99th-100th percentile of RRP into 0.1% increments and plots
    total export revenue within each sub-bucket.
    """
    # 11 edges from 99.0% to 100.0% in 0.1% steps → 10 bins
    edges = df["rrp"].quantile(np.linspace(0.99, 1.0, 11)).values
    edges = np.unique(edges)

    n_actual = len(edges) - 1
    labels = [
        f"{99.0 + i * 0.1:.1f}-{99.0 + (i + 1) * 0.1:.1f}%" for i in range(n_actual)
    ]

    df["bin"] = pd.cut(df["rrp"], bins=edges, labels=labels, include_lowest=True)

    buckets = (
        df.groupby("bin", observed=True)
        .agg(
            total_revenue=("export_revenue", "sum"),
            rrp_min=("rrp", "min"),
            rrp_max=("rrp", "max"),
            rrp_mean=("rrp", "mean"),
            interval_count=("rrp", "count"),
            export_intervals=("grid_export_kwh", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .rename(columns={"bin": "bucket_label"})
    )
    buckets["bucket_label"] = buckets.apply(
        lambda r: f"{r['bucket_label']}\n${r['rrp_min']:.0f}-${r['rrp_max']:.0f}", axis=1
    )

    output_file(output_html)

    p = figure(
        title="Export Revenue — 99th Percentile Breakdown (0.1% increments)",
        x_range=buckets["bucket_label"].tolist(),
        x_axis_label="RRP Percentile Bucket",
        y_axis_label="Total Export Revenue ($)",
        width=1000,
        height=500,
    )
    p.vbar(
        x=buckets["bucket_label"],
        top=buckets["total_revenue"],
        width=0.7,
        color="darkorange",
        alpha=0.8,
    )
    p.add_tools(
        HoverTool(
            tooltips=[
                ("Percentile", "@x"),
                ("Revenue", "@top{$0.00}"),
                ("RRP range", "@rrp_min{$0.00} – @rrp_max{$0.00}"),
                ("Mean RRP", "@rrp_mean{$0.00}"),
                ("Intervals with export", "@export_intervals / @interval_count"),
            ]
        )
    )
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    show(p)
    print(f"Plot saved to {output_html}")

    print("\n=== Revenue — 99th Percentile Breakdown ===")
    print(
        buckets[
            [
                "bucket_label",
                "rrp_min",
                "rrp_max",
                "rrp_mean",
                "total_revenue",
                "export_intervals",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    df = load_results("data_exploration/state_machine/sample_state_machine_results.csv")
    plot_revenue_by_rrp_percentile(
        df,
        top_pct=20,
        output_html="data_exploration/plots/revenue_by_rrp_percentile.html",
    )
    plot_revenue_99th_percentile(
        df,
        output_html="data_exploration/plots/revenue_99th_percentile.html",
    )
