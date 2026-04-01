"""
Correlation and autocorrelation analysis of NEM price and demand data.

Analyses:
- Pearson correlation between RRP and TOTALDEMAND
- ACF/PACF for daily patterns (30-min lags, up to 2 days)
- ACF/PACF for annual patterns (daily lags, up to 365 days)
- Residual ACF/PACF after removing trend and annual seasonality
"""

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import HoverTool, Label, Span
from bokeh.plotting import figure, output_file, show
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


def load_data(path):
    """
    Load merged NEM data and resample to uniform 30-min intervals
    to handle the mixed 5-min/30-min period transition (Oct 2021).
    """
    df = pd.read_csv(path)
    df["SETTLEMENTDATE"] = pd.to_datetime(
        df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    )
    df = df.set_index("SETTLEMENTDATE")[["RRP", "TOTALDEMAND"]].resample("30min").mean()
    return df


def correlation_analysis(df):
    """Pearson correlation between RRP and TOTALDEMAND."""
    corr = df["RRP"].corr(df["TOTALDEMAND"])
    if abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if corr > 0 else "negative"
    print(f"\n=== Correlation Analysis ===")
    print(f"Pearson correlation (RRP vs TOTALDEMAND): {corr:.4f}")
    print(f"Interpretation: {strength} {direction} correlation")
    return corr


def _mark_lag(p, lag, label):
    """Add a vertical dashed line and text label at a specific lag position."""
    p.add_layout(
        Span(
            location=lag,
            dimension="height",
            line_color="black",
            line_dash="dashed",
            line_width=1.5,
        )
    )
    p.add_layout(
        Label(
            x=lag + 0.5,
            y=0,
            y_units="screen",
            text=label,
            text_font_size="10px",
            text_color="black",
        )
    )


def _make_acf_plot(values, ci, title, x_label, color):
    """Helper to build a single ACF or PACF bar chart with confidence interval bands."""
    lags = list(range(len(values)))
    p = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label="Correlation",
        width=1200,
        height=350,
    )
    p.vbar(x=lags, top=values, width=0.8, color=color, alpha=0.7)
    p.line(lags, [0] * len(lags), color="black", line_width=1)
    # 95% confidence interval
    for sign in (1, -1):
        p.add_layout(
            Span(
                location=sign * ci,
                dimension="width",
                line_color="red",
                line_dash="dashed",
                line_width=1.5,
            )
        )
    return p


def plot_acf_daily(df, col_name="RRP", output_html="acf_daily.html"):
    """
    ACF/PACF for intraday patterns.
    Uses 30-min resampled data with lags up to 96 (2 days).
    A peak at lag 48 indicates a strong 24-hour cycle.
    """
    series = df[col_name].dropna()
    max_lag = 96
    acf_vals = acf(series, nlags=max_lag, fft=True)
    pacf_vals = pacf(series, nlags=max_lag, method="ywm")
    ci = 1.96 / np.sqrt(len(series))

    output_file(output_html)
    p_acf = _make_acf_plot(
        acf_vals,
        ci,
        f"ACF - {col_name}  |  Daily pattern (30-min lags, lag 48 = 24 hrs)",
        "Lag (30-min intervals)",
        "navy",
    )
    p_pacf = _make_acf_plot(
        pacf_vals,
        ci,
        f"PACF - {col_name}  |  Daily pattern (30-min lags)",
        "Lag (30-min intervals)",
        "steelblue",
    )
    for p in (p_acf, p_pacf):
        _mark_lag(p, 48, "24 hrs")
        _mark_lag(p, 96, "48 hrs")
    show(column(p_acf, p_pacf))
    print(f"Daily ACF/PACF saved to {output_html}")


def plot_acf_annual(df, col_name="RRP", output_html="acf_annual.html"):
    """
    ACF/PACF for annual/seasonal patterns.
    Resamples to daily means, lags up to 365 days.
    A peak at lag 365 indicates a strong annual cycle.
    """
    daily = df[col_name].resample("D").mean().dropna()
    max_lag = 365
    acf_vals = acf(daily, nlags=max_lag, fft=True)
    pacf_vals = pacf(daily, nlags=max_lag, method="ywm")
    ci = 1.96 / np.sqrt(len(daily))

    output_file(output_html)
    p_acf = _make_acf_plot(
        acf_vals,
        ci,
        f"ACF - {col_name}  |  Annual pattern (daily lags, lag 365 = 1 year)",
        "Lag (days)",
        "navy",
    )
    p_pacf = _make_acf_plot(
        pacf_vals,
        ci,
        f"PACF - {col_name}  |  Annual pattern (daily lags)",
        "Lag (days)",
        "steelblue",
    )
    for p in (p_acf, p_pacf):
        _mark_lag(p, 365, "1 year")
    show(column(p_acf, p_pacf))
    print(f"Annual ACF/PACF saved to {output_html}")


if __name__ == "__main__":
    df = load_data("data_exploration/data/merged_nem_data_outlier_removed.csv")

    print("\n=== Data Summary ===")
    print(df.describe())

    correlation_analysis(df)

    for col in ["RRP", "TOTALDEMAND"]:
        print(f"\n=== Daily ACF/PACF for {col} ===")
        plot_acf_daily(
            df,
            col_name=col,
            output_html=f"data_exploration/plots/{col.lower()}_acf_daily.html",
        )

        print(f"\n=== Annual ACF/PACF for {col} ===")
        plot_acf_annual(
            df,
            col_name=col,
            output_html=f"data_exploration/plots/{col.lower()}_acf_annual.html",
        )
