"""
load_forecast_plot.py

LSTM net-local forecast against the meter on the test month (FORECAST_NOTES: load model).

    python -m plotting.load_forecast_plot        # from trading/, writes plots/load_forecast.html

Panels: the half-hourly actual with the forecast made 30 min ahead and 24 h ahead
(shared time axis, first week in view, pan for the rest), then MAE by lead time.
The 7-day profile is drawn on every panel as the forecast the LSTM replaces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, Range1d

from forecasting.base import HORIZON_SLOTS, INTERVALS_PER_HOUR, SLOTS_PER_DAY, Frame, load_frame
from forecasting.load_lstm import LstmLoadForecaster
from forecasting.naive import PerfectPriceForecaster
from plotting.theme import AQUA, INK, ORANGE, heading, line_hover, make_figure, note, save_page, section, stat_tiles, style, zero_line

ACTUAL, LSTM, PROFILE = "Actual (meter)", "LSTM", "7-day profile"
COLOUR = {ACTUAL: INK, LSTM: ORANGE, PROFILE: AQUA}
PROFILE_DAYS = 7


def _series_panel(src: ColumnDataSource, field: str, title: str, x_range) -> object:
    p = make_figure(height=300, title=title, x_axis_type="datetime", x_range=x_range)
    zero_line(p)
    r = p.line("time", "actual", source=src, color=COLOUR[ACTUAL], line_width=1.5, legend_label=ACTUAL)
    p.line("time", "profile", source=src, color=COLOUR[PROFILE], line_width=2, legend_label=PROFILE)
    p.line("time", field, source=src, color=COLOUR[LSTM], line_width=2, legend_label=LSTM)
    p.yaxis.axis_label = "Net-local (kW, export positive)"
    line_hover(p, [r], [("", "@time{%d %b %H:%M}"), (ACTUAL, "@actual{0.00} kW"), (LSTM, f"@{field}{{0.00}} kW"),
                        (PROFILE, "@profile{0.00} kW")], formatters={"@time": "datetime"})
    p.legend.orientation = "horizontal"
    p.legend.location = "top_left"
    return style(p)


def plot_load_forecast(frame: Frame, output_path: str = "plots/load_forecast.html", horizon: int = 288) -> None:
    lstm, profile = LstmLoadForecaster(frame), PerfectPriceForecaster(frame)
    hh = frame.net_local_half_hourly()
    first, n = int(frame.slot_of[frame.test_start]), frame.n_slots
    s = np.arange(first + HORIZON_SLOTS - 1, n)                # target slots that have a 24 h-ahead forecast in the test month
    days = SLOTS_PER_DAY * np.arange(1, PROFILE_DAYS + 1)
    time = frame.slot_start_times()[s]
    src = ColumnDataSource({
        "time": time, "actual": hh[s], "profile": hh[s[:, None] - days[None, :]].mean(axis=1),
        "lstm_30min": lstm.hh_net_forecasts[s, 0], "lstm_24h": lstm.hh_net_forecasts[s - (HORIZON_SLOTS - 1), HORIZON_SLOTS - 1],
    })
    x_range = Range1d(time[0], time[0] + pd.Timedelta(days=7), bounds=(time[0], time[-1]))
    near = _series_panel(src, "lstm_30min", "Forecast for the next half hour", x_range)
    far = _series_panel(src, "lstm_24h", "Forecast made 24 hours ahead", x_range)

    # Error by lead at 5-min resolution, forecasts issued every half hour (as forecast_trading.lead_time_errors).
    issues = range(frame.test_start, frame.n - horizon + 1, 6)
    actual = np.stack([frame.net_local[t : t + horizon] for t in issues])
    err = {name: np.abs(np.stack([fc.net_local(t, horizon) for t in issues]) - actual) for name, fc in ((LSTM, lstm), (PROFILE, profile))}
    lead = ColumnDataSource({"hours": (np.arange(horizon) + 1) / INTERVALS_PER_HOUR,
                             "lstm": err[LSTM].mean(axis=0), "profile": err[PROFILE].mean(axis=0)})
    q = make_figure(height=280, title="Mean absolute error by lead time", x_range=(0, 24), y_range=(0, 1.4))
    q.line("hours", "profile", source=lead, color=COLOUR[PROFILE], line_width=2, legend_label=PROFILE)
    r = q.line("hours", "lstm", source=lead, color=COLOUR[LSTM], line_width=2, legend_label=LSTM)
    q.xaxis.axis_label = "Lead time (hours ahead)"
    q.xaxis.ticker = list(range(0, 25, 3))
    q.yaxis.axis_label = "MAE (kW)"
    line_hover(q, [r], [("Lead", "@hours{0.0} h"), (LSTM, "@lstm{0.000} kW"), (PROFILE, "@profile{0.000} kW")])
    q.legend.orientation = "horizontal"
    q.legend.location = "bottom_right"
    style(q)

    mae = {k: float(v.mean()) for k, v in err.items()}
    hour = {k: float(v[:, :INTERVALS_PER_HOUR].mean()) for k, v in err.items()}
    tiles = stat_tiles([
        {"label": "LSTM MAE", "value": f"{mae[LSTM]:.2f} kW", "sub": "all leads, 5-min resolution"},
        {"label": "7-day profile MAE", "value": f"{mae[PROFILE]:.2f} kW", "sub": "all leads, 5-min resolution"},
        {"label": "LSTM against profile", "value": f"{100 * (mae[LSTM] / mae[PROFILE] - 1):+.0f}%", "sub": "change in MAE",
         "tone": "good" if mae[LSTM] < mae[PROFILE] else "bad"},
        {"label": "First hour", "value": f"{hour[LSTM]:.2f} kW", "sub": f"profile {hour[PROFILE]:.2f} kW"},
    ])
    start, end = frame.start_times[frame.test_start], frame.start_times[-1]
    save_page([
        heading("Net-local load: LSTM forecast against the meter",
                f"{start:%-d %b %Y} to {end:%-d %b %Y}. Two stacked 20-unit LSTM layers after Kong et al. (2019), issuing a 48-slot forecast "
                "every half hour from the previous day of load. Net-local is export minus import: positive when the house sends solar to the grid."),
        tiles,
        section("Forecast against actual", "Half-hourly means. The first week is in view; drag to pan through the month. "
                "Both panels share a time axis. Click a legend entry to hide that series."),
        near, far,
        section("Error by lead time"),
        q,
        note("MAE is measured at 5-min resolution on 24 h forecasts issued every half hour, as the MPC sees them; "
             "the half-hourly forecast is held for six 5-min steps."),
    ], title="Load forecast JAN25", output_path=output_path)


if __name__ == "__main__":
    plot_load_forecast(load_frame())
