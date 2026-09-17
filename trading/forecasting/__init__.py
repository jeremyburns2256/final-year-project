"""
State forecasting for the household BESS MILP (thesis Sections 4.1, 5.1, 6.4).

Design decisions are recorded in trading/FORECAST_NOTES.md.

    Forecaster            interface: price(t, horizon), net_local(t, horizon)
    PerfectForecaster     actuals (perfect foresight re-run on the MPC loop)
    PerfectPriceForecaster  actual price, 7-day net-local profile (isolates the price forecast)
    SeasonalNaiveForecaster  same 30-min slot previous day; 7-day net-local profile
    AemoPredispatchForecaster  latest AEMO PREDISPATCHPRICE run before t
    LstmPriceForecaster   LSTM on asinh(RRP/100), 48 half-hour outputs
"""

from forecasting.base import Forecaster, Frame, load_frame
from forecasting.naive import PerfectForecaster, PerfectPriceForecaster, SeasonalNaiveForecaster

__all__ = ["Forecaster", "Frame", "load_frame", "PerfectForecaster", "PerfectPriceForecaster", "SeasonalNaiveForecaster"]
