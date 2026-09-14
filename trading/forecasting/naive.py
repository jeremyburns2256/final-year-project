"""
naive.py

Perfect-foresight and seasonal-naive forecasters (FORECAST_NOTES: baselines).
"""

from __future__ import annotations

import numpy as np

from forecasting.base import HORIZON_SLOTS, SLOTS_PER_DAY, Forecaster, Frame, HalfHourlyPriceMixin


class PerfectForecaster(Forecaster):
    """Actual prices and actual net-local: perfect foresight on the MPC loop."""

    name = "perfect"

    def price(self, t, horizon):
        return self.frame.rrp[t : t + horizon].copy()

    def net_local(self, t, horizon):
        return self.frame.net_local[t : t + horizon].copy()


class SeasonalNaiveForecaster(HalfHourlyPriceMixin, Forecaster):
    """
    Price: the mean RRP of the same half-hour slot on the previous day.
    Net-local: 7-day time-of-day profile.
    """

    name = "naive"

    def __init__(self, frame: Frame):
        super().__init__(frame)
        hh = frame.rrp_half_hourly()
        n = frame.n_slots
        fc = np.full((n, HORIZON_SLOTS), np.nan)
        for k in range(SLOTS_PER_DAY, n):
            src = k - SLOTS_PER_DAY + np.arange(HORIZON_SLOTS)   # slots k-48 .. k-1, all observed before k
            fc[k] = hh[src]
        self.hh_forecasts = fc

    def net_local(self, t, horizon):
        return self.seven_day_profile(t, horizon)
