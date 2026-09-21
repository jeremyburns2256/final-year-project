"""
base.py

Forecaster interface and the shared data frame used by the MPC loop.

Conventions
-----------
* Intervals are 5 minutes. Index t refers to the interval whose *start* is
  frame.start_times[t]; SETTLEMENTDATE in the CSVs is the interval end.
* A forecaster returns arrays for intervals t, t+1, ..., t+horizon-1. The MPC
  loop overwrites element 0 of the price forecast with the actual dispatch
  price (FORECAST_NOTES: step-0 actual). Net-local element 0 stays forecast.
* Price forecasts are produced at 30-min resolution and held for six 5-min
  steps; `HalfHourlyPriceMixin` does the expansion so that naive, AEMO and LSTM
  forecasters only have to supply a (n_slots, 48) matrix of half-hourly
  forecasts, row k issued at the start of half-hour slot k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.data import merge_optional_csv

INTERVALS_PER_HOUR = 12
INTERVALS_PER_SLOT = 6          # 5-min intervals per 30-min slot
INTERVALS_PER_DAY = 288
SLOTS_PER_DAY = 48
HORIZON_SLOTS = 48              # 24 h at 30 min


@dataclass
class Frame:
    """History + test data on one 5-min timeline. Test intervals are [test_start, n)."""

    start_times: pd.DatetimeIndex   # interval start, length n
    rrp: np.ndarray                 # $/MWh
    demand: np.ndarray              # MW (NSW TOTALDEMAND)
    export_kw: np.ndarray
    import_kw: np.ndarray
    test_start: int

    @property
    def n(self) -> int:
        return len(self.rrp)

    @property
    def net_local(self) -> np.ndarray:
        return self.export_kw - self.import_kw

    @property
    def slot_of(self) -> np.ndarray:
        """Half-hour slot index of each interval, counted from the frame start."""
        return np.arange(self.n) // INTERVALS_PER_SLOT

    @property
    def offset_of(self) -> np.ndarray:
        """Position (0..5) of each interval within its half-hour slot."""
        return np.arange(self.n) % INTERVALS_PER_SLOT

    @property
    def n_slots(self) -> int:
        return int(np.ceil(self.n / INTERVALS_PER_SLOT))

    def slot_start_times(self) -> pd.DatetimeIndex:
        return self.start_times[::INTERVALS_PER_SLOT]

    def rrp_half_hourly(self) -> np.ndarray:
        """Mean RRP per half-hour slot (partial trailing slot averaged over what exists)."""
        pad = self.n_slots * INTERVALS_PER_SLOT - self.n
        x = np.concatenate([self.rrp, np.full(pad, np.nan)]) if pad else self.rrp
        return np.nanmean(x.reshape(-1, INTERVALS_PER_SLOT), axis=1)

    def demand_half_hourly(self) -> np.ndarray:
        pad = self.n_slots * INTERVALS_PER_SLOT - self.n
        x = np.concatenate([self.demand, np.full(pad, np.nan)]) if pad else self.demand
        return np.nanmean(x.reshape(-1, INTERVALS_PER_SLOT), axis=1)

    def test_df(self) -> pd.DataFrame:
        """The test slice in the column layout the MILP/plotting code expects."""
        i = slice(self.test_start, self.n)
        return pd.DataFrame(
            {
                "SETTLEMENTDATE": (self.start_times[i] + pd.Timedelta(minutes=5)).strftime("%-d/%m/%Y %-H:%M"),
                "TOTALDEMAND": self.demand[i],
                "RRP": self.rrp[i],
                "EXPORT_KW": self.export_kw[i],
                "IMPORT_KW": self.import_kw[i],
            }
        )


def _load_month(price_csv: str, export_csv: str | None, import_csv: str | None) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    if export_csv:
        df = merge_optional_csv(df, export_csv, "EXPORT_KW")
    else:
        df["EXPORT_KW"] = 0.0
    if import_csv:
        df = merge_optional_csv(df, import_csv, "IMPORT_KW")
    else:
        df["IMPORT_KW"] = 0.0
    df["_dt"] = pd.to_datetime(df["SETTLEMENTDATE"], dayfirst=True)
    return df


def load_frame(
    history_csv: str = "data/price_DEC24.csv",
    test_csv: str = "data/price_JAN25.csv",
    history_export_csv: str | None = "data/export_DEC24.csv",
    history_import_csv: str | None = "data/import_DEC24.csv",
    test_export_csv: str | None = "data/export_JAN25.csv",
    test_import_csv: str | None = "data/import_JAN25.csv",
    household: bool = True,
    n_test_days: float | None = None,
) -> Frame:
    """
    Concatenate the history month (needed for the 7-day profile and the
    previous-day naive on 1 January) with the test month.
    """
    hist = _load_month(history_csv, history_export_csv if household else None, history_import_csv if household else None)
    test = _load_month(test_csv, test_export_csv if household else None, test_import_csv if household else None)
    if n_test_days is not None:
        test = test.iloc[: int(n_test_days * INTERVALS_PER_DAY)]
    df = pd.concat([hist, test], ignore_index=True)
    dt = df["_dt"]
    step = dt.diff().dropna().dt.total_seconds().unique()
    if not np.allclose(step, 300):
        raise ValueError(f"timeline is not contiguous 5-min: steps {step}")
    return Frame(
        start_times=pd.DatetimeIndex(dt - pd.Timedelta(minutes=5)),
        rrp=df["RRP"].to_numpy(dtype=float),
        demand=df["TOTALDEMAND"].to_numpy(dtype=float),
        export_kw=df["EXPORT_KW"].to_numpy(dtype=float),
        import_kw=df["IMPORT_KW"].to_numpy(dtype=float),
        test_start=len(hist),
    )


class Forecaster:
    """Interface. Subclasses implement price() and net_local()."""

    name: str = "base"

    def __init__(self, frame: Frame):
        self.frame = frame

    def price(self, t: int, horizon: int) -> np.ndarray:
        """Forecast RRP ($/MWh) for intervals t .. t+horizon-1, made with information available before t."""
        raise NotImplementedError

    def net_local(self, t: int, horizon: int) -> np.ndarray:
        """Forecast G - A (kW) for intervals t .. t+horizon-1."""
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------

    def seven_day_profile(self, t: int, horizon: int, n_days: int = 7) -> np.ndarray:
        """Mean of the same 5-min slot over the previous n_days days (similar-day method)."""
        idx = t + np.arange(horizon)[:, None] - INTERVALS_PER_DAY * np.arange(1, n_days + 1)[None, :]
        if idx.min() < 0:
            raise IndexError(f"not enough history for a {n_days}-day profile at t={t}")
        return self.frame.net_local[idx].mean(axis=1)

    def expand_half_hourly(self, hh: np.ndarray, t: int, horizon: int) -> np.ndarray:
        """
        Expand a half-hourly forecast `hh` (length HORIZON_SLOTS, element 0 = the
        slot containing t) to 5-min steps t .. t+horizon-1. Steps past the last
        slot repeat its value (only happens for the last <30 min of the horizon).
        """
        off = self.frame.offset_of[t]
        r = (off + np.arange(horizon)) // INTERVALS_PER_SLOT
        r = np.minimum(r, len(hh) - 1)
        return hh[r]


class HalfHourlyPriceMixin:
    """
    Price forecasters that issue a 48-slot forecast at the start of every half
    hour. Subclasses fill `self.hh_forecasts` of shape (n_slots, HORIZON_SLOTS)
    in $/MWh (rows for slots without a forecast may be NaN and must not be used).
    """

    hh_forecasts: np.ndarray

    def price(self, t: int, horizon: int) -> np.ndarray:
        k = self.frame.slot_of[t]
        hh = self.hh_forecasts[k]
        if np.isnan(hh).any():
            raise ValueError(f"{self.name}: no half-hourly forecast for slot {k} (t={t})")
        return self.expand_half_hourly(hh, t, horizon)
