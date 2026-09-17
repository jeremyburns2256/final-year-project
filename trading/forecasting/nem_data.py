"""
nem_data.py

Build the half-hourly NSW price/demand series used to train the price model
from the 10-year merged AEMO file (data_exploration/data/merged_nem_data.csv).

SETTLEMENTDATE is the interval *end*: 30-min periods before 2021-10-01, 5-min
after. Both are mapped to interval start and averaged into 30-min slots, so the
whole 2015-2024 record is one uniform half-hourly series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MERGED_NEM_CSV = "../data_exploration/data/merged_nem_data.csv"
FIVE_MIN_START = pd.Timestamp("2021-10-01 00:05:00")   # first 5-min settlement stamp
ASINH_SCALE = 100.0                                     # RRP / 100 before asinh


def price_transform(rrp):
    """asinh(RRP / 100): log-like for |RRP| >> 100, linear near zero, defined for negatives."""
    return np.arcsinh(np.asarray(rrp, dtype=float) / ASINH_SCALE)


def price_inverse(z):
    return np.sinh(np.asarray(z, dtype=float)) * ASINH_SCALE


def load_half_hourly(path: str = MERGED_NEM_CSV) -> pd.DataFrame:
    """Return a DataFrame indexed by 30-min slot start with columns RRP, TOTALDEMAND."""
    df = pd.read_csv(path, usecols=["SETTLEMENTDATE", "TOTALDEMAND", "RRP"])
    end = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    period = np.where(end >= FIVE_MIN_START, 5, 30)
    start = end - pd.to_timedelta(period, unit="m")
    df = df.assign(start=start).set_index("start").sort_index()
    hh = df[["RRP", "TOTALDEMAND"]].resample("30min", label="left", closed="left").mean()
    missing = hh["RRP"].isna().sum()
    if missing:
        hh = hh.interpolate(limit_direction="both")
    hh.attrs["missing_slots_filled"] = int(missing)
    return hh


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos of hour-of-day, day-of-week, day-of-year: shape (n, 6)."""
    hod = (index.hour + index.minute / 60.0) / 24.0
    dow = index.dayofweek / 7.0
    doy = (index.dayofyear - 1) / 365.25
    cols = []
    for f in (hod, dow, doy):
        cols += [np.sin(2 * np.pi * f), np.cos(2 * np.pi * f)]
    return np.stack(cols, axis=1).astype(np.float32)
