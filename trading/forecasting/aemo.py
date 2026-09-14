"""
aemo.py

AEMO pre-dispatch price forecaster (FORECAST_NOTES: AEMO baseline).

Data: MMSDM "all runs" PREDISPATCHPRICE archives, one zip per month, from
  https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/<yyyy>/MMSDM_<yyyy>_<mm>/
      MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_ARCHIVE%23PREDISPATCHPRICE%23ALL%23FILE01%23<yyyymm>010000.zip
Each pre-dispatch run (PREDISPATCHSEQNO) is published at LASTCHANGED and gives a
30-min RRP for every period (DATETIME = period end) to the end of the next
trading day. The monthly file in DATA/ (without ALL) keeps only the final run
per period and is not usable as a day-ahead forecast.

At interval t the forecaster uses the most recent run published at or before
the interval start, holds each 30-min forecast for its six 5-min steps, and
fills any steps beyond the run's horizon with the seasonal naive.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

from forecasting.base import Forecaster, Frame
from forecasting.naive import SeasonalNaiveForecaster

CACHE_DIR = "data/nemosis_cache"
ARCHIVE_URL = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m}/"
    "MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_ARCHIVE%23PREDISPATCHPRICE%23ALL%23FILE01%23{y}{m}010000.zip"
)


def archive_path(year: int, month: int, cache_dir: str = CACHE_DIR) -> str:
    return f"{cache_dir}/predispatch_all_{year}_{month:02d}.zip"


def download_archive(year: int, month: int, cache_dir: str = CACHE_DIR) -> str:
    import urllib.request

    path = archive_path(year, month, cache_dir)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(ARCHIVE_URL.format(y=year, m=f"{month:02d}"), path)
    return path


def load_predispatch_runs(months: list[tuple[int, int]], region: str = "NSW1", cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Long table: PREDISPATCHSEQNO, published (LASTCHANGED), period_end (DATETIME), RRP. Intervention runs excluded."""
    parts = []
    for y, m in months:
        path = download_archive(y, m, cache_dir)
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0])
        lines = [ln for ln in raw.decode().splitlines() if ln.startswith("I,") or ln.startswith("D,")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), header=0)
        df = df[(df["REGIONID"] == region) & (df["INTERVENTION"] == 0)]
        parts.append(df[["PREDISPATCHSEQNO", "LASTCHANGED", "DATETIME", "RRP"]])
    runs = pd.concat(parts, ignore_index=True)
    runs["published"] = pd.to_datetime(runs["LASTCHANGED"], format="%Y/%m/%d %H:%M:%S")
    runs["period_end"] = pd.to_datetime(runs["DATETIME"], format="%Y/%m/%d %H:%M:%S")
    # A run's publish time is the same for all its periods; keep the max in case of stragglers.
    runs["published"] = runs.groupby("PREDISPATCHSEQNO")["published"].transform("max")
    return runs.sort_values(["published", "period_end"]).reset_index(drop=True)


class AemoPredispatchForecaster(Forecaster):
    """Latest AEMO pre-dispatch run before t, held to 5-min, naive-filled beyond its horizon."""

    name = "aemo"

    def __init__(self, frame: Frame, region: str = "NSW1", cache_dir: str = CACHE_DIR):
        super().__init__(frame)
        self.naive = SeasonalNaiveForecaster(frame)
        months = sorted({(d.year, d.month) for d in (frame.start_times[frame.test_start], frame.start_times[-1])})
        first_test = frame.start_times[frame.test_start]
        prev = first_test - pd.Timedelta(days=1)
        months = sorted(set(months) | {(prev.year, prev.month)})
        runs = load_predispatch_runs(months, region, cache_dir)
        self.run_published = []
        self.run_period_end = []
        self.run_rrp = []
        for _, g in runs.groupby("PREDISPATCHSEQNO", sort=False):
            self.run_published.append(g["published"].iloc[0])
            self.run_period_end.append(g["period_end"].to_numpy())
            self.run_rrp.append(g["RRP"].to_numpy(dtype=float))
        order = np.argsort(np.array(self.run_published, dtype="datetime64[ns]"))
        self.run_published = np.array(self.run_published, dtype="datetime64[ns]")[order]
        self.run_period_end = [self.run_period_end[i] for i in order]
        self.run_rrp = [self.run_rrp[i] for i in order]
        self.n_runs = len(self.run_rrp)
        self.fill_steps = 0
        self.calls = 0

    def price(self, t, horizon):
        now = np.datetime64(self.frame.start_times[t])
        i = int(np.searchsorted(self.run_published, now, side="right")) - 1
        if i < 0:
            raise ValueError(f"aemo: no pre-dispatch run published before {now}")
        ends, rrp = self.run_period_end[i], self.run_rrp[i]
        times = now + np.arange(horizon) * np.timedelta64(5, "m")
        idx = np.searchsorted(ends, times, side="right")        # first period whose end is > time
        out = self.naive.price(t, horizon).astype(float)
        ok = idx < len(ends)
        out[ok] = rrp[idx[ok]]
        self.fill_steps += int((~ok).sum())
        self.calls += 1
        return out

    def net_local(self, t, horizon):
        return self.seven_day_profile(t, horizon)
