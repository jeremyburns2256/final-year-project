"""
mock_data.py

Generates synthetic solar generation and household load CSV files for BESS
simulation. Designed for NSW/QLD summer (January–February) with 5-minute
settlement intervals matching AEMO price data.

Output files (one SETTLEMENTDATE column + one data column each):
    data/solar_<name>.csv  — columns: SETTLEMENTDATE, SOLAR_KW
    data/load_<name>.csv   — columns: SETTLEMENTDATE, LOAD_KW

Solar model
-----------
  5 kW peak system (matching BESS inverter capacity).
  Sin-curve envelope between sunrise (06:00) and sunset (20:00) AEDT.
  Day-to-day cloud-cover factor via slow random walk (0.3–1.0).
  Small per-interval Gaussian noise when sun is up.

Load model (Australian household, summer)
-----------------------------------------
  Base overnight standby:   ~0.3 kW
  Morning routine (07:30):  ~2.5 kW  (kettle, shower, toaster)
  Afternoon AC (14:00):     ~2.0 kW  (summer cooling)
  Evening peak (19:00):     ~3.5 kW  (cooking, TV, lights)
  Day-to-day scale factor:  0.85–1.15 (occupancy / behaviour variation)

Usage
-----
  # Generate from a price DataFrame and return the two separate DataFrames:
  from utils.mock_data import generate_solar_df, generate_load_df

  price_df  = pd.read_csv("data/price_JAN26.csv")
  solar_df  = generate_solar_df(price_df, seed=42)
  load_df   = generate_load_df(price_df, seed=42)

  # Or run as a script to write all files:
  python utils/mock_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ── Deterministic profile shapes ─────────────────────────────────────────────

def _solar_shape(hour: float, sunrise: float = 6.0, sunset: float = 20.0) -> float:
    """Normalised solar shape [0, 1] for a given hour. Sin envelope, zero outside daylight."""
    if hour <= sunrise or hour >= sunset:
        return 0.0
    return np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))


def _load_shape(hour: float) -> float:
    """Household load shape (kW) for a given hour. Australian summer profile."""
    base    = 0.30
    morning = 2.50 * np.exp(-0.5 * ((hour -  7.5) / 0.8) ** 2)
    ac      = 2.00 * np.exp(-0.5 * ((hour - 14.0) / 2.5) ** 2)
    evening = 3.50 * np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2)
    return base + morning + ac + evening


# ── Generation functions ──────────────────────────────────────────────────────

def generate_solar_df(
    price_df: pd.DataFrame,
    peak_kw: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a solar generation DataFrame aligned to a price DataFrame.

    Parameters
    ----------
    price_df : DataFrame with SETTLEMENTDATE column (d/m/Y H:MM format).
    peak_kw  : Nameplate PV system capacity in kW.
    seed     : Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: SETTLEMENTDATE, SOLAR_KW
    """
    rng   = np.random.default_rng(seed)
    times = pd.to_datetime(price_df["SETTLEMENTDATE"], dayfirst=True)
    hours = (times.dt.hour + times.dt.minute / 60.0).to_numpy()
    dates = times.dt.date.to_numpy()

    # Slow random walk for cloud cover (0.3–1.0), one value per day
    unique_dates = sorted(set(dates))
    cloud = 0.85
    cloud_factor: dict = {}
    for d in unique_dates:
        cloud = float(np.clip(cloud + rng.normal(0, 0.12), 0.30, 1.00))
        cloud_factor[d] = cloud

    solar_vals = np.empty(len(price_df))
    for i, (h, d) in enumerate(zip(hours, dates)):
        base  = peak_kw * _solar_shape(h)
        noise = rng.normal(0.0, 0.04 * base) if base > 0 else 0.0
        solar_vals[i] = max(0.0, base * cloud_factor[d] + noise)

    return pd.DataFrame({
        "SETTLEMENTDATE": price_df["SETTLEMENTDATE"].values,
        "SOLAR_KW":       np.round(solar_vals, 3),
    })


def generate_load_df(
    price_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a household load DataFrame aligned to a price DataFrame.

    Parameters
    ----------
    price_df : DataFrame with SETTLEMENTDATE column (d/m/Y H:MM format).
    seed     : Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: SETTLEMENTDATE, LOAD_KW
    """
    rng   = np.random.default_rng(seed)
    times = pd.to_datetime(price_df["SETTLEMENTDATE"], dayfirst=True)
    hours = (times.dt.hour + times.dt.minute / 60.0).to_numpy()
    dates = times.dt.date.to_numpy()

    unique_dates = sorted(set(dates))
    load_scale   = {d: rng.uniform(0.85, 1.15) for d in unique_dates}

    load_vals = np.empty(len(price_df))
    for i, (h, d) in enumerate(zip(hours, dates)):
        base          = _load_shape(h)
        load_vals[i]  = max(0.1, base * load_scale[d] + rng.normal(0.0, 0.08))

    return pd.DataFrame({
        "SETTLEMENTDATE": price_df["SETTLEMENTDATE"].values,
        "LOAD_KW":        np.round(load_vals, 3),
    })


# ── CLI: write separate solar and load CSVs ───────────────────────────────────

def _write_pair(price_path: str, solar_seed: int, load_seed: int) -> None:
    """Write solar_<name>.csv and load_<name>.csv alongside the price CSV."""
    name      = os.path.basename(price_path).replace("price_", "").replace(".csv", "")
    solar_out = os.path.join(os.path.dirname(price_path), f"solar_{name}.csv")
    load_out  = os.path.join(os.path.dirname(price_path), f"load_{name}.csv")

    price_df = pd.read_csv(price_path)
    n_days   = len(price_df) * (5 / 60) / 24

    solar_df = generate_solar_df(price_df, seed=solar_seed)
    solar_df.to_csv(solar_out, index=False)
    solar_kwh = (solar_df["SOLAR_KW"] * (5 / 60)).sum()
    print(f"  {solar_out}")
    print(f"    {solar_kwh:.0f} kWh total  ({solar_kwh/n_days:.1f} kWh/day avg)"
          f"  peak {solar_df['SOLAR_KW'].max():.2f} kW")

    load_df = generate_load_df(price_df, seed=load_seed)
    load_df.to_csv(load_out, index=False)
    load_kwh = (load_df["LOAD_KW"] * (5 / 60)).sum()
    print(f"  {load_out}")
    print(f"    {load_kwh:.0f} kWh total  ({load_kwh/n_days:.1f} kWh/day avg)"
          f"  peak {load_df['LOAD_KW'].max():.2f} kW")


if __name__ == "__main__":
    pairs = [
        ("data/price_JAN26.csv", 42,  43),
        ("data/price_FEB26.csv", 99, 100),
    ]

    print("Generating mock solar and load CSVs...\n")
    for price_path, solar_seed, load_seed in pairs:
        if not os.path.exists(price_path):
            print(f"  Skipping {price_path} — not found.")
            continue
        print(f"  Source: {price_path}")
        _write_pair(price_path, solar_seed, load_seed)
        print()

    print("Done.")
    print("\nExample usage:")
    print("  run_trading_simulation(")
    print("      solar_csv='data/solar_JAN26.csv',")
    print("      load_csv='data/load_JAN26.csv',")
    print("  )")
