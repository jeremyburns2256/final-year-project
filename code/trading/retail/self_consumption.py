"""
self_consumption.py

The retail-tariff battery model: a household battery run purely for
self-consumption against a standard set-rate residential offer.

The rule is the whole model, and it needs no forecast and no optimiser:

    surplus solar (G_t > A_t)   charge the battery with the excess, up to the
                                inverter and capacity limits; export whatever
                                still does not fit.
    deficit      (A_t > G_t)    discharge the battery to serve the household
                                first, up to the inverter and capacity limits;
                                import only the shortfall.

The battery therefore never imports to charge and never exports, so it only ever
moves energy from the household's own generation to the household's own load.

Why this is the right model for a set-rate tariff
-------------------------------------------------
On a standard residential offer the import rate p^i exceeds the feed-in rate p^e
at every instant, so there is no spot spread to trade against. Every kWh the
battery holds on site is worth (p^i - p^e) - the retail spread - and that is the
entire economic story. Given a flat rate, no export limit and no aging cost, this
greedy rule is not just a heuristic, it is optimal: storing surplus as early as
possible maximises the energy on hand at every instant, and with one import rate
all deficits are equally worth serving, so there is nothing to save charge for.
An optimiser has no foresight advantage to exploit.

The value is the bill saving, not a trading profit - see billing.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from milp.model import INTERVAL_HOURS, BatteryParams


def _dispatch_frame(df, soc, charge, discharge, grid_import, grid_export, p_imp, p_exp):
    """Assemble the shared results schema, so plotting and metrics work on any run."""
    dt = INTERVAL_HOURS
    import_kwh = grid_import * dt
    export_kwh = grid_export * dt
    out = pd.DataFrame({
        "time": df["SETTLEMENTDATE"].to_numpy(),
        "battery_state": soc,
        "rrp": 1000.0 * p_imp,           # the plot's price axis shows the retail import rate ($/MWh)
        "retail_import_price": p_imp,
        "retail_export_price": p_exp,
        "export_kw": df["EXPORT_KW"].to_numpy(dtype=float),
        "import_kw": df["IMPORT_KW"].to_numpy(dtype=float),
        "charge_kw": charge,
        "discharge_kw": discharge,
        "grid_import_kwh": import_kwh,
        "grid_export_kwh": export_kwh,
        "degradation_cost": np.zeros(len(soc)),
        "cumulative_cost": np.cumsum(import_kwh * p_imp),
        "cumulative_revenue": np.cumsum(export_kwh * p_exp),
        "cumulative_degradation": np.zeros(len(soc)),
    })
    out["cumulative_profit"] = out["cumulative_revenue"] - out["cumulative_cost"]
    return out


def simulate_self_consumption(df: pd.DataFrame, params: BatteryParams, tariff) -> pd.DataFrame:
    """
    Run the self-consumption battery over the meter data.

    df must carry SETTLEMENTDATE and the meter's EXPORT_KW (B1, net export) and
    IMPORT_KW (E1, grid import) columns. Only their difference is used:
    net local power G_t - A_t, positive when the household has surplus.

    Returns a results frame with the same columns as milp.rolling.simulate_milp,
    so plotting/battery_plot.py and the rainflow summary work unchanged.
    """
    dt = INTERVAL_HOURS
    net_local = df["EXPORT_KW"].to_numpy(dtype=float) - df["IMPORT_KW"].to_numpy(dtype=float)
    ts = df["SETTLEMENTDATE"]
    p_imp = tariff.import_price(ts)
    p_exp = tariff.export_price(ts)

    T = len(df)
    soc = np.zeros(T)
    charge = np.zeros(T)
    discharge = np.zeros(T)
    e = params.e_initial

    for t in range(T):
        pc = pd_ = 0.0
        if net_local[t] > 0:
            # Surplus: store the excess. Capacity headroom is expressed as an AC
            # power limit, since only eta_c of what is drawn reaches the cells.
            room_kw = max(0.0, (params.e_max - e) / (params.eta_c * dt))
            pc = min(params.p_max_charge, net_local[t], room_kw)
        elif net_local[t] < 0:
            # Deficit: serve the household before importing. Delivering pd_ kW AC
            # costs pd_/eta_d kWh of stored energy, hence the eta_d factor here.
            avail_kw = max(0.0, (e - params.e_min) * params.eta_d / dt)
            pd_ = min(params.p_max_discharge, -net_local[t], avail_kw)

        e += pc * params.eta_c * dt - pd_ / params.eta_d * dt
        e = min(params.e_max, max(params.e_min, e))
        soc[t], charge[t], discharge[t] = e, pc, pd_

    # Eq. 2.8: positive is a net draw from the grid. By construction charge never
    # exceeds the surplus and discharge never exceeds the deficit, so the battery
    # contributes nothing to either grid direction on its own account.
    grid_net = charge - discharge - net_local
    grid_import = np.clip(grid_net, 0, None)
    grid_export = np.clip(-grid_net, 0, None)
    return _dispatch_frame(df, soc, charge, discharge, grid_import, grid_export, p_imp, p_exp)


def simulate_no_battery(df: pd.DataFrame, tariff) -> pd.DataFrame:
    """
    The counterfactual as a results frame: same household, same month, no battery.

    The meter already reports the post-solar position, so this is just the meter
    read straight through at the published rates.
    """
    T = len(df)
    ts = df["SETTLEMENTDATE"]
    return _dispatch_frame(
        df,
        soc=np.zeros(T),
        charge=np.zeros(T),
        discharge=np.zeros(T),
        grid_import=df["IMPORT_KW"].to_numpy(dtype=float),
        grid_export=df["EXPORT_KW"].to_numpy(dtype=float),
        p_imp=tariff.import_price(ts),
        p_exp=tariff.export_price(ts),
    )
