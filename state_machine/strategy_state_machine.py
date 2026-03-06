"""
strategy_state_machine.py

Threshold-based (state-machine) trading strategy for BESS arbitrage.
Returns a strategy callable compatible with bess_simulator.simulate() and
bess_simulator.simulate_profit().
"""

from __future__ import annotations
from utils.bess_simulator import (
    BESS_SIZE,
    BESS_INVERTER_CAPACITY,
    _ENERGY_PER_INTERVAL_BESS,
    _ENERGY_PER_INTERVAL_SOLAR,
    INTERVAL_HOURS,
)


def make_strategy(buy_threshold: float, sell_threshold: float):
    """
    Return a strategy callable for threshold-based trading.


    Parameters
    ----------
    buy_threshold  : Charge from grid when RRP is strictly below this value ($/MWh).
    sell_threshold : Discharge to grid when RRP is at or above this value ($/MWh).

    Returns
    -------
    strategy(rrp, battery_state, export_kw, import_kw) -> returns grid_net and bess_delta in kW'
    """

    def strategy(
        rrp: float,
        battery_state: float,
        export_kw: float = 0.0,
        import_kw: float = 0.0,
    ) -> tuple[float, float]:

        net_export = export_kw * INTERVAL_HOURS
        net_import = import_kw * INTERVAL_HOURS

        if rrp >= sell_threshold:
            if import_kw > 0:
                # Load deficit, expensive price, discharge battery to offset import and sell surplus
                bess_delta = -min(
                    battery_state, _ENERGY_PER_INTERVAL_BESS
                )  # Discharge to sell at high price
                grid_net = bess_delta + net_import  # Net grid position: discharge offsets import, surplus exported
            elif export_kw > 0:
                # Export all surplus at high price, then discharge battery to maximum capacity to sell at high price
                bess_delta = -min(
                    battery_state, _ENERGY_PER_INTERVAL_BESS - net_export
                )  # Battery discharge limited by remaining inverter capacity after solar export
                grid_net = bess_delta - net_export  # Total export: solar surplus + battery discharge (both negative)
            else:
                # No local generation, discharge battery to sell at high price
                bess_delta = -min(
                    battery_state, _ENERGY_PER_INTERVAL_BESS
                )  # Discharge to sell at high price
                grid_net = bess_delta  # Export battery discharge to grid (negative = export)
        elif rrp < buy_threshold:
            if import_kw > 0:
                # Load deficit, cheap price, charge battery from grid, supply load from grid
                bess_delta = min(
                    _ENERGY_PER_INTERVAL_BESS, BESS_SIZE - battery_state
                )  # Max charge
                grid_net = (
                    import_kw * INTERVAL_HOURS
                ) + bess_delta  # Total grid import (load + battery charge)
            elif export_kw > 0:
                # Charge battery with excess solar at cheap price, export surplus if battery full
                if battery_state < BESS_SIZE:
                    bess_delta = min(
                        net_export,
                        _ENERGY_PER_INTERVAL_SOLAR,
                        BESS_SIZE - battery_state,
                    )  # Charge battery with excess solar up to limits
                    grid_net = bess_delta - net_export  # Export remaining surplus after battery charge (negative = export)
                else:
                    # Battery full, export all excess at cheap price
                    bess_delta = 0.0  # No battery movement
                    grid_net = -net_export  # Export all excess energy at cheap price (negative = export)
            else:
                # No local generation, charge battery from grid at cheap price
                bess_delta = min(
                    _ENERGY_PER_INTERVAL_BESS, BESS_SIZE - battery_state
                )  # Max charge
                grid_net = bess_delta  # Grid import for battery charge at cheap price
        else:
            # In between buy and sell thresholds: import and export grid without using battery
            bess_delta = 0.0  # No battery movement
            grid_net = (
                -net_export + net_import
            )  # Net grid position based on local generation

        return grid_net, bess_delta

    return strategy
