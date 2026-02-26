"""
strategy_state_machine.py

Threshold-based (state-machine) trading strategy for BESS arbitrage.
Returns a strategy callable compatible with bess_simulator.simulate() and
bess_simulator.simulate_profit().
"""

from __future__ import annotations


def make_strategy(buy_threshold: float, sell_threshold: float):
    """
    Return a strategy callable for threshold-based trading.

    Buys when rrp < buy_threshold, sells when rrp >= sell_threshold.
    The simulator enforces SoC limits — the strategy only expresses intent.
    Solar and load are passed through but not used by this strategy; the
    simulator handles passive self-consumption flows automatically in 'hold'.

    Parameters
    ----------
    buy_threshold  : Charge from grid when RRP is strictly below this value ($/MWh).
    sell_threshold : Discharge to grid when RRP is at or above this value ($/MWh).

    Returns
    -------
    strategy(rrp, battery_state, solar_kw, load_kw) -> 'buy'|'sell'|'hold'
    """
    def strategy(
        rrp: float,
        battery_state: float,
        solar_kw: float = 0.0,
        load_kw: float = 0.0,
    ) -> str:
        if rrp < buy_threshold:
            return "buy"
        if rrp >= sell_threshold:
            return "sell"
        return "hold"

    return strategy
