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

    Parameters
    ----------
    buy_threshold  : Charge when RRP is strictly below this value ($/MWh).
    sell_threshold : Discharge when RRP is at or above this value ($/MWh).

    Returns
    -------
    strategy(rrp: float, battery_state: float) -> 'buy'|'sell'|'hold'
    """
    def strategy(rrp: float, battery_state: float) -> str:
        if rrp < buy_threshold:
            return "buy"
        if rrp >= sell_threshold:
            return "sell"
        return "hold"

    return strategy
