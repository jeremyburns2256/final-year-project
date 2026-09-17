"""
Retail (set-rate) tariff model: a household battery operated for self-consumption
against a standard Australian residential electricity offer, rather than traded
against the NEM spot price.
"""

from retail.billing import bill, no_battery_bill, period_breakdown, retail_metrics
from retail.self_consumption import simulate_no_battery, simulate_self_consumption
from retail.tariffs import FLAT, TARIFFS, TOU, RetailTariff, TouBlock, get_tariff

__all__ = [
    "RetailTariff", "TouBlock", "FLAT", "TOU", "TARIFFS", "get_tariff",
    "bill", "no_battery_bill", "retail_metrics", "period_breakdown",
    "simulate_self_consumption", "simulate_no_battery",
]
