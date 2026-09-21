"""
billing.py

Bill accounting for the retail (set-rate) model.

Under a spot tariff the natural score is trading profit, because the battery can
earn money outright. Under a standard residential tariff it cannot: exports are
paid a feed-in tariff well below the import rate, so a household battery never
runs a surplus, it only makes the bill smaller. The figure of merit is therefore

    saving = bill(no battery) - bill(with battery) - aging cost

where both bills are computed on the same meter data and the same published
rates, and the no-battery bill is just the meter read straight through. The
daily supply charge appears in both bills and cancels out of the saving; it is
still reported because it sets how much of the bill a battery cannot touch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from milp.model import INTERVAL_HOURS

DAYS_PER_YEAR = 365.25


def bill(import_kwh: np.ndarray, export_kwh: np.ndarray, import_price: np.ndarray,
         export_price: np.ndarray, supply_charge_total: float) -> dict:
    """Energy charges, feed-in credit and supply charge over the period ($)."""
    usage = float(np.sum(import_kwh * import_price))
    credit = float(np.sum(export_kwh * export_price))
    return {
        "usage_charge": usage,
        "feed_in_credit": credit,
        "supply_charge": float(supply_charge_total),
        "total_bill": usage - credit + float(supply_charge_total),
        "import_kwh": float(np.sum(import_kwh)),
        "export_kwh": float(np.sum(export_kwh)),
    }


def no_battery_bill(df: pd.DataFrame, import_price: np.ndarray, export_price: np.ndarray,
                    supply_charge_total: float, import_col: str = "IMPORT_KW",
                    export_col: str = "EXPORT_KW") -> dict:
    """
    The counterfactual: the same household, same month, no battery.

    The meter already reports the post-solar position (E1 import, B1 net export),
    so the no-battery bill needs no dispatch model at all.
    """
    imp = df[import_col].to_numpy(dtype=float) * INTERVAL_HOURS
    exp = df[export_col].to_numpy(dtype=float) * INTERVAL_HOURS
    return bill(imp, exp, import_price, export_price, supply_charge_total)


def battery_bill(results: pd.DataFrame, import_price: np.ndarray, export_price: np.ndarray,
                 supply_charge_total: float) -> dict:
    """Bill implied by a dispatch result (grid_import_kwh / grid_export_kwh columns)."""
    return bill(results["grid_import_kwh"].to_numpy(dtype=float),
                results["grid_export_kwh"].to_numpy(dtype=float),
                import_price, export_price, supply_charge_total)


def retail_metrics(results: pd.DataFrame, df: pd.DataFrame, tariff, degradation_cost: float,
                   r_cell: float, import_col: str = "IMPORT_KW", export_col: str = "EXPORT_KW") -> dict:
    """
    Bill, counterfactual bill and saving for one dispatch run, annualised.

    degradation_cost is the ex-post rainflow aging cost over the simulated period
    (in $), so `net_saving` is what the household is actually ahead by once the
    life consumed by the cycling is paid for.

    Annualisation scales the simulated period to a full year. The test data is a
    single summer month, so the annual figures assume that month is representative
    -- it is not (summer has the most solar, so the most surplus to store), and the
    payback figure should be read as an upper bound on performance, not a forecast.
    """
    ts = df["SETTLEMENTDATE"]
    p_imp = tariff.import_price(ts)
    p_exp = tariff.export_price(ts)
    supply = tariff.supply_charge_total(ts)
    n_days = len(df) * INTERVAL_HOURS / 24.0

    base = no_battery_bill(df, p_imp, p_exp, supply, import_col, export_col)
    with_batt = battery_bill(results, p_imp, p_exp, supply)

    gross = base["total_bill"] - with_batt["total_bill"]
    net = gross - degradation_cost
    scale = DAYS_PER_YEAR / n_days

    # kWh of the household's own surplus that the battery kept on site rather
    # than exporting: the self-consumption channel, separated from ToU shifting.
    export_avoided = base["export_kwh"] - with_batt["export_kwh"]
    import_avoided = base["import_kwh"] - with_batt["import_kwh"]

    return {
        "tariff": tariff.name,
        "days": n_days,
        "bill_no_battery": base["total_bill"],
        "bill_with_battery": with_batt["total_bill"],
        "usage_charge_no_battery": base["usage_charge"],
        "usage_charge_with_battery": with_batt["usage_charge"],
        "feed_in_credit_no_battery": base["feed_in_credit"],
        "feed_in_credit_with_battery": with_batt["feed_in_credit"],
        "supply_charge": base["supply_charge"],
        "import_kwh_no_battery": base["import_kwh"],
        "import_kwh_with_battery": with_batt["import_kwh"],
        "export_kwh_no_battery": base["export_kwh"],
        "export_kwh_with_battery": with_batt["export_kwh"],
        "import_avoided_kwh": import_avoided,
        "export_retained_kwh": export_avoided,
        "gross_saving": gross,
        "degradation_cost": degradation_cost,
        "net_saving": net,
        "net_saving_per_day": net / n_days if n_days else float("nan"),
        "annualised_gross_saving": gross * scale,
        "annualised_net_saving": net * scale,
        "simple_payback_years": (r_cell / (net * scale)) if net * scale > 0 else float("inf"),
    }


def period_breakdown(results: pd.DataFrame, df: pd.DataFrame, tariff) -> pd.DataFrame:
    """
    Where the saving comes from, split by tariff period.

    Shows the mechanism directly. The self-consumption rule never imports to
    charge, so every row should show import shed and export shed, never import
    added; on a ToU offer the peak row is where most of the value sits, because
    that is where the displaced evening import is priced.
    """
    ts = df["SETTLEMENTDATE"]
    labels = tariff.period_label(ts)
    p_imp = tariff.import_price(ts)
    p_exp = tariff.export_price(ts)

    base_imp = df["IMPORT_KW"].to_numpy(dtype=float) * INTERVAL_HOURS
    base_exp = df["EXPORT_KW"].to_numpy(dtype=float) * INTERVAL_HOURS
    batt_imp = results["grid_import_kwh"].to_numpy(dtype=float)
    batt_exp = results["grid_export_kwh"].to_numpy(dtype=float)

    frame = pd.DataFrame({
        "period": labels,
        "rate_c_per_kwh": 100 * p_imp,
        "import_kwh_no_battery": base_imp,
        "import_kwh_with_battery": batt_imp,
        "export_kwh_no_battery": base_exp,
        "export_kwh_with_battery": batt_exp,
        "usage_saving": base_imp * p_imp - batt_imp * p_imp,
        "feed_in_change": batt_exp * p_exp - base_exp * p_exp,
    })
    out = frame.groupby("period", as_index=False).agg(
        rate_c_per_kwh=("rate_c_per_kwh", "max"),
        hours=("period", lambda s: len(s) * INTERVAL_HOURS),
        import_kwh_no_battery=("import_kwh_no_battery", "sum"),
        import_kwh_with_battery=("import_kwh_with_battery", "sum"),
        export_kwh_no_battery=("export_kwh_no_battery", "sum"),
        export_kwh_with_battery=("export_kwh_with_battery", "sum"),
        usage_saving=("usage_saving", "sum"),
        feed_in_change=("feed_in_change", "sum"),
    )
    out["net_saving"] = out["usage_saving"] + out["feed_in_change"]
    return out.sort_values("rate_c_per_kwh", ascending=False).reset_index(drop=True)
