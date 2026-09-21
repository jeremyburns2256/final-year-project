"""
Household battery on a standard Australian residential retail tariff.

The battery is run for self-consumption only: charge from excess solar, discharge
to serve the house before importing. No spot price, no arbitrage, no optimiser.
The question is how much the battery takes off the bill, and how that changes with
the tariff, the feed-in rate and the battery size.

Run from the trading/ directory:
    python retail_trading.py                       # all tariffs on JAN25
    python retail_trading.py --tariff flat         # one tariff
    python retail_trading.py --fit 0.02            # low feed-in sensitivity

Companion to milp_trading.py; run_retail_simulation mirrors run_milp_simulation.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from milp.degradation import rainflow_aging_cost
from milp.model import INTERVAL_HOURS, BatteryParams
from plotting.battery_plot import plot_battery_trading
from retail.billing import period_breakdown, retail_metrics
from retail.self_consumption import simulate_no_battery, simulate_self_consumption
from retail.tariffs import RATE_PROVENANCE, TARIFFS, get_tariff
from utils.data import merge_optional_csv

TEST_CSV = "data/price_JAN25.csv"
TEST_EXPORT_CSV = "data/export_JAN25.csv"
TEST_IMPORT_CSV = "data/import_JAN25.csv"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
R_CELL = 12_000.0  # AUD, placeholder replacement cost for a 13.5 kWh Powerwall 3 (see milp/MODEL_NOTES.md)


def load_test_data(test_csv=TEST_CSV, export_csv=TEST_EXPORT_CSV, import_csv=TEST_IMPORT_CSV, n_days=None):
    df = pd.read_csv(test_csv)
    df = merge_optional_csv(df, export_csv, "EXPORT_KW")
    df = merge_optional_csv(df, import_csv, "IMPORT_KW")
    if n_days is not None:
        df = df.iloc[: int(n_days * 288)].reset_index(drop=True)
    return df


def run_retail_simulation(tariff, params: BatteryParams | None = None, n_days=None, r_cell=None,
                          df=None, verbose=True, plot=True, plot_output_path=None):
    """
    Simulate the self-consumption battery on one tariff. Returns dict(results_df, metrics, params).

    r_cell prices the ex-post rainflow life loss. The controller does not optimise
    against it -- it has no price signal to weigh a cycle against -- so the aging
    cost is charged after the fact, which is the honest way to score a rule that
    cycles whenever the sun and the load tell it to.
    """
    params = params or BatteryParams()
    r_cell = R_CELL if r_cell is None else r_cell
    df = load_test_data(n_days=n_days) if df is None else df
    results = simulate_self_consumption(df, params, tariff)

    rf = rainflow_aging_cost(results["battery_state"].to_numpy(), params.e_max, r_cell,
                             params.phi_a, params.phi_k)
    m = retail_metrics(results, df, tariff, degradation_cost=rf["rainflow_cost"], r_cell=r_cell)
    m.update({
        "e_max": params.e_max,
        "r_cell": r_cell,
        "feed_in_c_per_kwh": 100 * tariff.feed_in_rate,
        "throughput_kwh": float(results["discharge_kw"].sum() * INTERVAL_HOURS),
        "equivalent_full_cycles": float(results["discharge_kw"].sum() * INTERVAL_HOURS) / params.e_max,
        "life_loss_pct": 100 * rf["life_loss_fraction"],
        "rainflow_cycles": rf["n_cycles"],
        "mean_cycle_depth": rf["mean_cycle_depth"],
    })

    if verbose:
        print_metrics(f"{tariff.name}  {params.e_max:g} kWh  FiT {100 * tariff.feed_in_rate:.0f} c/kWh", m)
    if plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_battery_trading(
            results,
            title=f"Retail self-consumption — {tariff.name} ({params.e_max:g} kWh)",
            subtitle=(f"{tariff.plan}. The price panel shows the retail import rate in $/MWh, and the "
                      "profit tiles are the with-battery energy bill, not a trading result: the model is "
                      "scored as bill saving net of rainflow aging in results/retail_summary.csv."),
            output_path=plot_output_path or f"{PLOTS_DIR}/retail_{tariff.name}_E{params.e_max:g}.html",
            bess_size=params.e_max,
            show_plot=False,
        )
    return {"results_df": results, "metrics": m, "params": params, "tariff": tariff}


def print_metrics(label: str, m: dict) -> None:
    print(f"{'-' * 60}\n  {label}   ({m['days']:.0f} days)\n{'-' * 60}")
    print(f"Bill without battery:         ${m['bill_no_battery']:9.2f}"
          f"   (usage ${m['usage_charge_no_battery']:.2f}, FiT credit -${m['feed_in_credit_no_battery']:.2f},"
          f" supply ${m['supply_charge']:.2f})")
    print(f"Bill with battery:            ${m['bill_with_battery']:9.2f}"
          f"   (usage ${m['usage_charge_with_battery']:.2f}, FiT credit -${m['feed_in_credit_with_battery']:.2f},"
          f" supply ${m['supply_charge']:.2f})")
    print(f"Gross saving:                 ${m['gross_saving']:9.2f}   ${m['gross_saving'] / m['days']:.2f}/day")
    print(f"Degradation (rainflow):       ${m['degradation_cost']:9.2f}   "
          f"{m['life_loss_pct']:.3f}% of life, {m['rainflow_cycles']:.0f} cycles, mean depth {m['mean_cycle_depth']:.2f}")
    print(f"Net saving:                   ${m['net_saving']:9.2f}   ${m['net_saving_per_day']:.2f}/day")
    print(f"Annualised net saving:        ${m['annualised_net_saving']:9.2f}   "
          f"simple payback {m['simple_payback_years']:.1f} yr at R_cell=${m['r_cell']:.0f}")
    print(f"Grid import {m['import_kwh_no_battery']:.0f} -> {m['import_kwh_with_battery']:.0f} kWh"
          f"   export {m['export_kwh_no_battery']:.0f} -> {m['export_kwh_with_battery']:.0f} kWh"
          f"   throughput {m['throughput_kwh']:.0f} kWh ({m['equivalent_full_cycles']:.1f} EFC)")


def run_tariff_comparison(tariff_names=None, n_days=None, feed_in_rate=None, e_max=None,
                          plot=True, verbose=True):
    """Same battery, same meter data, every tariff. Isolates the effect of the price structure."""
    names = tariff_names or list(TARIFFS)
    df = load_test_data(n_days=n_days)
    rows = []
    for name in names:
        tariff = get_tariff(name, feed_in_rate=feed_in_rate)
        params = BatteryParams(e_max=e_max) if e_max else BatteryParams()
        if e_max:
            params.e_initial = e_max * 0.5
        out = run_retail_simulation(tariff, params, df=df, plot=plot, verbose=verbose)
        rows.append(out["metrics"])
        if verbose:
            print("\n  Saving by tariff period:")
            print(period_breakdown(out["results_df"], df, tariff).round(2).to_string(index=False))
            print()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tariff", action="append", choices=sorted(TARIFFS),
                    help="tariff to run (repeatable); default is all of them")
    ap.add_argument("--days", type=float, default=None, help="limit to the first N days of the test data")
    ap.add_argument("--fit", type=float, default=None, help="override the feed-in tariff, $/kWh (e.g. 0.02)")
    ap.add_argument("--e-max", type=float, default=None, help="override usable capacity, kWh")
    ap.add_argument("--rates", action="store_true", help="print where every rate came from, and exit")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.rates:
        print(RATE_PROVENANCE)
        return

    verbose = not args.quiet
    summary = run_tariff_comparison(args.tariff, n_days=args.days, feed_in_rate=args.fit,
                                    e_max=args.e_max, plot=not args.no_plot, verbose=verbose)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.days:g}d" if args.days else ""
    summary.to_csv(f"{RESULTS_DIR}/retail_summary{suffix}.csv", index=False)

    cols = ["tariff", "e_max", "r_cell", "feed_in_c_per_kwh", "days", "bill_no_battery", "bill_with_battery",
            "gross_saving", "degradation_cost", "net_saving", "annualised_net_saving",
            "simple_payback_years", "equivalent_full_cycles"]
    pd.set_option("display.width", 250)
    print(f"\n==== Retail self-consumption summary (JAN25{suffix}) ====")
    print(summary[cols].round(2).to_string(index=False))
    print(f"\nWritten {RESULTS_DIR}/retail_summary{suffix}.csv")


if __name__ == "__main__":
    main()
