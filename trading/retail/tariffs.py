"""
tariffs.py

Standard (set-rate) Australian residential retail tariffs, used to price the
household's grid position instead of the NEM spot price.

The wholesale model prices both directions off one number: imports at
R_t + N and exports at R_t. A retail customer does not see R_t at all. They see

    imports  at a published rate  p^i_t  ($/kWh, GST inclusive, network costs
             already bundled in by the retailer -- so N must NOT be added again),
    exports  at a feed-in tariff  p^e_t  ($/kWh), and
    a fixed daily supply charge   S      ($/day), which no dispatch decision can change.

Because p^i_t > p^e_t at every instant on every standard offer, the arbitrage
that drives the wholesale model disappears. The battery's value is then entirely
self-consumption: solar that would have been exported at the feed-in rate
(3 c/kWh here) is stored and later displaces an import at the usage rate
(29.82 c/kWh flat, up to 54.18 c/kWh at peak). The spread is the tariff spread,
not the spot spread, and it is available every sunny day.

The tariff still matters, but only through *which* imports get displaced. The
dispatch is identical on every offer here, because the self-consumption rule
never looks at price; a ToU offer simply values the displaced evening import at
the peak rate instead of the flat rate. Time-of-use *shifting* -- buying cheap
off-peak energy to spend in the peak window -- would need grid charging, which
this model does not do.

Period definitions and rates
----------------------------
Rates are GST-inclusive retail rates for the Ausgrid distribution zone (NSW),
chosen to match the price data in data/ (NSW1) and the Ausgrid network tariff
already used by the wholesale model. See RATE_PROVENANCE below for how each
number was set and which ones still need a citation in the write-up.

Clock time
----------
NEM SETTLEMENTDATE is interval-*ending* and is always in AEST (UTC+10); it does
not shift for daylight saving. Retail ToU windows are defined in *local clock*
time, which in NSW is AEDT (UTC+11) over summer and AEST otherwise. The period
lookup therefore converts each NEM timestamp to the tariff's `timezone`
(Australia/Sydney by default, so daylight saving is applied per date rather than
assumed), and uses the interval *start* (timestamp - 5 min) so that the interval
labelled 15:00 is priced by the period covering 14:55-15:00. Getting either wrong
shifts the whole peak window by an hour and materially changes the ToU results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

INTERVAL_MINUTES = 5
NEM_TZ = "Etc/GMT-10"   # AEST, fixed UTC+10 (POSIX sign convention: Etc/GMT-10 is UTC+10)

# ---------------------------------------------------------------------------
# Rate provenance. Every number the retail model prices with is defined here.
# ---------------------------------------------------------------------------
RATE_PROVENANCE = """\
FLAT usage 29.82 c/kWh + supply 149.57 c/day, feed-in 3.0 c/kWh
    ACTUAL PUBLISHED PLAN: AGL "Residential Smart Saver", single rate pricing,
    Ausgrid distribution zone. Cite AGL's plan summary / Basic Plan Information
    Document for this plan. AGL states "all prices listed are inclusive of GST
    except where indicated", so these are GST-inclusive rates.

    Sanity check against the regulated reference price: this plan bills
        3900 * 0.2982 + 365 * 1.4957 = 1162.98 + 545.93 = $1,708.91/yr
    at the DMO model usage of 3,900 kWh, against the AER Default Market Offer
    2025-26 figure of $1,965/yr for the Ausgrid residential reference customer.
    A market offer sitting ~13% below the DMO reference is exactly what the DMO
    is for, so the two numbers agree rather than conflict. The DMO is a useful
    cross-check to quote, not the source of these rates.

    Note the shape, not just the level: this plan pairs a LOW usage rate with a
    HIGH supply charge. That is the worst combination for a battery, because the
    battery can only attack the usage charge, and the spread it earns on is
    (29.82 - 3.0) = 26.82 c/kWh rather than the ~35 c/kWh a higher-usage-rate
    plan would give.

TOU peak 54.18 / off-peak 21.63 c/kWh + supply 158.63 c/day, feed-in 3.0 c/kWh
    ACTUAL PUBLISHED PLAN: AGL "Residential Smart Saver", time of use pricing,
    Ausgrid distribution zone. GST inclusive, as above.
        Peak      3:00pm - 8:59pm   54.18 c/kWh
        Off-peak  9:00pm - 2:59pm   21.63 c/kWh
    Two periods, no shoulder, which matches the two-period structure Ausgrid moved
    residential time-of-use to on 1 July 2024.

    Note this plan carries NO seasonal restriction: peak applies year round. The
    Ausgrid EA025 *network* tariff does restrict the residential peak to Nov-Mar
    and Jun-Aug, but the retail plan as published does not pass that through, and
    the plan is what the customer is billed on. Modelled as published. Worth one
    sentence in the write-up, since it is a real difference between the network
    tariff the wholesale model uses and the retail plan this model uses.

    The supply charge is 158.63 c/day, higher than the same plan's single-rate
    variant (149.57 c/day). Both are real, so bills across the two presets differ
    by a genuine $0.09/day rather than by an assumption. This does not affect any
    saving, since the supply charge is identical in the with- and without-battery
    bills and cancels out.

FEED-IN 3.0 c/kWh flat
    AGL Residential Smart Saver, published as "+3c per kWh exported" on both the
    single-rate and time-of-use variants, so it is not an assumption here.
    FIT_LOW (2.0 c/kWh) is kept for sensitivity, since the value of a
    self-consumption battery is almost entirely driven by (p_import - p_export)
    and this is the smaller, more volatile of the two.

Both presets are published AGL plans. An earlier synthetic "solar soaker" preset
(a cheap 10am-3pm daytime import window) was removed: it was not a real offer, and
a self-consumption battery never charges from the grid, so it changed nothing but
the valuation of the residual midday import.
"""

FIT_STANDARD = 0.030   # $/kWh, AGL Residential Smart Saver
FIT_LOW = 0.020        # $/kWh, sensitivity case


@dataclass(frozen=True)
class TouBlock:
    """
    One time-of-use window.

    hours   : half-open local-clock intervals [start, end) in decimal hours.
              A window with start > end wraps midnight (e.g. (22.0, 7.0)).
    months  : 1-12, None for all months.
    weekdays_only : restrict to Mon-Fri.
    """

    label: str
    rate: float                                  # $/kWh
    hours: tuple[tuple[float, float], ...]
    months: tuple[int, ...] | None = None
    weekdays_only: bool = False

    def mask(self, hour: np.ndarray, month: np.ndarray, is_weekday: np.ndarray) -> np.ndarray:
        in_hours = np.zeros(hour.shape, dtype=bool)
        for start, end in self.hours:
            if start <= end:
                in_hours |= (hour >= start) & (hour < end)
            else:  # wraps midnight
                in_hours |= (hour >= start) | (hour < end)
        if self.months is not None:
            in_hours &= np.isin(month, self.months)
        if self.weekdays_only:
            in_hours &= is_weekday
        return in_hours


@dataclass(frozen=True)
class RetailTariff:
    """
    A standard set-rate residential offer.

    default_rate  : $/kWh charged whenever no import block matches (the off-peak
                    / anytime rate).
    import_blocks : evaluated in order, first match wins.
    feed_in_rate  : $/kWh paid on exports whenever no export block matches.
    export_blocks : optional time-varying feed-in tariff.
    supply_charge : $/day, fixed. Enters the bill but never the dispatch, since
                    no battery decision can change it.
    timezone      : IANA zone the ToU windows are defined in. NEM timestamps are
                    converted from fixed AEST into it, so daylight saving is
                    handled per date. Australia/Sydney for NSW.
    """

    name: str
    default_rate: float
    import_blocks: tuple[TouBlock, ...] = ()
    feed_in_rate: float = FIT_STANDARD
    export_blocks: tuple[TouBlock, ...] = ()
    supply_charge: float = 1.4957
    timezone: str = "Australia/Sydney"
    plan: str = ""      # the published plan these rates come from, "" if synthetic
    notes: str = ""

    # -- period lookup -----------------------------------------------------
    def _clock_parts(self, settlementdate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local-clock hour-of-day, month and weekday flag for the START of each interval."""
        ts = pd.to_datetime(pd.Series(settlementdate), dayfirst=True)
        local = ts.dt.tz_localize(NEM_TZ).dt.tz_convert(self.timezone) - pd.Timedelta(minutes=INTERVAL_MINUTES)
        hour = (local.dt.hour + local.dt.minute / 60.0).to_numpy()
        return hour, local.dt.month.to_numpy(), (local.dt.dayofweek < 5).to_numpy()

    def _series(self, settlementdate, blocks, default, default_label):
        hour, month, weekday = self._clock_parts(settlementdate)
        rate = np.full(hour.shape, float(default))
        label = np.full(hour.shape, default_label, dtype=object)
        assigned = np.zeros(hour.shape, dtype=bool)
        for blk in blocks:
            m = blk.mask(hour, month, weekday) & ~assigned
            rate[m] = blk.rate
            label[m] = blk.label
            assigned |= m
        return rate, label

    def import_price(self, settlementdate) -> np.ndarray:
        """p^i_t in $/kWh for each interval."""
        return self._series(settlementdate, self.import_blocks, self.default_rate, "off_peak")[0]

    def export_price(self, settlementdate) -> np.ndarray:
        """p^e_t in $/kWh for each interval."""
        return self._series(settlementdate, self.export_blocks, self.feed_in_rate, "fit")[0]

    def period_label(self, settlementdate) -> np.ndarray:
        """Name of the import period each interval falls in, for reporting."""
        return self._series(settlementdate, self.import_blocks, self.default_rate, "off_peak")[1]

    def supply_charge_total(self, settlementdate) -> float:
        """S x (number of days covered by the data), pro-rated for a partial day."""
        ts = pd.to_datetime(pd.Series(settlementdate), dayfirst=True)
        n_days = len(ts) * INTERVAL_MINUTES / (60.0 * 24.0)
        return self.supply_charge * n_days

    # -- convenience -------------------------------------------------------
    def with_feed_in(self, rate: float, name_suffix: str | None = None) -> "RetailTariff":
        suffix = name_suffix or f"_fit{int(round(rate * 100))}c"
        return replace(self, feed_in_rate=rate, name=self.name + suffix)

    def rate_table(self) -> pd.DataFrame:
        rows = [{"period": b.label, "rate_c_per_kwh": 100 * b.rate,
                 "hours": "; ".join(f"{s:g}-{e:g}" for s, e in b.hours),
                 "months": "all" if b.months is None else ",".join(map(str, b.months)),
                 "days": "Mon-Fri" if b.weekdays_only else "all"}
                for b in self.import_blocks]
        rows.append({"period": "off_peak / anytime", "rate_c_per_kwh": 100 * self.default_rate,
                     "hours": "all other", "months": "all", "days": "all"})
        rows.append({"period": "feed-in", "rate_c_per_kwh": 100 * self.feed_in_rate,
                     "hours": "all", "months": "all", "days": "all"})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Presets. Both are published AGL "Residential Smart Saver" plans, Ausgrid zone.
# ---------------------------------------------------------------------------
FLAT = RetailTariff(
    name="flat",
    default_rate=0.2982,
    supply_charge=1.4957,
    feed_in_rate=FIT_STANDARD,
    plan="AGL Residential Smart Saver, single rate pricing, Ausgrid zone",
    notes="Published AGL plan. Low usage rate paired with a high supply charge.",
)

TOU = RetailTariff(
    name="tou",
    default_rate=0.2163,   # off-peak, 9:00pm - 2:59pm
    import_blocks=(
        TouBlock("peak", 0.5418, ((15.0, 21.0),)),   # 3:00pm - 8:59pm, year round
    ),
    feed_in_rate=FIT_STANDARD,
    supply_charge=1.5863,
    plan="AGL Residential Smart Saver, time of use pricing, Ausgrid zone",
    notes="Published AGL plan. Two periods, no shoulder, no seasonal restriction.",
)

TARIFFS: dict[str, RetailTariff] = {t.name: t for t in (FLAT, TOU)}


def get_tariff(name: str, feed_in_rate: float | None = None, timezone: str | None = None) -> RetailTariff:
    """Look up a preset by name, optionally overriding the feed-in tariff or the local time zone."""
    try:
        t = TARIFFS[name]
    except KeyError:
        raise ValueError(f"Unknown tariff {name!r}; choose from {sorted(TARIFFS)}") from None
    if feed_in_rate is not None:
        t = replace(t, feed_in_rate=feed_in_rate)
    if timezone is not None:
        t = replace(t, timezone=timezone)
    return t
