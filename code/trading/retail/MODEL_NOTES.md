# Retail (set-rate) tariff model — notes

A second operating model for the Chapter 2 battery: instead of trading against the
NEM spot price, the battery is run for **self-consumption** against a standard
Australian residential retail offer.

## The model

One rule, no forecast, no optimiser:

| Condition | Action |
|---|---|
| `G_t > A_t` (excess solar) | charge with the excess, up to `P^c_max` and remaining capacity; export the rest |
| `G_t < A_t` (deficit) | discharge to serve the house, up to `P^d_max` and stored energy; import only the shortfall |

The battery never imports to charge and never exports, so it only ever moves the
household's own generation to the household's own load. Implemented in
`retail/self_consumption.py`; the state recursion, efficiencies and power limits
are the same `BatteryParams` the MILP uses, so the two models describe the same
physical battery.

## Why this rule and not an optimiser

On a set-rate offer the import rate exceeds the feed-in rate at every instant, so
there is no spread to trade against — the battery cannot earn, it can only make
the bill smaller. Given a flat rate, no export limit and no aging cost, the greedy
rule is not a heuristic but **optimal**: charging as early as possible maximises
stored energy at every instant, and with a single import rate all deficits are
equally worth serving, so there is nothing to save charge for. A perfect-foresight
optimiser has no advantage to exploit *before aging*. Once cycles are priced it does:
`retail_trading.py` also runs the Chapter 2 MILP on the tariff's rates
(`--controller milp`; `import_price_kwh` / `export_price_kwh` and `relax_binaries`,
exact whenever both prices are non-negative), and that is the like-for-like
comparator for the perfect-foresight spot study.

## Scoring

The figure of merit is the **bill saving**, not trading profit:

```
saving = bill(no battery) − bill(with battery) − rainflow aging cost
```

Both bills use the same meter data and the same published rates. The no-battery
bill needs no model at all — the meter already reports the post-solar position. Its
import and export channels are netted within each 5-min interval, because the dispatch
models see only the net and net their own grid position the same way (billing the raw
channels credited the battery with $1.6 it did not earn). The rule has no terminal
constraint, so it starts from the SoC the same rule reaches at the end of DEC24
(0 kWh; it also ends JAN25 at 0), not from the MILP's 6.75 kWh. The daily supply charge appears in both and
cancels out of the saving; it is reported because it sets the floor a battery
cannot touch. Aging is charged ex post by rainflow, because the controller has no
price signal to weigh a cycle against.

## Results on JAN25 (13.5 kWh, AGL Residential Smart Saver, FiT 3 c/kWh)

Re-run 2026-09-21 after the audit fixes listed under "Scoring". No-battery bill:
$168.21 single rate, $196.53 time of use.

| Plan | Controller | Bill with | Gross saving | Degradation | Net | Life lost |
|---|---|---|---|---|---|---|
| single rate | rule | $112.27 | $55.94 | $63.36 | **−$7.42** | 0.528% |
| single rate | MILP J = 4 | $138.01 | $30.20 | $12.38 | **$17.82** | 0.103% |
| time of use | rule | $125.32 | $71.21 | $63.36 | $7.85 | 0.528% |
| time of use | MILP J = 4 | $134.93 | $61.61 | $25.47 | **$36.14** | 0.212% |

Rule: grid import falls 458 → 246 kWh and export 490 → 252 kWh on both plans; the
dispatch is identical, because the rule never looks at price. Only the valuation changes.

**Four things worth stating separately.**

*1. This household should stay on the single-rate plan, with or without a battery.*
The ToU plan prices 194 kWh of its evening import at 54.18 c/kWh, and the cheaper
off-peak rate does not make that back. Plan choice is worth $28 a month without a
battery, against $56 gross for the rule.

*2. The battery is worth more on the plan the household should not be on*, under
either controller, because the imports it displaces are dearer there.

*3. The rule does not pay for its modelled aging on single rate, but that is the
controller, not the tariff.* The MILP with the same aging cost the spot study uses
cycles about half as much, gives up $10–26 of gross saving, avoids $38–51 of aging and
nets positive on both plans. The rule is optimal only *before* aging. Any spot-vs-retail
comparison must therefore pair the spot MILP with the retail MILP (both perfect
foresight) and the forecast-driven spot MPC with the rule (both realisable).

*4. The degradation cost is placeholder-driven and probably overstated.* It rests on
`R_cell = 12 000 AUD`, still needing a cited installed-cost figure, and the Xu NMC
stress function, which overstates the penalty for LFP. A full-depth cycle costs
12 000 × 5.24e-4 = $6.29 for 12.7 kWh delivered, 49.5 c/kWh, which is above the
26.82 c/kWh single-rate spread, so under these inputs no deep-cycling self-consumption
battery can pay. The supply charge is $46.37 of the $168.21 bill (28%) and storage
cannot touch it.

## Rates

All rates are GST-inclusive Ausgrid-zone (NSW) retail rates, defined in
`retail/tariffs.py`. Run `python retail_trading.py --rates` to print the provenance
of every one. In short:

Both presets are published **AGL "Residential Smart Saver"** plans for the Ausgrid
zone, GST inclusive (AGL: "all prices listed are inclusive of GST except where
indicated"). Cite AGL's plan summary / Basic Plan Information Document.

- **Single rate** — 29.82 c/kWh, supply 149.57 c/day, feed-in 3.0 c/kWh.
- **Time of use** — peak 54.18 c/kWh (3:00pm–8:59pm), off-peak 21.63 c/kWh
  (9:00pm–2:59pm), supply 158.63 c/day, feed-in 3.0 c/kWh. Two periods, no shoulder,
  matching the structure Ausgrid moved residential ToU to on 1 July 2024.
- **No seasonal restriction on the retail peak.** The Ausgrid EA025 *network* tariff
  confines the residential peak to Nov–Mar and Jun–Aug; this retail plan does not pass
  that through, and the plan is what the customer is billed on. Modelled as published.
  This is a real difference between the network tariff the wholesale model uses and the
  retail plan this one uses, and deserves a sentence in the write-up.
- **Cross-check against the regulated reference price.** The single-rate plan bills
  $1,708.91/yr at the DMO model usage of 3,900 kWh, against the AER Default Market
  Offer 2025-26 reference of $1,965/yr for the Ausgrid residential customer. A market
  offer ~13% under the DMO is exactly what the DMO is for, so the two agree.
- `--fit` overrides the feed-in rate; the saving is almost entirely driven by
  (import rate − feed-in rate), so it is the single most sensitive input.
- An earlier synthetic "solar soaker" preset was removed — it was not a real offer.
- The retailer's rates already bundle network costs, so the Ausgrid EA010 network
  tariff `N` used by the spot model must **not** be added on top here.

## Stated simplifications

- **Clock time.** NEM `SETTLEMENTDATE` is interval-ending and always AEST; retail ToU
  windows are local clock time, AEDT over the summer test data. Each timestamp is
  converted from fixed AEST to the tariff's `timezone` (Australia/Sydney), so daylight
  saving is applied per date, and the period lookup uses the interval *start*.
  Getting either wrong shifts the peak window by an hour.
- No public holidays; ToU weekday/weekend handling is per-block but the modelled
  Ausgrid shape applies peak on all days.
- No demand charges, no controlled load, no VPP or event dispatch.
- Annualisation scales a single summer month to a full year. Summer has the most
  solar, so the most surplus to store. The annual and payback figures are an upper
  bound on performance, not a forecast.
