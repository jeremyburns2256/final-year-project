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
optimiser has no advantage to exploit. (`milp/model.py` will accept retail price
arrays via `import_price_kwh` / `export_price_kwh` if that claim ever needs
checking against the MILP; it also gains `relax_binaries`, exact whenever both
prices are non-negative, and `allow_grid_charging`.)

## Scoring

The figure of merit is the **bill saving**, not trading profit:

```
saving = bill(no battery) − bill(with battery) − rainflow aging cost
```

Both bills use the same meter data and the same published rates. The no-battery
bill needs no model at all — the meter already reports the post-solar position, so
it is the meter read straight through. The daily supply charge appears in both and
cancels out of the saving; it is reported because it sets the floor a battery
cannot touch. Aging is charged ex post by rainflow, because the controller has no
price signal to weigh a cycle against.

## Results on JAN25 (13.5 kWh, AGL Residential Smart Saver, FiT 3 c/kWh)

| Plan | Bill without | Bill with | Gross saving | Degradation | Net | Payback |
|---|---|---|---|---|---|---|
| single rate | $169.77 | $110.36 | $59.41 | $64.04 | **−$4.64** | never |
| time of use | $198.92 | $124.39 | $74.54 | $64.04 | $10.49 | 97 yr |

Grid import falls 464 → 240 kWh and export 495 → 252 kWh on both: the dispatch is
identical, because the rule never looks at price. Only the valuation changes.

**Three things worth stating separately.**

*1. This household should stay on the single-rate plan, with or without a battery.*
Its bill is lower on single rate both ways ($169.77 vs $198.92 without, $110.36 vs
$124.39 with). The ToU plan prices 198 kWh of its evening import at 54.18 c/kWh, and
the cheaper off-peak rate does not make that back. So the plan choice is worth more
than the battery here — $29 a month against $59 — which is a result in itself.

*2. The battery is worth more on the plan the household should not be on.* The gross
saving is larger on ToU ($74.54 vs $59.41) precisely because the imports it displaces
are dearer there. That is not a contradiction, it is the distinction between the level
of a bill and the value of shaving its peak, and the two point opposite ways here.

*3. On the plan it should actually be on, the battery does not pay for itself.*
$59.41/month gross against $64.04/month of modelled aging. Two causes, which should
not be conflated:

  - *The plan shape is hostile to storage.* Smart Saver single rate pairs a low usage
    rate (29.82 c/kWh) with a high supply charge (149.57 c/day). A battery only ever
    attacks the usage charge, and earns on the spread (29.82 − 3.0) = 26.82 c/kWh.
    The supply charge is $46.37 of the $169.77 bill — 27% — and storage cannot touch
    a cent of it.
  - *The degradation cost is placeholder-driven and probably overstated.* It rests on
    `R_cell = 12 000 AUD`, still needing a cited installed-cost figure, and the Xu NMC
    stress function `Φ(δ) = 5.24e-4 δ^2.03`, which overstates the penalty for the
    Powerwall 3's LFP chemistry. Both are flagged in `milp/MODEL_NOTES.md`.

The sign of the net saving is therefore not yet defensible. The gross saving is the
robust number, and even on gross alone the payback against $12 000 is ~17 years on
single rate. The honest statement: this battery is marginal-to-uneconomic on this
plan, and fixing the two degradation inputs decides which side of the line it lands.

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
