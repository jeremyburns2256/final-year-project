# State forecasting: design decisions

Decisions from the 2026-09-14 grilling session. Companion to `milp/MODEL_NOTES.md`.
Thesis references: Sections 4.1, 5.1, 6.4; Gantt items 4.1-4.3.

## Research question

Primary result: **dispatch value of forecasts**. Headline metric is MILP net profit
including rainflow degradation on JAN25 under four forecasters, with the gap to
perfect foresight as the quantity of interest. Forecast MAE/RMSE (in $/MWh, plus
relative MAE vs seasonal naive) is reported as a secondary table.

| Forecaster | Price | Net-local |
|---|---|---|
| Perfect | actual | actual |
| PerfectPrice | actual | 7-day time-of-day profile |
| SeasonalNaive | same 30-min slot previous day | 7-day time-of-day profile |
| AemoPredispatch | latest PREDISPATCHPRICE run before t, naive fill beyond horizon | 7-day profile |
| LstmPrice | LSTM (below) | 7-day profile |

## What is forecast

- **Price (RRP)**: ML model. Section 5.1 Eq. 5.3 stands: inputs are lagged RRP and
  lagged system demand (plus calendar). No AEMO inputs, no weather, so the ML and
  AEMO forecasters are independent information sources.
- **Household**: one net-local series (`export_kw - import_kw`), consistent with
  MODEL_NOTES correction 2. Forecast = mean of the same 5-min slot over the previous
  7 days (DEC24 supplies history for early January). Presented as a similar-day
  method (Cao et al. [27]). Section 5.1's separate G and A forecasts are dropped:
  the meter gives only B1/E1 and one house for two months cannot train an LSTM.

## Rolling loop (MPC)

- 24 h horizon at 5-min resolution (288 steps), re-plan every 5 min, commit one
  interval. 8928 solves per forecaster on JAN25.
- Step 0 price is the **actual** dispatch RRP (AEMO publishes it before the
  interval starts; Amber-style retailers relay it). Steps 1..287 are forecast.
  Step 0 net-local is forecast (meter reading is not available in advance).
- Settlement: the committed P_c, P_d are executed as planned; D_i, D_e are
  recomputed from the balance with actual net-local; cost uses actual RRP.
  Feasible because no grid limits are enforced. Per-segment SoC carried forward.
- Terminal constraint Eq. 2.6k stays at the end of each 24 h horizon.
- Perfect foresight is **re-run with the same 24 h / 5-min loop** so the gap
  isolates forecast error rather than horizon choice.
- PerfectPrice (added 2026-09-17) keeps the actual price but forecasts net-local
  with the same 7-day profile as the real forecasters. It is the fair reference
  for the price forecasters: perfect − PerfectPrice is the cost of the household
  forecast, PerfectPrice − forecaster is the cost of the price forecast alone.
- Solver time limit reduced per window (8928 windows; a 120 s stall is fatal);
  accept the incumbent rather than raise when the limit hits.
- Configuration: household scenario, R_cell = 12 000, J = 4 only. The J-sweep
  had flattened by J = 4 (28.7 vs 32.4 at J = 16) and it solves in ~0.2 s per
  24 h window; roughly 2 h of HiGHS across the four forecasters.

## Price model

- Resolution: forecast 48 half-hour values, each held for six 5-min steps.
  Uses all ten years uniformly (30-min pre-Oct-2021 as-is, 5-min after averaged
  to 30-min) and matches pre-dispatch resolution for a like-for-like comparison.
- Architecture: 2-layer LSTM encoder over the previous 7 days (336 steps) of
  [asinh(RRP/100), scaled TOTALDEMAND, sin/cos hour-of-day, day-of-week,
  day-of-year], dense head with 48 outputs (direct multi-output, no recursion).
  PyTorch, CPU. XGBoost with horizon-as-feature is a stretch goal to test the
  Zhou et al. [25] claim.
- Target transform: asinh(RRP/100), spikes and negative prices kept, Huber loss.
  Follows Uniejewski, Weron & Ziel (2018, IEEE TPWRS) and the 2026 follow-up
  (arXiv 2511.13603); Zhou et al. mean-replace outliers, which would blind the
  controller to exactly the intervals that carry the revenue (Fig. 5.8).
- Split: train 2015-2023, validate 2024 (early stopping, hyperparameters),
  test JAN25 (touched only for the final forecast-then-dispatch run).

## AEMO pre-dispatch

- Source: MMSDM monthly archives, `PREDISP_ALL_DATA/PUBLIC_ARCHIVE#PREDISPATCHPRICE#ALL#...zip`
  for 2024-12 and 2025-01, downloaded directly from NEMWeb by `forecasting/aemo.py`
  and cached in `data/nemosis_cache/` (gitignored). The `nemosis` package was
  planned but has no pre-dispatch tables, and the plain `DATA/` archive keeps only
  the final run per period (a 30-min-ahead series), so it cannot serve as a
  day-ahead forecast. The all-runs file has every PREDISPATCHSEQNO with 32-79
  periods each; intervention runs are dropped.
- Finding (2026-09-14): on JAN25 pre-dispatch has a lower *median* error than the
  naive (16 vs 26 $/MWh) but a much higher MAE (162 vs 62) because it forecasts
  market-price-cap spikes on the hot days (14-15 and 21-22 Jan) that mostly did
  not eventuate after rebidding. Forecast intervals above 1000 $/MWh are ~1% of
  AEMO's forecasts vs ~0.1% of actuals and carry 76% of its absolute error. The
  study therefore reports MAE, median AE and a spike-clipped MAE, and does not
  clip the AEMO forecast itself (a real controller sees the unclipped signal).
- At each re-plan use the most recent run published before t; hold each 30-min
  forecast to 5-min; fill any shortfall inside the 24 h horizon (the trading-day
  boundary at 04:00) with the seasonal-naive value.
- Precedent: Karimi-Arpanahi et al. [29] Section 5.4 use pre-dispatch prices as
  the realistic-information scenario against perfect information.

## Code layout

```
trading/
  forecasting/
    __init__.py
    base.py            Forecaster interface: price(t) -> (288,), net_local(t) -> (288,)
    naive.py           SeasonalNaive price, SevenDayProfile net-local
    aemo.py            AemoPredispatch (+ NEMWeb MMSDM fetch/cache)
    nem_data.py        10-year half-hourly series, asinh transform, calendar features
    lstm.py            LstmPrice model definition, dataset, inference
    train_price_lstm.py
  milp/rolling.py      + simulate_milp_mpc(df, params, forecaster, ...)
  forecast_trading.py  driver mirroring milp_trading.py; writes results/forecast_summary.csv
```
Existing perfect-foresight `simulate_milp` is untouched. requirements: add
`torch`; `tensorflow` removed. No nemosis (see AEMO section).

## Thesis follow-ups

- Chapter 4: add AEMO pre-dispatch/P5MIN as the operational baseline; add the
  variance-stabilising-transform literature; note that no cited NEM ML paper
  compares against AEMO or a named naive baseline.
- Section 5.1: replace separate G/A forecasts with a single net-local similar-day
  forecast and state why (meter data, single house).
- Section 6.4: state the MPC loop, the step-0 actual price assumption, and that
  the comparison is against perfect foresight re-run at the same horizon.

## Results (2026-09-15, JAN25, household, J = 4, R_cell = 12 000)

| Forecaster | Net profit incl. rainflow deg. | Gap to perfect | Price MAE | Median AE | Forecast spikes > 1000 |
|---|---|---|---|---|---|
| Perfect | 28.74 | 0 | 0 | 0 | actual: 0.1% |
| Perfect price, forecast load | 19.64 | 9.10 | 0 | 0 | – |
| Naive | 15.92 | 12.82 | 61.3 | 25.3 | 0.4% |
| AEMO pre-dispatch | 14.67 | 14.07 | 161.6 | 16.4 | 1.0% |
| LSTM | 15.94 | 12.80 | 44.6 | 22.0 | 0.0% |

- Perfect foresight on the 24 h / 5-min MPC loop reproduces the 48 h / 24 h
  rolling result (28.74) exactly: the day-ahead commit cost nothing under
  perfect information, so the whole gap below is forecast error.
- Perfect price with the 7-day load profile (added 2026-09-17) earns 19.64: the
  household forecast alone costs 9.10 of the 12.8-14.1 gap, so the price
  forecasts cost only 3.70 (LSTM) to 4.97 (AEMO) against the fair reference.
  Most of the "forecast error" gap is the net-local forecast, not the price.
- All three real forecasters capture ~55% of the perfect-foresight profit.
  The LSTM cuts price MAE by 27% relative to the naive (rMAE 0.73 on JAN25,
  0.68 on the 2024 validation year) but delivers the same dispatch value,
  because it never forecasts a spike and the unrealised ~$13 is spike revenue.
- AEMO pre-dispatch is the most accurate in the normal regime (median AE 16)
  and the worst by MAE because of forecast market-cap spikes that did not
  eventuate; in dispatch it finishes marginally below the naive.
- Forecast accuracy (MAE) and dispatch value are therefore only loosely
  coupled in this month: what matters is spike timing, which none of the
  forecasters captures. Candidates for Part B: a spike-probability model,
  a stochastic/scenario MILP, or a blended AEMO-plus-LSTM forecast.
- **Correction (2026-09-21).** The two bullets above misattribute the gap. Broken
  down by price band, all five controllers earn the same $101.53 in the ten
  intervals above 1000 $/MWh (AEMO $100.56): the step-0 actual price reveals each
  spike and every controller held charge, so none needed to forecast it. The gap
  is made in ordinary arbitrage (100-300 $/MWh band: perfect $23.89 vs
  $17.22-$20.69) and negative prices. A spike model would protect the $101 by
  ensuring charge is held; it would not close the measured gap. See RESULTS.md
  Study 2.
- Outputs: results/forecast_summary.csv, results/mpc_household_J4_<name>.csv/.json,
  plots/forecast_study.html, plots/mpc_household_J4_<name>.html, models/price_lstm.pt.

## Plots (2026-09-17)

- `plots/forecast_study.html` is one page: KPI row; net profit per forecaster
  beside the accuracy-vs-value scatter; MAE / median AE / clipped MAE and error by
  lead time; cumulative profit and cumulative shortfall against perfect foresight
  with the 15 Jan spike marked; the 15 Jan case study (forecasts issued 04:04 and
  12:05 on an asinh price axis, actual price, SoC per controller); table view.
- Per-run pages (plotting/battery_plot.py: KPI row, an overview strip whose
  window drives the price / SoC / power panels, cumulative $ over the whole run,
  rainflow cycle-depth histogram) are drawn only with `--per-run`, and are not
  tracked; the one tracked per-run page is the headline perfect-foresight
  configuration, `plots/milp_household_J4_R12000.html`.
- Colours are fixed per forecaster in plotting/theme.py (perfect = ink, naive =
  aqua, AEMO = blue, LSTM = orange) so a hue means the same thing on every panel.
- `python forecast_trading.py --plot-only [--per-run]` redraws from results/
  without solving; `python milp_trading.py --plot-only [--per-run]` does the same
  for the degradation study (`plots/milp_j_sweep.html`).
