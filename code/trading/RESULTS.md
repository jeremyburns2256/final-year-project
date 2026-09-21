# Results to date

Written 2026-09-21. Every number is read from `results/*.csv`; nothing is re-run.
Design decisions behind each study are in `milp/MODEL_NOTES.md`, `FORECAST_NOTES.md`
and `retail/MODEL_NOTES.md`. All dollar figures are AUD for the 31 days of January
2025 (JAN25, NSW1, 8928 five-minute intervals) unless stated otherwise.

## 1. Setup common to all studies

- **Battery:** Powerwall 3, 13.5 kWh usable, 5 kW charge / 11.04 kW discharge,
  η_c = η_d = 0.943, E_0 = 6.75 kWh.
- **Household:** one house, net meter data only (B1 export, E1 import), so the model
  sees net local power G − A and never G and A separately. Without a battery the
  house imports 458 kWh and exports 490 kWh over the month.
- **Spot settlement:** import at RRP + N (Ausgrid EA010, 10.80 c/kWh), export at RRP.
- **Degradation:** Xu et al. (2018) J-segment cycle-aging cost inside the MILP;
  every run is also valued ex post by rainflow counting with Φ(δ) = 5.24e-4 δ^2.03
  and R_cell = 12 000 AUD. "Net profit" below always uses the rainflow figure.
- **January 2025 prices:** mean 83 $/MWh, median 76, negative in 15.2% of intervals,
  above 300 $/MWh in 0.45%, above 1000 $/MWh in **10 intervals** (1, 14, 15 and
  22 Jan; maximum 17 500 on 15 Jan).

**No-battery reference on spot pricing.** The household alone pays $101.84 for
imports and earns $10.53 for exports: a net position of **−$91.31** for the month.
Household results below are the household's whole net position, so the value the
battery adds is the result minus −$91.31.

## 2. Study 1 — perfect-foresight MILP and the cost of cycling

Rolling horizon, 48 h solved / 24 h committed, 31 windows, HiGHS.

### Household scenario

| Model | Net profit ex. deg. | Rainflow deg. | **Net profit** | Life lost | Equiv. full cycles | Model-vs-rainflow error | Solve (s) |
|---|---|---|---|---|---|---|---|
| J = 1, R = 0 (degradation ignored) | 95.38 | 194.76 | **−99.38** | 1.62% | 42.3 | – | 15 |
| J = 1 | 14.92 | 2.43 | 12.49 | 0.020% | 1.0 | 168% | 3 |
| J = 2 | 43.97 | 22.58 | 21.40 | 0.188% | 8.0 | 15.6% | 7 |
| **J = 4** | 43.84 | 15.10 | **28.74** | 0.126% | 9.3 | 4.2% | 13 |
| J = 8 | 47.99 | 16.65 | 31.34 | 0.139% | 12.7 | 1.1% | 27 |
| J = 16 | 49.01 | 16.62 | 32.39 | 0.139% | 15.0 | 4.1% | 63 |
| State machine (68.22 / 127.20, 20 kWh lossless) | 54.88 | 135.82 | −80.95 | 1.13% | 25.5 | – | – |

### Battery only (no household)

| Model | Net profit ex. deg. | Rainflow deg. | **Net profit** | Equiv. full cycles | Model-vs-rainflow error |
|---|---|---|---|---|---|
| J = 1, R = 0 | 136.23 | 139.31 | **−3.09** | 25.4 | – |
| J = 1 | 103.98 | 1.68 | 102.30 | 0.9 | 252% |
| J = 2 | 105.91 | 2.94 | 102.98 | 1.4 | 51% |
| J = 4 | 113.97 | 6.77 | 107.20 | 4.8 | 12.5% |
| J = 8 | 113.41 | 5.22 | 108.19 | 5.0 | 6.9% |
| J = 16 | 113.55 | 5.16 | 108.39 | 5.4 | 2.6% |

### Findings

1. **Ignoring degradation destroys the value of trading.** The R = 0 optimiser earns
   the most at the meter ($95.38 household, $136.23 battery only) but cycles the
   battery 42 and 25 times in a month, consuming 1.6% and 1.2% of its life. Valued
   by rainflow this is a net loss in both scenarios. The threshold state machine
   fails the same way (−$80.95): it earns more at the meter than any
   degradation-aware MILP and gives all of it back in aging.
2. **A degradation-aware controller trades about a fifth as often and keeps the
   profit.** At J = 4 the household battery does 9.3 equivalent full cycles, loses
   0.126% of life, and nets $28.74, i.e. **$120.05 of value added** over the
   no-battery −$91.31 after $15.10 of aging.
3. **J = 4 is the working resolution.** A single segment prices all cycling at one
   marginal cost and barely trades (1 cycle). Net profit rises steeply to J = 4 and
   then flattens (28.74 → 31.34 → 32.39 for a 2× and 5× longer solve). The MILP's
   internal degradation cost agrees with rainflow to 4% at J = 4 in the household
   case. The error is not monotone in J (1.1% at J = 8, 4.1% at J = 16), so beyond
   J = 4 it is within the noise of a single month.
4. **Co-located solar and load add to the arbitrage value.** Value added before
   aging is $135.16 with the household against $113.97 for the battery alone at
   J = 4. This is consistent with the battery charging from surplus solar that would
   otherwise be exported at RRP, and discharging into load that would otherwise
   pay RRP + N, so it earns the network tariff as well as the price spread.

## 3. Study 2 — forecast-driven MPC (dispatch value of forecasts)

Household, J = 4, R_cell = 12 000. 24 h horizon at 5-min resolution, re-planned every
interval (8928 solves per forecaster, 0.14–0.33 s each). The price of the current
interval is known (AEMO publishes it before the interval starts); every later price
and all net-local power, including the current interval, are forecast. Decisions are
settled against actual price and actual net-local power.

### Headline

| Forecaster | Price MAE ($/MWh) | Median AE | rMAE vs naive | Net profit ex. deg. | Rainflow deg. | **Net profit** | Gap to perfect | % of perfect |
|---|---|---|---|---|---|---|---|---|
| Perfect (price and load) | 0 | 0 | – | 43.85 | 15.11 | **28.74** | 0 | 100% |
| Perfect price, forecast load | 0 | 0 | – | 34.33 | 14.69 | **19.64** | 9.10 | 68% |
| LSTM | 44.6 | 22.0 | 0.73 | 33.29 | 17.35 | **15.94** | 12.80 | 55% |
| Seasonal naive | 61.3 | 25.3 | 1.00 | 31.71 | 15.79 | **15.92** | 12.82 | 55% |
| AEMO pre-dispatch | 161.6 | 16.4 | 2.63 | 30.24 | 15.57 | **14.67** | 14.07 | 51% |

Net-local forecast (7-day time-of-day profile, shared by all but Perfect): MAE 1.15 kW.
LSTM validation (2024): MAE 70.3 vs naive 103.1 $/MWh, rMAE 0.68, best epoch 5 of 10.

### Where the battery's value comes from

Value added over the no-battery household, before aging, by price band of the
interval in which it was earned:

| RRP band ($/MWh) | Intervals | Perfect | Perfect price | Naive | AEMO | LSTM |
|---|---|---|---|---|---|---|
| < 0 | 1411 | 2.35 | −0.49 | 0.15 | −0.25 | −0.07 |
| 0 – 50 | 1211 | −1.20 | −3.14 | −3.04 | −3.02 | −3.18 |
| 50 – 100 | 3928 | 2.02 | 1.40 | 1.44 | 1.37 | 0.79 |
| 100 – 300 | 2338 | 23.89 | 20.69 | 17.22 | 18.65 | 19.87 |
| 300 – 1000 | 30 | 6.57 | 5.65 | 5.71 | 4.24 | 5.66 |
| > 1000 | 10 | 101.53 | 101.53 | 101.53 | 100.56 | 101.53 |
| **Total** | 8928 | **135.16** | **125.64** | **123.01** | **121.55** | **124.60** |

### Findings

1. **The rolling loop itself costs nothing.** Perfect foresight on the 24 h / 5-min
   MPC loop reproduces the 48 h / 24 h result exactly ($28.74), so every gap in the
   table is forecast error, not horizon or re-planning.
2. **Most of the forecast gap is the household forecast, not the price forecast.**
   Knowing prices perfectly but forecasting load with the 7-day profile already
   loses $9.10 of the $12.8–14.1 total. Against that fair reference the price
   forecasts cost only $3.70 (LSTM), $3.72 (naive) and $4.97 (AEMO).
3. **A 27% better price forecast bought no extra dispatch value.** The LSTM cuts
   MAE from 61.3 to 44.6 $/MWh and finishes two cents ahead of the naive. It earns
   $1.59 more at the meter but cycles deeper (mean depth 0.196 vs 0.158) and pays
   $1.56 more in aging.
4. **Three quarters of the battery's value is earned in ten intervals, and every
   controller captures it.** $101.53 of the $135.16 is earned in the 50 minutes
   above 1000 $/MWh. All five controllers discharge at the full 11.04 kW in those
   intervals (AEMO sits out one, the 1051 $/MWh interval on 14 Jan, for $0.97),
   because the current price is observed and every controller happened to hold
   charge. **No forecaster predicted these spikes and none needed to.** The
   differences between forecasters are made in ordinary arbitrage: the 100–300
   band (perfect $23.89 vs $17.22–$20.69) and negative prices (perfect $2.35 vs
   about zero).
   *This corrects FORECAST_NOTES (2026-09-15), which attributed the unrealised
   ~$13 to missed spike revenue.*
5. **AEMO pre-dispatch is the best forecast in normal conditions and the worst on
   average.** Lowest median error (16.4) but MAE 162, because it forecasts
   market-cap spikes on the hot days (14–15, 21–22 Jan) that mostly did not
   eventuate: 1.0% of its forecast intervals exceed 1000 $/MWh against 0.1% of
   actuals. With spikes clipped at 1000 its MAE (48.9) is between the LSTM (44.7)
   and the naive (53.2). In dispatch it finishes $1.25 below the naive.
6. **Forecast accuracy and dispatch value are only loosely coupled here.** MAE
   ranks LSTM < naive < AEMO; net profit puts LSTM and naive level and AEMO just
   behind, all within $1.27.

### What this implies

- The assumption carrying the most weight is **"the current interval's price is
  known before dispatch"**. It is what secures the $101. It is realistic (AEMO
  publishes the dispatch price at the start of the interval) but it should be
  stated as an assumption and tested: `simulate_milp_mpc(step0_actual_price=False)`
  already exists and has not been run.
- The spike revenue was captured because the battery was not empty when each spike
  arrived. That is one month and four spike days. It is not evidence that a
  controller will generally be holding charge; a reserve-SoC policy or a
  spike-probability input addresses that risk, not the $13 gap.
- The largest recoverable loss is the **net-local forecast** ($9.10). A better
  household forecast (weather-driven solar, or simply using the latest meter
  reading for the current interval) is worth more than any further price-model work.

## 4. Study 3 — retail self-consumption (AGL Residential Smart Saver, Ausgrid)

Same battery, no optimiser: charge from surplus solar, discharge into load, never
trade with the grid. Feed-in 3 c/kWh. Aging valued ex post by rainflow.

| Plan | Bill without | Bill with | Gross saving | Rainflow deg. | **Net saving** | Simple payback |
|---|---|---|---|---|---|---|
| Single rate (29.82 c/kWh, 149.57 c/day) | 169.77 | 110.36 | 59.41 | 64.04 | **−4.64** | never |
| Time of use (54.18 / 21.63 c/kWh, 158.63 c/day) | 198.92 | 124.39 | 74.54 | 64.04 | **10.49** | 97 yr |

Dispatch is identical on both plans (the rule never looks at price): import falls
464 → 240 kWh, export 495 → 252 kWh, 16.1 equivalent full cycles, 0.53% of life.

### Findings

1. **The household is better off on the single-rate plan with or without the
   battery** ($169.77 vs $198.92 without; $110.36 vs $124.39 with). Plan choice is
   worth $29 a month; the battery's gross saving on the right plan is $59.
2. **The battery is worth more on the plan the household should not be on**
   ($74.54 vs $59.41 gross), because the imports it displaces are dearer there.
3. **On the single-rate plan the battery does not cover its modelled aging**
   ($59.41 gross vs $64.04). The supply charge is 27% of the bill and storage
   cannot touch it; the battery earns only on the 26.82 c/kWh spread between the
   usage rate and the feed-in tariff.
4. **Self-consumption ages the battery four times faster than degradation-aware
   spot trading** (0.53% vs 0.126% of life per month; 16.1 vs 9.3 equivalent full
   cycles) because the rule has no aging signal and cycles on every surplus.

## 5. Across the three operating models

Value added by the battery over the same household with no battery, January 2025:

| Operating model | Gross value added | Rainflow deg. | **Net value added** |
|---|---|---|---|
| Spot, perfect foresight, J = 4 | 135.16 | 15.11 | **120.05** |
| Spot, LSTM-driven MPC | 124.60 | 17.35 | **107.25** |
| Spot, naive-driven MPC | 123.01 | 15.79 | **107.23** |
| Spot, degradation ignored (J = 1, R = 0) | 186.69 | 194.76 | **−8.07** |
| Retail time of use, self-consumption | 74.54 | 64.04 | **10.49** |
| Retail single rate, self-consumption | 59.41 | 64.04 | **−4.64** |

On this month, a forecast-driven, degradation-aware controller on spot prices is
worth about $107 to the household against roughly zero for self-consumption on a
retail plan, and it achieves that with a quarter of the battery wear. About $100 of
that is the ten spike intervals. Take those out and the two are roughly level: the
LSTM-driven controller adds $23.07 gross and $5.72 after aging, against −$4.64 and
$10.49 on the retail plans. The case for spot exposure on this month is the spikes.

## 6. Limits on what can be claimed

- **One month, one house, one region.** January is the month with the most solar and
  contained two market-cap days. Spot results should not be annualised: 75% of the
  value is ten intervals. The retail annualisation in `retail_summary.csv` is an
  upper bound for the same reason.
- **Degradation inputs are placeholders.** R_cell = 12 000 AUD needs a cited
  installed-cost figure, and the Xu NMC stress function overstates aging for the
  Powerwall 3's LFP cells. The sign of the retail single-rate result and the size of
  every "net" figure depend on both. Gross figures are robust to them; the dispatch
  of the degradation-aware MILP is not.
- **Spot exposure is modelled without a retailer.** No subscription fee, no retail
  margin and no daily supply charge are included on the spot side, while the retail
  bills include a $46–49 supply charge. "Value added by the battery" is comparable
  across the two because each is measured against its own no-battery case; the
  absolute bills are not.
- **No grid import or export limit**, fixed efficiencies, no calendar aging, and
  the state machine baseline runs on its own 20 kWh lossless battery, so it is a
  qualitative comparison only.
- **The test month was touched once per forecaster.** LSTM hyperparameters were
  chosen on 2024; JAN25 results are out of sample.

## 7. Open items

1. Run the MPC study with `step0_actual_price=False` to price the current-interval
   assumption.
2. Improve the net-local forecast; it is the largest recoverable loss ($9.10).
3. Replace the placeholder R_cell and the NMC stress function, then re-run all three
   studies; the retail conclusion may change sign.
4. Extend beyond JAN25 (at least one winter month) before making any annual or
   payback claim for spot trading.
5. Decide whether a spike model is still a Part B candidate given Study 2 finding 4: its
   value would be in guaranteeing charge is held, not in closing the measured gap.

## Source files

`results/milp_summary.csv`, `results/forecast_summary.csv`, `results/retail_summary.csv`,
`results/mpc_household_J4_<forecaster>.csv` (price-band table and no-battery reference
are computed from these), `results/train_price_lstm.log`. Plots:
`plots/milp_j_sweep.html`, `plots/milp_household_J4_R12000.html`,
`plots/forecast_study.html`, `plots/retail_flat_E13.5.html`, `plots/retail_tou_E13.5.html`.
