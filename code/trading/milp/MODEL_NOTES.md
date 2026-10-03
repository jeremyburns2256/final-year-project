# MILP implementation notes and thesis model changes

Decisions from the 2026-09-07 design session. Each item is either a correction
to the Chapter 2 model, a value the thesis names but does not assign, or a
simplification the code makes that the thesis should state.

## Corrections

1. **Eq. 2.3 is missing a factor of J.** Xu et al. (2018) Eq. (5) is
   `c_j = R * J / (eta_d * E_max) * [Phi(j/J) - Phi((j-1)/J)]`.
   Segment j delivers only `eta_d * E_max / J` kWh, so its share of the
   replacement cost must be spread over `E_max / J` kWh. Without the J every
   J > 1 model under-prices degradation by a factor of J and the J-sweep in
   Section 6.2 is not comparable across J. Implemented in `degradation.segment_costs`.

2. **The household enters through one net term.** The meter provides B1
   (net export) and E1 (import), never A_t and G_t separately. Eq. 2.8 is
   unchanged but the model only ever uses `G_t - A_t = export_kw - import_kw`.
   The nomenclature / Section 2.3.4 should say the agent observes net local
   power, not A and G individually.

## Values assigned

| Symbol | Value | Source / status |
|---|---|---|
Since 2026-10-03 the battery is a generic lithium-ion battery with the test
parameters of Xu et al. (2018) Sec. V-A, not a Tesla Powerwall 3. The size and
inverter limits are unchanged; efficiency, SoC window and chemistry follow Xu.

| Symbol | Value | Source / status |
|---|---|---|
| E_rate | 13.5 kWh | rated capacity, household scale (unchanged) |
| E_min / E_max | 15% / 95% of E_rate = 2.025 / 12.825 kWh | Xu Sec. V-A |
| E_0 | 6.75 kWh (50%) | agreed |
| P_max^c / P_max^d | 5 / 11.04 kW | unchanged |
| eta_c = eta_d | 0.95 | Xu Sec. V-A |
| R_cell | 800 AUD/kWh x E_rate = 10 800 AUD | installed cost, see below |
| N | 0.108007 $/kWh | Ausgrid EA010 |
| Phi(delta) | 5.24e-4 * delta^2.03 | Eq. 3.8, Xu Eq. (24), NMC 18650 cells |

Xu's remaining test parameters are consistent with these but not used by the
model: 3000 cycles at 80% depth is Phi(0.8) = 3.33e-4 = 1/3003; the 10 year
shelf life and 25 degC cell temperature only matter to calendar aging, which is
not modelled.

**E_rate vs E_max.** With a SoC window the rated capacity and the upper SoC
bound are different numbers. Following Xu, the segments are E_rate / J wide,
c_j divides by E_rate, and rainflow depth is a fraction of E_rate; E_min and
E_max only bound the aggregate SoC (Eq. 2.6i). The deepest possible cycle is
therefore 80%. In code: `BatteryParams.e_rated`, `soc_min`, `soc_max`, with
`e_min` / `e_max` derived.

**R_cell.** Set per kWh (`R_CELL_PER_KWH` in `model.py`) so it scales with
capacity. 800 AUD/kWh is the Solar Choice Battery Price Index, August 2026:
average installed price including installation and GST, before the federal
rebate, 778 (battery only) to 828 (with inverter) AUD/kWh at 10 kWh and 749 to
789 at 20 kWh. Two caveats: the index page blocked automated access, so these
figures came through a search summary and **need checking against the page**
(https://www.solarchoice.net.au/solar-batteries/price/); and Xu's R is the
battery pack replacement cost (300 USD/kWh), whereas an installed price also
covers labour and balance of system, so 800 is an upper bound on the cost of
replacing the cells.

## Solution method

- Perfect foresight, rolling horizon: 48 h solved, 24 h committed, per-segment
  energy `E_{t,j}` carried between windows. Eq. 2.6k (`sum_j E_T,j >= E_0`) is
  applied at the end of each solved window, so the committed day is free to
  carry charge into the next day; it binds for real only at the end of the data.
- Binaries u^c, u^d are always enforced. They are necessary in the NEM because
  negative prices plus round-trip losses would otherwise reward simultaneous
  charge and discharge.
- Solver: HiGHS through PuLP by default; Gurobi selectable (`--solver GUROBI`)
  once an academic licence is installed.
- Ex-post validation: rainflow count of the SoC trajectory, life loss
  `L = sum count * Phi(depth)`, cost `R_cell * L`, relative error per Xu Eq. (26).

## Stated simplifications

- No grid import or export limit (parameters exist, default off).
- One charge variable capped at P_max^c regardless of source. A separate
  DC-coupled solar path is not modelled, and cannot be separated with net-meter
  data anyway.
- Fixed efficiencies, no temperature or calendar aging.
- The state machine baseline is run with the Table 5.1 thresholds on its own
  20 kWh lossless battery; its degradation cost is computed ex post by rainflow.
