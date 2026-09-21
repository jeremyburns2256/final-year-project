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
| E_max | 13.5 kWh | Powerwall 3, Table 3.1 |
| E_min | 0 | usable capacity already includes the reserve |
| E_0 | 6.75 kWh (50%) | agreed |
| P_max^c / P_max^d | 5 / 11.04 kW | Table 3.1 |
| eta_c = eta_d | sqrt(0.89) = 0.943 | Section 3.3.3 |
| R_cell | 12 000 AUD | **placeholder**, needs a cited installed-cost figure |
| N | 0.108007 $/kWh | Ausgrid EA010 |
| Phi(delta) | 5.24e-4 * delta^2.03 | Eq. 3.8 (NMC, overstates LFP penalty) |

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
- One charge variable capped at P_max^c regardless of source. The Powerwall 3's
  DC-coupled solar path (faster, 89% vs 97.5% path efficiency) is not modelled,
  and cannot be separated with net-meter data anyway.
- Fixed efficiencies, no temperature or calendar aging.
- The state machine baseline is run with the Table 5.1 thresholds on its own
  20 kWh lossless battery; its degradation cost is computed ex post by rainflow.
