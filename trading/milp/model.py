"""
model.py

Single-window MILP for household battery dispatch — thesis Chapter 2 model.

Objective (Eq. 2.5), minimise over the window:
    sum_t [ D_i_t (N + R_t) dt  -  D_e_t R_t dt  +  sum_j c_j P_d_{t,j} dt ]

Every constraint below is annotated with its thesis equation number.
The household enters only through net local power  G_t - A_t  (kW), which is
export_kw - import_kw from the meter; the model never needs A_t and G_t separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pulp

from milp.degradation import PHI_A, PHI_K, segment_costs

INTERVAL_HOURS = 5 / 60  # thesis: NEM dispatch period, 5 minutes


@dataclass
class BatteryParams:
    """Battery, inverter, tariff and degradation parameters. Defaults: Tesla Powerwall 3 (thesis Table 3.1)."""

    e_max: float = 13.5            # kWh usable capacity
    e_min: float = 0.0             # kWh (usable capacity already includes the manufacturer reserve)
    p_max_charge: float = 5.0      # kW AC, Eq. 2.6d
    p_max_discharge: float = 11.04 # kW AC, Eq. 2.6e
    eta_c: float = 0.89 ** 0.5     # thesis Sec. 3.3.3: split the 89% round trip evenly
    eta_d: float = 0.89 ** 0.5
    e_initial: float = 13.5 * 0.5  # kWh, E_0 (50% SoC)
    r_cell: float = 12_000.0       # AUD replacement cost. PLACEHOLDER, needs a cited figure.
    n_segments: int = 1            # J cycle-depth segments; J = 1 is the linear model
    phi_a: float = PHI_A
    phi_k: float = PHI_K
    network_tariff: float = 0.108007  # $/kWh on imports, Ausgrid EA010
    export_limit_kw: float | None = None  # not applied when None (no grid limits agreed)
    import_limit_kw: float | None = None

    @property
    def segment_capacity(self) -> float:
        return self.e_max / self.n_segments  # E_j, thesis Sec. 2.3.1

    @property
    def segment_costs(self) -> np.ndarray:
        return segment_costs(self.r_cell, self.e_max, self.eta_d, self.n_segments, self.phi_a, self.phi_k)

    def initial_segment_energy(self, e_total: float | None = None) -> np.ndarray:
        """Split an aggregate SoC across segments, shallowest (cheapest) segment first. Eq. 2.6j."""
        e_total = self.e_initial if e_total is None else e_total
        out = np.zeros(self.n_segments)
        remaining = e_total
        for j in range(self.n_segments):
            out[j] = min(self.segment_capacity, remaining)
            remaining -= out[j]
        return out

    @classmethod
    def state_machine_equivalent(cls, **overrides) -> "BatteryParams":
        """The 20 kWh / 11.04 kW symmetric lossless battery used by the state machine, for like-for-like runs."""
        base = dict(e_max=20.0, p_max_charge=11.04, p_max_discharge=11.04, eta_c=1.0, eta_d=1.0, e_initial=10.0)
        base.update(overrides)
        return cls(**base)


@dataclass
class WindowResult:
    charge_kw: np.ndarray          # P_c_t
    discharge_kw: np.ndarray       # P_d_t
    grid_import_kw: np.ndarray     # D_i_t
    grid_export_kw: np.ndarray     # D_e_t
    soc_kwh: np.ndarray            # E_t = sum_j E_{t,j}, end of interval
    segment_soc_kwh: np.ndarray    # E_{t,j}, shape (T, J)
    degradation_cost: np.ndarray   # sum_j c_j P_d_{t,j} dt, per interval ($)
    objective: float
    status: str
    solve_seconds: float
    mip_gap: float | None = None
    binaries_at_bound: int = 0     # intervals where u_c + u_d = 1 (a sanity statistic)


def make_solver(name: str = "HiGHS", time_limit: float | None = 120.0, gap: float = 1e-4, msg: bool = False):
    """Return a PuLP solver. 'HiGHS' (default), 'GUROBI', or 'CBC'."""
    name = name.upper()
    if name == "HIGHS":
        return pulp.HiGHS(msg=msg, timeLimit=time_limit, gapRel=gap)
    if name == "GUROBI":
        return pulp.GUROBI(msg=msg, timeLimit=time_limit, gapRel=gap)
    if name == "CBC":
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap)
    raise ValueError(f"Unknown solver {name!r}; use HiGHS, GUROBI or CBC")


def build_and_solve_window(
    rrp: np.ndarray,
    net_local_kw: np.ndarray,
    params: BatteryParams,
    e_start_segments: np.ndarray,
    e_terminal_min: float,
    solver=None,
    dt: float = INTERVAL_HOURS,
) -> WindowResult:
    """
    Build and solve the thesis MILP over one window of T intervals.

    rrp              : $/MWh per interval, R_t
    net_local_kw     : G_t - A_t per interval (positive = household surplus)
    e_start_segments : E_{0,j} carried in from the previous window, Eq. 2.6j
    e_terminal_min   : lower bound on sum_j E_{T,j} at the window end, Eq. 2.6k
    """
    T = len(rrp)
    J = params.n_segments
    c = params.segment_costs
    price_kwh = np.asarray(rrp, dtype=float) / 1000.0  # $/MWh -> $/kWh
    N = params.network_tariff
    solver = solver or make_solver()

    prob = pulp.LpProblem("household_bess_dispatch", pulp.LpMinimize)

    t_idx = range(T)
    j_idx = range(J)
    Pc = pulp.LpVariable.dicts("Pc", (t_idx, j_idx), lowBound=0)          # P^c_{t,j}
    Pd = pulp.LpVariable.dicts("Pd", (t_idx, j_idx), lowBound=0)          # P^d_{t,j}
    E = pulp.LpVariable.dicts("E", (t_idx, j_idx), lowBound=0, upBound=params.segment_capacity)  # Eq. 2.6h
    Di = pulp.LpVariable.dicts("Di", t_idx, lowBound=0, upBound=params.import_limit_kw)   # D^i_t
    De = pulp.LpVariable.dicts("De", t_idx, lowBound=0, upBound=params.export_limit_kw)   # D^e_t
    uc = pulp.LpVariable.dicts("uc", t_idx, cat=pulp.LpBinary)             # Eq. 2.6g
    ud = pulp.LpVariable.dicts("ud", t_idx, cat=pulp.LpBinary)

    # Objective, Eq. 2.5
    prob += pulp.lpSum(
        Di[t] * (N + price_kwh[t]) * dt
        - De[t] * price_kwh[t] * dt
        + pulp.lpSum(c[j] * Pd[t][j] * dt for j in j_idx)
        for t in t_idx
    )

    for t in t_idx:
        pc_t = pulp.lpSum(Pc[t][j] for j in j_idx)  # Eq. 2.6b
        pd_t = pulp.lpSum(Pd[t][j] for j in j_idx)  # Eq. 2.6c

        prob += pc_t <= uc[t] * params.p_max_charge, f"charge_limit_{t}"       # Eq. 2.6d
        prob += pd_t <= ud[t] * params.p_max_discharge, f"discharge_limit_{t}" # Eq. 2.6e
        prob += uc[t] + ud[t] <= 1, f"mode_{t}"                                # Eq. 2.6f

        for j in j_idx:
            e_prev = e_start_segments[j] if t == 0 else E[t - 1][j]
            prob += (
                E[t][j] == e_prev + Pc[t][j] * params.eta_c * dt - Pd[t][j] / params.eta_d * dt
            ), f"soc_{t}_{j}"                                                   # Eq. 2.6a

        e_t = pulp.lpSum(E[t][j] for j in j_idx)
        prob += e_t >= params.e_min, f"soc_min_{t}"                             # Eq. 2.6i
        prob += e_t <= params.e_max, f"soc_max_{t}"

        # Eq. 2.8 rearranged: D_i - D_e = P_c - P_d - (G - A)
        prob += Di[t] - De[t] == pc_t - pd_t - float(net_local_kw[t]), f"balance_{t}"

    prob += pulp.lpSum(E[T - 1][j] for j in j_idx) >= e_terminal_min, "terminal_soc"  # Eq. 2.6k

    t0 = time.perf_counter()
    prob.solve(solver)
    elapsed = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        raise RuntimeError(f"MILP window did not solve to optimality: status={status}")

    def val(v):
        x = v.value()
        return 0.0 if x is None else float(x)

    seg = np.array([[val(E[t][j]) for j in j_idx] for t in t_idx])
    pd_arr = np.array([[val(Pd[t][j]) for j in j_idx] for t in t_idx])
    charge = np.array([sum(val(Pc[t][j]) for j in j_idx) for t in t_idx])
    discharge = pd_arr.sum(axis=1)
    deg = (pd_arr * c[None, :]).sum(axis=1) * dt
    binaries_on = int(sum(round(val(uc[t]) + val(ud[t])) for t in t_idx))

    return WindowResult(
        charge_kw=charge,
        discharge_kw=discharge,
        grid_import_kw=np.array([val(Di[t]) for t in t_idx]),
        grid_export_kw=np.array([val(De[t]) for t in t_idx]),
        soc_kwh=seg.sum(axis=1),
        segment_soc_kwh=seg,
        degradation_cost=deg,
        objective=float(pulp.value(prob.objective)),
        status=status,
        solve_seconds=elapsed,
        binaries_at_bound=binaries_on,
    )
