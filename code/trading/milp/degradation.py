"""
degradation.py

Cycle-depth stress function, piecewise-linear marginal aging costs and an
ex-post rainflow check, following Xu et al. (2018), "Factoring the Cycle Aging
Cost of Batteries Participating in Electricity Markets".

Thesis references: Eq. 2.3 (c_j), Eq. 3.8 (Phi), Section 6.2 (rainflow validation).

NOTE ON EQ. 2.3: the thesis as drafted omits a factor of J. Xu Eq. (5) is

    c_j = R * J / (eta_d * E_max) * [Phi(j/J) - Phi((j-1)/J)]      [$/kWh]

The J is required dimensionally: segment j delivers only eta_d * E_max / J kWh,
so the life consumed by that segment, R*[Phi(j/J)-Phi((j-1)/J)] dollars, must be
spread over E_max/J kWh, not E_max kWh. The corrected form is used here.
"""

from __future__ import annotations

import numpy as np

# Xu et al. (2018) Eq. (24), NMC 18650 cells. Thesis Eq. 3.8.
PHI_A = 5.24e-4
PHI_K = 2.03


def stress_function(delta, a: float = PHI_A, k: float = PHI_K):
    """Phi(delta): fraction of cell life consumed by one full cycle of depth delta in [0, 1]."""
    return a * np.power(delta, k)


def segment_costs(
    r_cell: float,
    e_max: float,
    eta_d: float,
    n_segments: int,
    a: float = PHI_A,
    k: float = PHI_K,
) -> np.ndarray:
    """
    Marginal aging cost c_j ($/kWh of AC discharge) for each of J evenly spaced
    cycle-depth segments. Corrected thesis Eq. 2.3 / Xu Eq. (5).

    Returns an array of length J, non-decreasing because Phi is convex.
    """
    j = np.arange(1, n_segments + 1)
    d_phi = stress_function(j / n_segments, a, k) - stress_function((j - 1) / n_segments, a, k)
    return r_cell * n_segments / (eta_d * e_max) * d_phi


def rainflow_life_loss(soc_kwh: np.ndarray, e_max: float, a: float = PHI_A, k: float = PHI_K) -> tuple[float, list]:
    """
    Ex-post cycle life loss from a SoC trajectory using rainflow counting (Xu Sec. II-C, Eq. 1).

    Each counted cycle of depth delta (range of the normalised SoC swing) contributes
    count * Phi(delta), where count is 0.5 for a half cycle and 1.0 for a full cycle.

    Returns (life_loss_fraction, cycles) where cycles is a list of
    (depth, mean, count, i_start, i_end) tuples.
    """
    import rainflow

    sigma = np.asarray(soc_kwh, dtype=float) / e_max
    cycles = list(rainflow.extract_cycles(sigma))
    loss = float(sum(count * stress_function(rng, a, k) for rng, _mean, count, _i0, _i1 in cycles))
    return loss, cycles


def rainflow_aging_cost(soc_kwh: np.ndarray, e_max: float, r_cell: float, a: float = PHI_A, k: float = PHI_K) -> dict:
    """
    Ex-post aging cost R * L and cycle statistics for a SoC trajectory.
    Used to validate the piecewise-linear model cost (Xu Eq. 26, thesis Sec. 6.2).
    """
    loss, cycles = rainflow_life_loss(soc_kwh, e_max, a, k)
    depths = np.array([c[0] for c in cycles]) if cycles else np.array([])
    counts = np.array([c[2] for c in cycles]) if cycles else np.array([])
    return {
        "life_loss_fraction": loss,
        "rainflow_cost": r_cell * loss,
        "n_cycles": float(counts.sum()) if len(counts) else 0.0,
        "mean_cycle_depth": float((depths * counts).sum() / counts.sum()) if len(counts) and counts.sum() > 0 else 0.0,
        "max_cycle_depth": float(depths.max()) if len(depths) else 0.0,
    }


def relative_error(model_cost: float, rainflow_cost: float) -> float:
    """Xu Eq. (26): |C_hat - R L| / (R L). NaN when the ex-post cost is zero."""
    if rainflow_cost <= 0:
        return float("nan")
    return abs(model_cost - rainflow_cost) / rainflow_cost
