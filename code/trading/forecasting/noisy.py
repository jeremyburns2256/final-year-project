"""
noisy.py

Synthetic forecast error on top of perfect foresight, for the noise sensitivity
study (FORECAST_NOTES: noise sensitivity).

Each forecast issued at t is the actual path plus a smooth error path that grows
with lead time h (h = 0 is interval t):

    e_h      AR(1) along the horizon, e_h = rho e_{h-1} + sqrt(1 - rho^2) eps_h,
             stationary N(0, 1), so errors persist for ~1/(1 - rho) steps
    sigma(h) = sigma_max * sqrt((h + 1) / 288), sigma_max reached at 24 h ahead

    price     : 100 sinh( asinh(R/100) + sigma(h) e_h ), the LSTM's target
                transform, roughly additive near $0 and multiplicative on spikes
    net-local : (G - A) + sigma(h) e_h, in kW

The error path is redrawn for every issue time t from a generator seeded on
(seed, t), so the forecaster is deterministic and price(t, .) returns the same
path however often it is called. Noise is added to price or net-local, not both:
the other input stays perfect so each sweep attributes the loss to one forecast.
"""

from __future__ import annotations

import numpy as np

from forecasting.base import INTERVALS_PER_DAY, Forecaster, Frame

PRICE, NET_LOCAL = 0, 1


class NoisyForecaster(Forecaster):
    def __init__(self, frame: Frame, price_sigma: float = 0.0, net_sigma_kw: float = 0.0,
                 rho: float = 0.95, seed: int = 0, name: str | None = None):
        super().__init__(frame)
        self.price_sigma = price_sigma
        self.net_sigma_kw = net_sigma_kw
        self.rho = rho
        self.seed = seed
        self.name = name or f"noise_p{price_sigma:g}_n{net_sigma_kw:g}"

    def error_path(self, t: int, horizon: int, which: int) -> np.ndarray:
        """sigma(h) e_h / sigma_max for h = 0 .. horizon-1 (unit sigma_max)."""
        rng = np.random.default_rng([self.seed, which, t])
        eps = rng.standard_normal(horizon)
        e = np.empty(horizon)
        e[0] = eps[0]
        k = np.sqrt(1 - self.rho ** 2)
        for h in range(1, horizon):
            e[h] = self.rho * e[h - 1] + k * eps[h]
        return np.sqrt((np.arange(horizon) + 1) / INTERVALS_PER_DAY) * e

    def price(self, t, horizon):
        actual = self.frame.rrp[t : t + horizon]
        if self.price_sigma == 0:
            return actual.copy()
        z = np.arcsinh(actual / 100.0) + self.price_sigma * self.error_path(t, len(actual), PRICE)
        return 100.0 * np.sinh(z)

    def net_local(self, t, horizon):
        actual = self.frame.net_local[t : t + horizon]
        if self.net_sigma_kw == 0:
            return actual.copy()
        return actual + self.net_sigma_kw * self.error_path(t, len(actual), NET_LOCAL)
