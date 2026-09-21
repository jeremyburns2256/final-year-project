"""
lstm.py

LSTM price forecaster (FORECAST_NOTES: price model).

    input   : previous LOOKBACK half-hours of [asinh(RRP/100), z(TOTALDEMAND), 6 calendar]
    output  : next 48 half-hours of asinh(RRP/100), direct multi-output
    head    : [LSTM final hidden state, calendar features of the 48 target slots] -> MLP -> 48
    loss    : Huber on the transformed scale
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import HORIZON_SLOTS, SLOTS_PER_DAY, Forecaster, Frame, HalfHourlyPriceMixin
from forecasting.nem_data import calendar_features, price_inverse, price_transform

MODEL_DIR = "models"
LOOKBACK = 7 * SLOTS_PER_DAY   # 336 half-hours


@dataclass
class LstmConfig:
    lookback: int = LOOKBACK
    horizon: int = HORIZON_SLOTS
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    head_hidden: int = 128
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5
    huber_delta: float = 1.0
    demand_mean: float = 0.0
    demand_std: float = 1.0
    seed: int = 0


class PriceLSTM(nn.Module):
    def __init__(self, cfg: LstmConfig, n_in: int = 8, n_cal: int = 6):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(n_in, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                            dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.horizon * n_cal, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.horizon),
        )

    def forward(self, x_hist, cal_future):
        _, (h, _) = self.lstm(x_hist)                       # h: (layers, B, hidden)
        z = torch.cat([h[-1], cal_future.flatten(1)], dim=1)
        return self.head(z)


class SlidingWindows(torch.utils.data.Dataset):
    """Windows over a half-hourly feature matrix; target = next `horizon` transformed prices."""

    def __init__(self, feats: np.ndarray, cal: np.ndarray, target: np.ndarray, idx: np.ndarray, cfg: LstmConfig):
        self.feats, self.cal, self.target, self.idx, self.cfg = feats, cal, target, idx, cfg

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]                                    # forecast issued at start of slot k
        L, H = self.cfg.lookback, self.cfg.horizon
        return (torch.from_numpy(self.feats[k - L : k]),
                torch.from_numpy(self.cal[k : k + H]),
                torch.from_numpy(self.target[k : k + H]))


def build_features(hh: pd.DataFrame, cfg: LstmConfig):
    """Feature matrix (n, 8) float32, calendar (n, 6), target (n,) = asinh price."""
    z = price_transform(hh["RRP"].to_numpy()).astype(np.float32)
    d = ((hh["TOTALDEMAND"].to_numpy() - cfg.demand_mean) / cfg.demand_std).astype(np.float32)
    cal = calendar_features(hh.index)
    feats = np.concatenate([z[:, None], d[:, None], cal], axis=1).astype(np.float32)
    return feats, cal, z


def save_model(model: PriceLSTM, cfg: LstmConfig, name: str = "price_lstm", history: dict | None = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{name}.pt")
    with open(f"{MODEL_DIR}/{name}.json", "w") as f:
        json.dump({"config": asdict(cfg), "history": history or {}}, f, indent=2)


def load_model(name: str = "price_lstm") -> tuple[PriceLSTM, LstmConfig]:
    with open(f"{MODEL_DIR}/{name}.json") as f:
        cfg = LstmConfig(**json.load(f)["config"])
    model = PriceLSTM(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{name}.pt", map_location="cpu"))
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_matrix(model: PriceLSTM, cfg: LstmConfig, feats: np.ndarray, cal: np.ndarray, issue_idx: np.ndarray,
                   batch_size: int = 512) -> np.ndarray:
    """Forecasts (len(issue_idx), horizon) in $/MWh, one row per issue slot k (needs feats[k-L:k])."""
    out = []
    for s in range(0, len(issue_idx), batch_size):
        ks = issue_idx[s : s + batch_size]
        xh = torch.from_numpy(np.stack([feats[k - cfg.lookback : k] for k in ks]))
        cf = torch.from_numpy(np.stack([cal[k : k + cfg.horizon] for k in ks]))
        out.append(model(xh, cf).numpy())
    return price_inverse(np.concatenate(out))


class LstmPriceForecaster(HalfHourlyPriceMixin, Forecaster):
    """
    Price: trained LSTM, forecast issued at the start of every half hour of the
    test period using the previous 7 days of the frame's own 5-min data averaged
    to half hours. Net-local: 7-day profile.
    """

    name = "lstm"

    def __init__(self, frame: Frame, model_name: str = "price_lstm"):
        super().__init__(frame)
        model, cfg = load_model(model_name)
        hh = pd.DataFrame(
            {"RRP": frame.rrp_half_hourly(), "TOTALDEMAND": frame.demand_half_hourly()},
            index=frame.slot_start_times(),
        )
        # Calendar features are needed HORIZON slots past the end of the frame.
        ext_index = pd.date_range(hh.index[0], periods=len(hh) + cfg.horizon, freq="30min")
        feats, _, _ = build_features(hh, cfg)
        cal = calendar_features(ext_index)
        n = frame.n_slots
        fc = np.full((n, HORIZON_SLOTS), np.nan)
        first = frame.slot_of[frame.test_start]
        issue = np.arange(max(first, cfg.lookback), n)
        fc[issue] = predict_matrix(model, cfg, feats, cal, issue)
        self.hh_forecasts = fc

    def net_local(self, t, horizon):
        return self.seven_day_profile(t, horizon)
