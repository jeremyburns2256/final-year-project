"""
load_lstm.py

LSTM net-local forecaster after Kong et al. (2019), "Short-Term Residential Load
Forecasting Based on LSTM Recurrent Neural Network" (FORECAST_NOTES: load model).

    input   : previous LOOKBACK half-hours of [min-max net-local, one-hot slot of day,
              one-hot day of week, holiday flag]                      (Kong's E, I, D, H)
    network : 2 stacked LSTM layers x 20 units, final hidden state -> feedforward
    output  : next 48 half-hours of net-local, direct multi-output    (Kong: next half-hour only)
    loss    : MSE on the scaled series
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import HORIZON_SLOTS, SLOTS_PER_DAY, Forecaster, Frame

MODEL_DIR = "models"
LOOKBACK = SLOTS_PER_DAY        # 48 half-hours
N_FEATURES = 1 + SLOTS_PER_DAY + 7 + 1
# NSW public holidays inside the meter record (20 Feb 2024 .. 20 Feb 2025).
NSW_HOLIDAYS = pd.to_datetime([
    "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01", "2024-04-25", "2024-06-10",
    "2024-10-07", "2024-12-25", "2024-12-26", "2025-01-01", "2025-01-27",
])


@dataclass
class LoadLstmConfig:
    lookback: int = LOOKBACK
    horizon: int = HORIZON_SLOTS
    hidden: int = 20
    layers: int = 2
    head_hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 150
    patience: int = 20
    net_min: float = 0.0
    net_max: float = 1.0
    seed: int = 0


class LoadLSTM(nn.Module):
    def __init__(self, cfg: LoadLstmConfig, n_in: int = N_FEATURES):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(n_in, cfg.hidden, num_layers=cfg.layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, cfg.horizon),
        )

    def forward(self, x_hist):
        _, (h, _) = self.lstm(x_hist)                       # h: (layers, B, hidden)
        return self.head(h[-1])


class LoadWindows(torch.utils.data.Dataset):
    """Windows over a half-hourly feature matrix; target = next `horizon` scaled net-local values."""

    def __init__(self, feats: np.ndarray, target: np.ndarray, idx: np.ndarray, cfg: LoadLstmConfig):
        self.feats, self.target, self.idx, self.cfg = feats, target, idx, cfg

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]                                    # forecast issued at start of slot k
        L, H = self.cfg.lookback, self.cfg.horizon
        return torch.from_numpy(self.feats[k - L : k]), torch.from_numpy(self.target[k : k + H])


def load_net_local_half_hourly(export_csv: str = "data/export.csv", import_csv: str = "data/import.csv") -> pd.Series:
    """Full meter record as net-local kW (export - import), indexed by 30-min slot start."""
    df = pd.read_csv(export_csv).merge(pd.read_csv(import_csv), on="SETTLEMENTDATE")
    start = pd.to_datetime(df["SETTLEMENTDATE"], dayfirst=True) - pd.Timedelta(minutes=5)
    net = pd.Series((df["EXPORT_KW"] - df["IMPORT_KW"]).to_numpy(), index=pd.DatetimeIndex(start)).sort_index()
    hh = net.resample("30min", label="left", closed="left").mean()
    if hh.isna().any():
        raise ValueError(f"meter record has {int(hh.isna().sum())} empty half-hour slots")
    return hh


def net_scale(net_kw, cfg: LoadLstmConfig):
    return (np.asarray(net_kw, dtype=float) - cfg.net_min) / (cfg.net_max - cfg.net_min)


def net_inverse(e, cfg: LoadLstmConfig):
    return np.asarray(e, dtype=float) * (cfg.net_max - cfg.net_min) + cfg.net_min


def build_features(index: pd.DatetimeIndex, net_kw: np.ndarray, cfg: LoadLstmConfig):
    """Feature matrix (n, N_FEATURES) float32 and target (n,) = min-max net-local."""
    e = net_scale(net_kw, cfg).astype(np.float32)
    tod = np.eye(SLOTS_PER_DAY)[index.hour * 2 + index.minute // 30]
    dow = np.eye(7)[index.dayofweek]
    hol = index.normalize().isin(NSW_HOLIDAYS)[:, None]
    feats = np.concatenate([e[:, None], tod, dow, hol], axis=1).astype(np.float32)
    return feats, e


def save_model(model: LoadLSTM, cfg: LoadLstmConfig, name: str = "load_lstm", history: dict | None = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{name}.pt")
    with open(f"{MODEL_DIR}/{name}.json", "w") as f:
        json.dump({"config": asdict(cfg), "history": history or {}}, f, indent=2)


def load_model(name: str = "load_lstm") -> tuple[LoadLSTM, LoadLstmConfig]:
    with open(f"{MODEL_DIR}/{name}.json") as f:
        cfg = LoadLstmConfig(**json.load(f)["config"])
    model = LoadLSTM(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{name}.pt", map_location="cpu"))
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_matrix(model: LoadLSTM, cfg: LoadLstmConfig, feats: np.ndarray, issue_idx: np.ndarray,
                   batch_size: int = 512) -> np.ndarray:
    """Forecasts (len(issue_idx), horizon) in kW, one row per issue slot k (needs feats[k-L:k])."""
    out = []
    for s in range(0, len(issue_idx), batch_size):
        ks = issue_idx[s : s + batch_size]
        out.append(model(torch.from_numpy(np.stack([feats[k - cfg.lookback : k] for k in ks]))).numpy())
    return net_inverse(np.concatenate(out), cfg)


class LstmLoadForecaster(Forecaster):
    """
    Net-local: trained LSTM, forecast issued at the start of every half hour of the
    test period from the previous day of the frame's own 5-min data averaged to
    half hours, each value held for six 5-min steps. Price: actual, so the run
    isolates the household forecast (compare with perfect and perfect_price).
    """

    name = "lstm_load"

    def __init__(self, frame: Frame, model_name: str = "load_lstm"):
        super().__init__(frame)
        model, cfg = load_model(model_name)
        feats, _ = build_features(frame.slot_start_times(), frame.net_local_half_hourly(), cfg)
        n = frame.n_slots
        fc = np.full((n, HORIZON_SLOTS), np.nan)
        first = frame.slot_of[frame.test_start]
        issue = np.arange(max(first, cfg.lookback), n)
        fc[issue] = predict_matrix(model, cfg, feats, issue)
        self.hh_net_forecasts = fc

    def price(self, t, horizon):
        return self.frame.rrp[t : t + horizon].copy()

    def net_local(self, t, horizon):
        k = self.frame.slot_of[t]
        hh = self.hh_net_forecasts[k]
        if np.isnan(hh).any():
            raise ValueError(f"{self.name}: no half-hourly forecast for slot {k} (t={t})")
        return self.expand_half_hourly(hh, t, horizon)
