"""
Train the LSTM net-local forecaster (FORECAST_NOTES: load model; train 20 Feb - Oct 2024,
validate Nov - Dec 2024, test JAN25).

Run from the trading/ directory:
    python -m forecasting.train_load_lstm                 # full training, then the JAN25 comparison
    python -m forecasting.train_load_lstm --epochs 2      # smoke test
    python -m forecasting.train_load_lstm --eval-only     # JAN25 comparison of the saved model

Writes models/load_lstm.pt and models/load_lstm.json, and prints net-local MAE in
kW against the 7-day profile on the validation months and on JAN25.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import INTERVALS_PER_HOUR, SLOTS_PER_DAY, load_frame
from forecasting.load_lstm import (LoadLSTM, LoadLstmConfig, LoadWindows, LstmLoadForecaster, build_features,
                                   load_net_local_half_hourly, net_inverse, save_model)
from forecasting.naive import PerfectPriceForecaster

TRAIN_END = pd.Timestamp("2024-11-01")
VAL_END = pd.Timestamp("2025-01-01")
PROFILE_DAYS = 7


def split_indices(index: pd.DatetimeIndex, cfg: LoadLstmConfig):
    """Issue-slot indices k for train and validation. A window is train if its target lies wholly before TRAIN_END."""
    n = len(index)
    k_all = np.arange(max(cfg.lookback, PROFILE_DAYS * SLOTS_PER_DAY), n - cfg.horizon + 1)
    target_end = index[k_all + cfg.horizon - 1]
    train = k_all[target_end < TRAIN_END]
    val = k_all[(index[k_all] >= TRAIN_END) & (target_end < VAL_END)]
    return train, val


def evaluate(model, loader, cfg, net_kw, val_idx) -> dict:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xh, y in loader:
            preds.append(model(xh).numpy()); ys.append(y.numpy())
    p = net_inverse(np.concatenate(preds), cfg); y = net_inverse(np.concatenate(ys), cfg)
    days = SLOTS_PER_DAY * np.arange(1, PROFILE_DAYS + 1)
    profile = np.stack([net_kw[k + np.arange(cfg.horizon)[:, None] - days[None, :]].mean(axis=1) for k in val_idx])
    ae, pae = np.abs(p - y), np.abs(profile - y)
    return {
        "val_mae": float(ae.mean()), "profile_mae": float(pae.mean()), "rmae": float(ae.mean() / pae.mean()),
        "val_mae_30min": float(ae[:, 0].mean()), "profile_mae_30min": float(pae[:, 0].mean()),
        "val_mae_24h": float(ae[:, -1].mean()), "profile_mae_24h": float(pae[:, -1].mean()),
        "val_mse": float(nn.functional.mse_loss(torch.from_numpy(np.concatenate(preds)), torch.from_numpy(np.concatenate(ys)))),
    }


def test_comparison(name: str, horizon: int = 288) -> pd.DataFrame:
    """
    JAN25 net-local error at 5-min resolution as the MPC sees it (forecasts issued every
    half hour, as forecast_trading.forecast_errors), LSTM against the 7-day profile.
    """
    frame = load_frame()
    fcs = {"profile": PerfectPriceForecaster(frame), "lstm": LstmLoadForecaster(frame, name)}
    issues = range(frame.test_start, frame.n - horizon + 1, 6)
    actual = np.stack([frame.net_local[t : t + horizon] for t in issues])
    leads = {"30 min": slice(0, 6), "1 h": slice(6, 12), "3 h": slice(30, 36), "6 h": slice(66, 72),
             "12 h": slice(138, 144), "24 h": slice(282, 288)}
    rows = {}
    for label, fc in fcs.items():
        e = np.stack([fc.net_local(t, horizon) for t in issues]) - actual
        rows[label] = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias": e.mean(),
                       "MAE first hour": np.abs(e[:, :INTERVALS_PER_HOUR]).mean(),
                       **{f"MAE {k}": np.abs(e[:, s]).mean() for k, s in leads.items()}}
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--name", default="load_lstm")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true", help="skip training; JAN25 comparison of the saved model")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    pd.set_option("display.width", 250)
    if a.eval_only:
        print(test_comparison(a.name).round(3).to_string())
        return

    cfg = LoadLstmConfig()
    if a.epochs: cfg.epochs = a.epochs
    if a.hidden: cfg.hidden = a.hidden
    if a.lookback: cfg.lookback = a.lookback
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    hh = load_net_local_half_hourly()
    print(f"half-hourly net-local {hh.index[0]} .. {hh.index[-1]}  n={len(hh)}")
    train_net = hh[hh.index < TRAIN_END]
    cfg.net_min, cfg.net_max = float(train_net.min()), float(train_net.max())
    net_kw = hh.to_numpy()
    feats, e = build_features(hh.index, net_kw, cfg)
    train_idx, val_idx = split_indices(hh.index, cfg)
    print(f"train windows {len(train_idx)}  val windows {len(val_idx)}  config {cfg}")

    dl = torch.utils.data.DataLoader
    train_loader = dl(LoadWindows(feats, e, train_idx, cfg), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = dl(LoadWindows(feats, e, val_idx, cfg), batch_size=1024, shuffle=False, num_workers=0)

    model = LoadLSTM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, cfg.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for xh, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xh), y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        ev = evaluate(model, val_loader, cfg, net_kw, val_idx)
        history.append({"epoch": ep, "train_mse": tot / nb, **ev})
        print(f"epoch {ep:3d}  train {tot / nb:.5f}  val mse {ev['val_mse']:.5f}  "
              f"val MAE {ev['val_mae']:.3f} kW (profile {ev['profile_mae']:.3f}, rMAE {ev['rmae']:.3f})  "
              f"30min {ev['val_mae_30min']:.3f}/{ev['profile_mae_30min']:.3f}  "
              f"24h {ev['val_mae_24h']:.3f}/{ev['profile_mae_24h']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if ev["val_mse"] < best - 1e-6:
            best, bad = ev["val_mse"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {ep}, best epoch {ep - bad}")
                break
    model.load_state_dict(best_state)
    save_model(model, cfg, a.name, {"epochs": history, "best_epoch": int(np.argmin([h["val_mse"] for h in history])) + 1})
    print(f"saved models/{a.name}.pt  best val mse {best:.5f}")
    print("\nJAN25 net-local error, kW (5-min resolution, 24 h forecasts issued every half hour)")
    print(test_comparison(a.name).round(3).to_string())


if __name__ == "__main__":
    main()
