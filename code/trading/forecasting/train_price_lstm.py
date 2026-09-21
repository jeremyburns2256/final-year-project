"""
Train the LSTM price forecaster (FORECAST_NOTES: train 2015-2023, validate 2024).

Run from the trading/ directory:
    python -m forecasting.train_price_lstm                # full training
    python -m forecasting.train_price_lstm --epochs 2 --stride 48   # smoke test

Writes models/price_lstm.pt and models/price_lstm.json, and prints validation
MAE in $/MWh against the seasonal-naive at 1 h and 24 h ahead.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting.base import SLOTS_PER_DAY
from forecasting.lstm import LstmConfig, PriceLSTM, SlidingWindows, build_features, save_model
from forecasting.nem_data import load_half_hourly, price_inverse

TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")


def split_indices(index: pd.DatetimeIndex, cfg: LstmConfig, stride: int = 1):
    """Issue-slot indices k for train and validation. A window is train if its target lies wholly before TRAIN_END."""
    n = len(index)
    k_all = np.arange(cfg.lookback, n - cfg.horizon + 1)
    target_end = index[k_all + cfg.horizon - 1]
    train = k_all[(target_end < TRAIN_END)][::stride]
    val = k_all[(index[k_all] >= TRAIN_END) & (target_end < VAL_END)][::max(stride, 1)]
    return train, val


def evaluate(model, loader, cfg, z_all, val_idx) -> dict:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xh, cf, y in loader:
            preds.append(model(xh, cf).numpy()); ys.append(y.numpy())
    p = price_inverse(np.concatenate(preds)); y = price_inverse(np.concatenate(ys))
    naive = price_inverse(np.stack([z_all[k - SLOTS_PER_DAY : k - SLOTS_PER_DAY + cfg.horizon] for k in val_idx]))
    ae, nae = np.abs(p - y), np.abs(naive - y)
    return {
        "val_mae": float(ae.mean()), "naive_mae": float(nae.mean()), "rmae": float(ae.mean() / nae.mean()),
        "val_mae_1h": float(ae[:, 1].mean()), "naive_mae_1h": float(nae[:, 1].mean()),
        "val_mae_24h": float(ae[:, -1].mean()), "naive_mae_24h": float(nae[:, -1].mean()),
        "val_huber": float(nn.functional.huber_loss(torch.from_numpy(np.concatenate(preds)),
                                                     torch.from_numpy(np.concatenate(ys)), delta=cfg.huber_delta)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="subsample training windows (1 = every half hour)")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--name", default="price_lstm")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    cfg = LstmConfig()
    if a.epochs: cfg.epochs = a.epochs
    if a.hidden: cfg.hidden = a.hidden
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    hh = load_half_hourly()
    print(f"half-hourly series {hh.index[0]} .. {hh.index[-1]}  n={len(hh)}  filled={hh.attrs['missing_slots_filled']}")
    train_mask = hh.index < TRAIN_END
    cfg.demand_mean = float(hh.loc[train_mask, "TOTALDEMAND"].mean())
    cfg.demand_std = float(hh.loc[train_mask, "TOTALDEMAND"].std())
    feats, cal, z = build_features(hh, cfg)
    train_idx, val_idx = split_indices(hh.index, cfg, a.stride)
    print(f"train windows {len(train_idx)}  val windows {len(val_idx)}  config {cfg}")

    dl = torch.utils.data.DataLoader
    train_loader = dl(SlidingWindows(feats, cal, z, train_idx, cfg), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = dl(SlidingWindows(feats, cal, z, val_idx, cfg), batch_size=1024, shuffle=False, num_workers=0)

    model = PriceLSTM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, cfg.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for xh, cf, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xh, cf), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        ev = evaluate(model, val_loader, cfg, z, val_idx)
        sched.step(ev["val_huber"])
        history.append({"epoch": ep, "train_huber": tot / nb, **ev})
        print(f"epoch {ep:3d}  train {tot / nb:.4f}  val huber {ev['val_huber']:.4f}  "
              f"val MAE {ev['val_mae']:.1f} (naive {ev['naive_mae']:.1f}, rMAE {ev['rmae']:.3f})  "
              f"1h {ev['val_mae_1h']:.1f}/{ev['naive_mae_1h']:.1f}  24h {ev['val_mae_24h']:.1f}/{ev['naive_mae_24h']:.1f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if ev["val_huber"] < best - 1e-4:
            best, bad = ev["val_huber"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model, cfg, a.name, {"epochs": history, "best_epoch": ep})
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {ep}, best epoch {ep - bad}")
                break
    model.load_state_dict(best_state)
    save_model(model, cfg, a.name, {"epochs": history, "best_epoch": int(np.argmin([h['val_huber'] for h in history])) + 1})
    print(f"saved models/{a.name}.pt  best val huber {best:.4f}")


if __name__ == "__main__":
    main()
