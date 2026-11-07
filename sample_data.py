"""Synthetic sample datasets for trying the app. Run `python sample_data.py` to write them as CSVs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SAMPLES = {
    "Smooth daily demand (2 years, weekly pattern)": "Steady daily values with a weekly cycle and mild growth.",
    "Erratic daily demand": "Demand every day, but order sizes swing widely.",
    "Intermittent daily demand": "Most days are zero; when demand occurs it is a similar size.",
    "Lumpy daily demand": "Most days are zero and order sizes vary enormously.",
    "Hourly data (needs aggregation to daily)": "120 days of hourly readings with a few missing hours.",
    "Stock-like daily price (business days only)": "A random-walk price; weekends are missing, so pick a fill method.",
    "Monthly seasonal demand (6 years)": "Monthly values with yearly seasonality and a trend.",
}


def make_sample(name: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if name.startswith("Smooth"):
        t = pd.date_range("2024-01-01", periods=730, freq="D")
        weekly = np.array([0, 3, 5, 6, 8, 14, 10])[t.dayofweek]
        v = 100 + 0.04 * np.arange(len(t)) + weekly + rng.normal(0, 4, len(t))
    elif name.startswith("Erratic"):
        t = pd.date_range("2025-01-01", periods=365, freq="D")
        v = rng.lognormal(mean=3.0, sigma=0.9, size=len(t))
    elif name.startswith("Intermittent"):
        t = pd.date_range("2025-01-01", periods=365, freq="D")
        v = np.where(rng.random(len(t)) < 0.18, 5 + rng.poisson(2, len(t)), 0).astype(float)
    elif name.startswith("Lumpy"):
        t = pd.date_range("2025-01-01", periods=365, freq="D")
        v = np.where(rng.random(len(t)) < 0.15, rng.lognormal(3.0, 1.0, len(t)), 0.0)
    elif name.startswith("Hourly"):
        t = pd.date_range("2026-03-01", periods=120 * 24, freq="h")
        hour = t.hour.to_numpy()
        v = 20 + 15 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 3, len(t))
        v = np.round(np.maximum(v, 0), 2)
        keep = rng.random(len(t)) > 0.03   # ~3% of hours missing
        return pd.DataFrame({"timestamp": t[keep].strftime("%Y-%m-%d %H:%M:%S"), "value": v[keep]})
    elif name.startswith("Stock"):
        t = pd.bdate_range("2023-01-02", periods=780)
        v = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(t))))
        v = np.round(v, 2)
    elif name.startswith("Monthly"):
        t = pd.date_range("2020-01-01", periods=72, freq="MS")
        season = 20 * np.sin((t.month.to_numpy() - 3) / 12 * 2 * np.pi)
        v = 200 + 1.5 * np.arange(len(t)) + season + rng.normal(0, 6, len(t))
    else:
        raise KeyError(name)
    return pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d"), "value": np.round(v, 2)})


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(out, exist_ok=True)
    for nm in SAMPLES:
        fname = nm.split(" (")[0].lower().replace(" ", "_").replace("-", "_") + ".csv"
        make_sample(nm).to_csv(os.path.join(out, fname), index=False)
        print("wrote", fname)
