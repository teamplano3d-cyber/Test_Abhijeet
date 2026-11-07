"""Core time-series forecasting logic. No Streamlit imports, so it can be tested on its own.

Pipeline (see run_forecast):
    parse -> detect/validate frequency -> aggregate to target frequency -> fill missing
    -> (optional) cap outliers -> classify demand pattern -> auto-select model -> forecast
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter

# --------------------------------------------------------------------------------------
# Options exposed to the UI
# --------------------------------------------------------------------------------------
TARGETS = ("Daily", "Weekly", "Monthly")

TARGET_CFG = {
    # rank: coarseness; alias: pandas date_range freq; season: seasonal period;
    # min_obs: minimum history (periods); max_h: max forecast horizon
    "Daily": dict(rank=1, alias="D", season=7, min_obs=30, max_h=365, default_h=30, unit="day"),
    "Weekly": dict(rank=2, alias="7D", season=52, min_obs=20, max_h=104, default_h=12, unit="week"),
    "Monthly": dict(rank=3, alias="MS", season=12, min_obs=12, max_h=36, default_h=6, unit="month"),
}

AGG_OPTIONS = {
    "Sum": "sum",
    "Average (mean)": "mean",
    "Median": "median",
    "Mode (most frequent value)": "mode",
    "Minimum": "min",
    "Maximum": "max",
    "First value": "first",
    "Last value": "last",
}

MISSING_OPTIONS = {
    "Fill with zero": "zero",
    "Forward fill (carry last value forward)": "ffill",
    "Backward fill (use next value)": "bfill",
    "Fill with mean": "mean",
    "Fill with median": "median",
    "Linear interpolation": "linear",
}

OUTLIER_OPTIONS = ("Simple (use data as is)", "Robust (cap outliers)")

MAX_FIT = 1000   # only the most recent N periods are used to fit models (keeps the app fast)
MIN_TRAIN = 8    # smallest training window used in the backtest

# Syntetos-Boylan classification cut-offs
ADI_CUT = 1.32
CV2_CUT = 0.49

CATEGORY_TEXT = {
    "Smooth": "Values occur in most periods and their size is stable (low ADI, low CV²). The easiest pattern to forecast.",
    "Erratic": "Values occur in most periods but their size varies a lot (low ADI, high CV²).",
    "Intermittent": "Many periods are zero, but non-zero values are of fairly stable size (high ADI, low CV²).",
    "Lumpy": "Many periods are zero and non-zero values vary a lot in size (high ADI, high CV²). The hardest pattern to forecast.",
    "No demand": "Every period is zero, so there is nothing to learn from. The forecast is zero.",
}


class ValidationError(Exception):
    """Raised with a user-friendly message when the input cannot be processed."""


# --------------------------------------------------------------------------------------
# 1. Reading and validating input
# --------------------------------------------------------------------------------------
def guess_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Best-guess (timestamp column, value column) for a freshly uploaded table."""
    cols = list(df.columns)
    hints = ("date", "time", "timestamp", "period", "day", "month", "ds")
    ts = next((c for c in cols if any(h in str(c).lower() for h in hints)), None)
    if ts is None:
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            sample = df[c].dropna().head(50)
            if len(sample) and pd.to_datetime(sample, errors="coerce", utc=True).notna().mean() >= 0.8:
                ts = c
                break
    if ts is None:
        ts = cols[0]
    val = None
    for c in cols:
        if c == ts:
            continue
        sample = df[c].dropna().head(200)
        if len(sample) and pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.8:
            val = c
            break
    if val is None:
        val = next((c for c in cols if c != ts), cols[-1])
    return ts, val


def _parse_timestamps(s: pd.Series, dayfirst: bool) -> pd.Series:
    """Parse to naive datetimes. A single UTC offset (e.g. +05:30) is dropped, keeping the wall-clock
    time as written, so a local-midnight daily series is not shifted onto the previous day. Only data
    with *mixed* offsets is converted to UTC."""
    try:
        ts = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        if not pd.api.types.is_datetime64_any_dtype(ts):   # older pandas returns object for mixed offsets
            raise ValueError("mixed offsets")
    except (ValueError, TypeError):
        ts = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=True)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return ts


def parse_input(df: pd.DataFrame, ts_col: str, val_col: str, dayfirst: bool = False):
    """Return (raw series indexed by timestamp, info dict). Rows with unparseable timestamps are dropped."""
    if ts_col == val_col:
        raise ValidationError("The timestamp column and the value column must be different.")
    work = pd.DataFrame({"ts": df[ts_col].to_numpy(), "val": df[val_col].to_numpy()})
    n_rows = len(work)
    if n_rows == 0:
        raise ValidationError("The file has no rows.")
    if pd.api.types.is_numeric_dtype(work["ts"]):
        raise ValidationError(
            f"Column '{ts_col}' contains plain numbers, not dates. Pick the column that holds the "
            "timestamps (e.g. 2026-11-07 or 07/11/2026)."
        )

    ts = _parse_timestamps(work["ts"], dayfirst)
    if pd.api.types.is_numeric_dtype(work["val"]):
        val = pd.to_numeric(work["val"], errors="coerce")
    else:
        cleaned = work["val"].astype(str).str.replace(",", "", regex=False).str.strip()
        val = pd.to_numeric(cleaned, errors="coerce")
    val = val.replace([np.inf, -np.inf], np.nan)

    ok = ts.notna()
    n_bad_ts = int((~ok).sum())
    if not ok.any():
        raise ValidationError(
            f"No valid dates found in '{ts_col}'. Check that you picked the right column and the day-first setting."
        )
    ts, val = ts[ok], val[ok]
    if val.notna().sum() == 0:
        raise ValidationError(f"Column '{val_col}' has no numeric values.")

    raw = pd.Series(
        val.to_numpy(dtype=float),
        index=pd.DatetimeIndex(ts.to_numpy()).astype("datetime64[ns]"),
        name="value",
    ).sort_index(kind="stable")
    info = dict(n_rows=n_rows, n_bad_timestamps=n_bad_ts, n_missing_values=int(raw.isna().sum()))
    return raw, info


def _to_bins(index: pd.DatetimeIndex, target: str) -> pd.DatetimeIndex:
    """Map each timestamp to the start of its day / week (Monday) / month."""
    days = index.floor("D")
    if target == "Daily":
        out = days
    elif target == "Weekly":
        out = days - pd.to_timedelta(days.dayofweek, unit="D")
    elif target == "Monthly":
        out = index.to_period("M").to_timestamp()
    else:
        raise ValueError(target)
    return pd.DatetimeIndex(out).astype("datetime64[ns]")


def detect_native_frequency(index: pd.DatetimeIndex) -> dict:
    """Describe the typical spacing of the timestamps."""
    uniq = pd.DatetimeIndex(index).unique().sort_values()
    if len(uniq) < 3:
        raise ValidationError("At least 3 distinct timestamps are needed.")
    steps = pd.Series(uniq).diff().dropna().dt.total_seconds()
    secs = float(steps.median())
    days = secs / 86400.0
    if secs < 0.9 * 86400:
        rank, label = 0.0, f"Intraday (about every {secs / 3600:.3g} hours)" if secs >= 3600 else \
            f"Intraday (about every {secs / 60:.3g} minutes)"
        if 0.9 * 3600 <= secs <= 1.1 * 3600:
            label = "Hourly"
    elif days <= 1.5:
        rank, label = 1.0, "Daily"
    elif days < 5:
        rank, label = 1.5, f"Every ~{days:.3g} days"
    elif days <= 9:
        rank, label = 2.0, "Weekly"
    elif days < 26:
        rank, label = 2.5, f"Every ~{days:.3g} days"
    elif days <= 35:
        rank, label = 3.0, "Monthly"
    else:
        rank, label = 4.0, f"Every ~{days:.3g} days (coarser than monthly)"
    return dict(label=label, rank=rank, step_seconds=secs)


def profile_frequency(raw: pd.Series, target: str) -> dict:
    """Validate that the data is at least as fine-grained as the target frequency.

    Returns a profile dict; raises ValidationError if the data is too coarse."""
    cfg = TARGET_CFG[target]
    nat = detect_native_frequency(raw.index)
    if nat["rank"] > cfg["rank"]:
        raise ValidationError(
            f"Your data looks like **{nat['label']}** (median spacing about "
            f"{nat['step_seconds'] / 86400:.3g} days), which is coarser than the **{target}** forecast "
            f"frequency you chose. A {target.lower()} forecast needs {target.lower()} data or finer. "
            "Choose a coarser forecast frequency or upload finer-grained data."
        )
    bins = _to_bins(raw.index, target)
    rows_per_period = pd.Series(1, index=bins).groupby(level=0).size()
    return dict(
        native_label=nat["label"],
        native_rank=nat["rank"],
        step_seconds=nat["step_seconds"],
        needs_aggregation=bool((rows_per_period > 1).any()),
        max_rows_per_period=int(rows_per_period.max()),
        n_rows=len(raw),
        start=raw.index.min(),
        end=raw.index.max(),
    )


# --------------------------------------------------------------------------------------
# 2. Cleaning: aggregate, fill, cap outliers
# --------------------------------------------------------------------------------------
def _mode(s: pd.Series):
    return s.mode().iloc[0] if s.notna().any() else np.nan


def aggregate(raw: pd.Series, target: str, agg: str | None):
    """Aggregate raw rows into one value per period (day/week/month) on a complete calendar.

    Periods with no rows (or only blank values) are NaN; they are filled afterwards."""
    cfg = TARGET_CFG[target]
    bins = _to_bins(raw.index, target)
    frame = pd.DataFrame({"v": raw.to_numpy(), "b": bins})
    g = frame.groupby("b")["v"]
    needs_agg = bool((g.size() > 1).any())
    if needs_agg and agg is None:
        raise ValidationError(
            f"Several rows fall into the same {cfg['unit']}. Choose how to combine them (sum, average, ...)."
        )
    if agg is None:
        out = g.first()
    elif agg == "sum":
        out = g.sum(min_count=1)   # an all-blank period stays NaN rather than becoming 0
    elif agg == "mode":
        out = g.agg(_mode)
    elif agg in ("mean", "median", "min", "max", "first", "last"):
        out = getattr(g, agg)()
    else:
        raise ValueError(agg)
    n_with_rows = len(out)
    full = pd.date_range(out.index.min(), out.index.max(), freq=cfg["alias"])
    out = out.reindex(full).astype(float)
    out.name = "value"
    return out, dict(n_empty_periods=int(len(full) - n_with_rows), aggregated=needs_agg)


def fill_missing(s: pd.Series, method: str) -> pd.Series:
    """Fill NaNs. Edge gaps that a directional fill cannot reach are filled from the other side."""
    if s.notna().sum() == 0:
        raise ValidationError("There are no valid numeric values left after cleaning.")
    if method == "zero":
        return s.fillna(0.0)
    if method == "ffill":
        return s.ffill().bfill()
    if method == "bfill":
        return s.bfill().ffill()
    if method == "mean":
        return s.fillna(float(s.mean()))
    if method == "median":
        return s.fillna(float(s.median()))
    if method == "linear":
        return s.interpolate(method="linear", limit_direction="both")
    raise ValueError(method)


def cap_outliers(s: pd.Series, target: str):
    """Winsorise outliers. Returns (treated series, number of points changed).

    * Mostly-zero series (>=30% zeros): cap only unusually large non-zero values (Q3 + 3*IQR of the
      non-zero values), so the zeros are never treated as the 'normal' level.
    * Otherwise: compare each point with a rolling median and cap points further than 3.5 robust
      standard deviations (MAD-based) away from it. Non-negative data is judged on a log scale, so
      the natural right-skew of demand/price data is not mistaken for outliers."""
    y = s.to_numpy(dtype=float)
    if len(y) < 8:
        return s, 0
    if np.mean(y == 0) >= 0.3:
        nz = y[y != 0]
        if len(nz) < 4:
            return s, 0
        q1, q3 = np.percentile(nz, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return s, 0
        hi = q3 + 3.0 * iqr
        treated = np.where(y > hi, hi, y)
    else:
        w = 7 if target == "Daily" else 5
        nonneg = bool(y.min() >= 0)
        z = np.log1p(y) if nonneg else y
        med = pd.Series(z).rolling(w, center=True, min_periods=1).median().to_numpy()
        r = z - med
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if scale == 0:
            scale = float(np.std(r))
        if scale == 0:
            return s, 0
        capped = np.clip(z, med - 3.5 * scale, med + 3.5 * scale)
        if nonneg:
            capped = np.maximum(np.expm1(capped), 0.0)
        # points inside the fences stay exactly as they were (no log/exp round-off)
        treated = np.where(np.isclose(capped, y, rtol=1e-9, atol=1e-12), y, capped)
    n_changed = int(np.sum(treated != y))
    return pd.Series(treated, index=s.index, name=s.name), n_changed


# --------------------------------------------------------------------------------------
# 3. Demand classification (Syntetos-Boylan)
# --------------------------------------------------------------------------------------
def classify_demand(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    n = len(y)
    nz = y[y != 0]
    if len(nz) == 0:
        return dict(category="No demand", adi=float("inf"), cv2=float("nan"),
                    n_nonzero=0, n_periods=n, description=CATEGORY_TEXT["No demand"])
    adi = n / len(nz)
    mean = abs(float(nz.mean()))
    cv2 = float((nz.std(ddof=0) / mean) ** 2) if mean > 0 else float("inf")
    if adi < ADI_CUT:
        cat = "Smooth" if cv2 < CV2_CUT else "Erratic"
    else:
        cat = "Intermittent" if cv2 < CV2_CUT else "Lumpy"
    return dict(category=cat, adi=float(adi), cv2=cv2, n_nonzero=int(len(nz)),
                n_periods=n, description=CATEGORY_TEXT[cat])


# --------------------------------------------------------------------------------------
# 4. Forecasting models (pure numpy / scipy). Each: fn(y, h, m) -> array of h forecasts
# --------------------------------------------------------------------------------------
def _normalised(fn):
    """Fit on standardised data so optimiser settings do not depend on the data's scale."""
    def wrapped(y, h, m):
        mu = float(np.mean(y))
        sc = float(np.std(y))
        if sc < 1e-12:
            sc = 1.0
        return fn((y - mu) / sc, h, m) * sc + mu
    return wrapped


def _ses_state(y, alpha):
    """Final level and sum of squared one-step errors of simple exponential smoothing."""
    zi = [(1.0 - alpha) * y[0]]
    lev, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], y[1:], zi=zi)
    prev = np.concatenate([[y[0]], lev[:-1]])
    err = y[1:] - prev
    return float(lev[-1]), float(err @ err)


def _fit_alpha(y) -> float:
    return float(minimize_scalar(lambda a: _ses_state(y, a)[1], bounds=(0.01, 0.99), method="bounded").x)


def _naive(y, h, m):
    return np.full(h, y[-1])


def _mean(y, h, m):
    return np.full(h, float(np.mean(y)))


def _drift(y, h, m):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def _snaive(y, h, m):
    return y[len(y) - m + (np.arange(h) % m)]


def _ses(y, h, m):
    level, _ = _ses_state(y, _fit_alpha(y))
    return np.full(h, level)


def _theta(y, h, m):
    """Theta method (Hyndman & Billah form): SES plus half the linear-trend slope."""
    n = len(y)
    a = _fit_alpha(y)
    level, _ = _ses_state(y, a)
    b0 = float(np.polyfit(np.arange(n), y, 1)[0])
    hh = np.arange(1, h + 1)
    return level + 0.5 * b0 * ((hh - 1) + 1.0 / a - ((1.0 - a) ** n) / a)


def _holt_run(params, y):
    a, b, phi = params
    level, trend, sse = y[0], 0.0, 0.0
    for t in range(1, len(y)):
        f = level + phi * trend
        e = y[t] - f
        sse += e * e
        level = f + a * e
        trend = phi * trend + a * b * e
    return sse, level, trend


def _damped(y, h, m):
    """Holt's linear trend with damping (additive error-correction form)."""
    best = None
    for x0 in ([0.3, 0.1, 0.9], [0.1, 0.02, 0.95]):
        r = minimize(lambda p: _holt_run(p, y)[0], x0,
                     bounds=[(0.01, 0.99), (0.001, 0.99), (0.80, 0.98)], method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    _, level, trend = _holt_run(best.x, y)
    phi = best.x[2]
    return level + np.cumsum(phi ** np.arange(1, h + 1)) * trend


def _seasonal_indices(y, m, multiplicative):
    n = len(y)
    if m % 2 == 1:
        w, k = np.ones(m) / m, (m - 1) // 2
    else:
        w, k = np.r_[0.5, np.ones(m - 1), 0.5] / m, m // 2
    ma = np.full(n, np.nan)
    conv = np.convolve(y, w, mode="valid")
    ma[k:k + len(conv)] = conv
    ratio = y / ma if multiplicative else y - ma
    pos = np.arange(n) % m
    idx = np.array([np.nanmean(ratio[pos == i]) for i in range(m)])
    idx = np.nan_to_num(idx, nan=1.0 if multiplicative else 0.0)
    return idx / idx.mean() if multiplicative else idx - idx.mean()


def _seasonal(base):
    """Deseasonalise (classical decomposition), forecast with `base`, then re-apply the seasonal pattern."""
    def wrapped(y, h, m):
        mult = bool(np.min(y) > 0)
        idx = _seasonal_indices(y, m, mult)
        n = len(y)
        pos = np.arange(n) % m
        yd = y / idx[pos] if mult else y - idx[pos]
        fd = base(yd, h, m)
        fpos = (n + np.arange(h)) % m
        return fd * idx[fpos] if mult else fd + idx[fpos]
    return wrapped


# --- intermittent-demand models ---
def _croston_run(y, alpha):
    nz = np.flatnonzero(y != 0)
    z, q, last = float(y[nz[0]]), float(nz[0] + 1), int(nz[0])
    f = np.full(len(y), np.nan)
    for t in range(last + 1, len(y)):
        f[t] = z / q
        if y[t] != 0:
            z += alpha * (y[t] - z)
            q += alpha * ((t - last) - q)
            last = t
    return z, q, f


def _croston_family(y, h, sba):
    if not np.any(y != 0):
        return np.zeros(h)
    best = None
    for a in (0.05, 0.1, 0.2, 0.3):
        z, q, f = _croston_run(y, a)
        fac = (1.0 - a / 2.0) if sba else 1.0
        ok = ~np.isnan(f)
        sse = float(np.sum((y[ok] - fac * f[ok]) ** 2)) if ok.any() else np.inf
        if best is None or sse < best[0]:
            best = (sse, fac * z / q)
    return np.full(h, best[1])


def _croston(y, h, m):
    return _croston_family(y, h, sba=False)


def _sba(y, h, m):
    return _croston_family(y, h, sba=True)


def _tsb_run(y, alpha, beta):
    d = (y != 0).astype(float)
    nz = np.flatnonzero(d)
    first = int(nz[0])
    z, p, sse = float(y[first]), 1.0 / (first + 1), 0.0
    for t in range(first + 1, len(y)):
        sse += (y[t] - p * z) ** 2
        if d[t]:
            z += alpha * (y[t] - z)
        p += beta * (d[t] - p)
    return p * z, sse


def _tsb(y, h, m):
    if not np.any(y != 0):
        return np.zeros(h)
    best = None
    for a in (0.05, 0.1, 0.2, 0.3):
        for b in (0.05, 0.1, 0.2, 0.3):
            rate, sse = _tsb_run(y, a, b)
            if best is None or sse < best[0]:
                best = (sse, rate)
    return np.full(h, best[1])


# (display name, function, needs seasonal period)
_SES_N, _THETA_N, _DAMPED_N = _normalised(_ses), _normalised(_theta), _normalised(_damped)
CONTINUOUS_MODELS = [
    ("Naive (last value)", _naive, False),
    ("Drift (random walk with drift)", _normalised(_drift), False),
    ("Simple exponential smoothing", _SES_N, False),
    ("Damped trend (Holt)", _DAMPED_N, False),
    ("Theta", _THETA_N, False),
    ("Seasonal naive", _snaive, True),
    ("Seasonal + exponential smoothing", _seasonal(_SES_N), True),
    ("Seasonal + damped trend", _seasonal(_DAMPED_N), True),
    ("Seasonal + Theta", _seasonal(_THETA_N), True),
]
INTERMITTENT_MODELS = [
    ("Historical mean", _mean, False),
    ("Simple exponential smoothing", _SES_N, False),
    ("Croston", _croston, False),
    ("Croston-SBA", _sba, False),
    ("TSB", _tsb, False),
]


# --------------------------------------------------------------------------------------
# 5. Model selection by rolling-origin backtest, then final forecast
# --------------------------------------------------------------------------------------
@dataclass
class ForecastResult:
    model_name: str
    values: np.ndarray
    backtest: pd.DataFrame
    val_h: int
    n_folds: int
    notes: list = field(default_factory=list)


def forecast_series(y: np.ndarray, target: str, horizon: int, category: str) -> ForecastResult:
    y = np.asarray(y, dtype=float)
    notes: list[str] = []
    if len(y) > MAX_FIT:
        y = y[-MAX_FIT:]
        notes.append(f"Only the most recent {MAX_FIT} periods were used to fit the models.")
    nonneg = bool(y.min() >= 0)

    if category == "No demand":
        return ForecastResult("Zero forecast (no non-zero history)", np.zeros(horizon),
                              pd.DataFrame(), 0, 0, notes)

    m = TARGET_CFG[target]["season"]
    n = len(y)
    intermittent = category in ("Intermittent", "Lumpy")
    models = INTERMITTENT_MODELS if intermittent else CONTINUOUS_MODELS
    metric = "RMSE" if intermittent else "MAE"   # MAE would reward all-zero forecasts on sparse data

    val_h = int(min(horizon, max(3, n // 4)))
    step = max(1, val_h // 2)
    origins = [o for o in (n - val_h - k * step for k in range(3)) if o >= MIN_TRAIN]

    rows = []
    for name, fn, seasonal in models:
        label = f"{name} (period {m})" if seasonal else name
        if seasonal and (not origins or min(origins) < 2 * m):
            continue   # not enough history for two full seasonal cycles
        errs = []
        try:
            for o in origins:
                pred = fn(y[:o], val_h, m)
                errs.append(y[o:o + val_h] - pred)
            if not errs:
                continue
            e = np.concatenate(errs)
            actual = np.concatenate([y[o:o + val_h] for o in origins])
            denom = float(np.abs(actual).sum())
            rows.append(dict(Model=label, MAE=float(np.mean(np.abs(e))),
                             RMSE=float(np.sqrt(np.mean(e ** 2))),
                             **{"WAPE %": 100.0 * float(np.abs(e).sum()) / denom if denom > 0 else np.nan},
                             _fn=fn, _name=label))
        except Exception:
            continue   # a model that fails on this series is simply not a candidate
    rows = [r for r in rows if np.isfinite(r[metric])]

    if rows:
        best = min(rows, key=lambda r: r[metric])
    else:   # too little history for a backtest: fall back to a robust default
        fallback = ("TSB", _tsb) if intermittent else ("Simple exponential smoothing", _SES_N)
        best = dict(_name=fallback[0] + " (default, history too short to compare models)", _fn=fallback[1])
        notes.append("History was too short to backtest several models, so a default model was used.")

    values = np.asarray(best["_fn"](y, horizon, m), dtype=float)
    if nonneg:
        values = np.maximum(values, 0.0)

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    if len(table):
        table = table.sort_values(metric).reset_index(drop=True)
        table.insert(1, "Selected", table["Model"] == best["_name"])
        table[["MAE", "RMSE", "WAPE %"]] = table[["MAE", "RMSE", "WAPE %"]].round(4)
    return ForecastResult(best["_name"], values, table, val_h, len(origins), notes)


def future_index(last: pd.Timestamp, target: str, horizon: int) -> pd.DatetimeIndex:
    if target == "Daily":
        start = last + pd.Timedelta(days=1)
    elif target == "Weekly":
        start = last + pd.Timedelta(days=7)
    else:
        start = last + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=horizon, freq=TARGET_CFG[target]["alias"])


# --------------------------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    target: str
    horizon: int
    profile: dict
    parse_info: dict
    n_missing_filled: int
    n_empty_periods: int
    n_outliers_capped: int
    aggregated_flag: bool
    cleaned: pd.Series          # history actually used for modelling (filled, outliers treated)
    classification: dict
    model_name: str
    backtest: pd.DataFrame
    val_h: int
    n_folds: int
    forecast: pd.DataFrame      # index = future period start, column 'forecast'
    warnings: list = field(default_factory=list)


def run_forecast(df: pd.DataFrame, ts_col: str, val_col: str, target: str, horizon: int,
                 agg: str | None, missing: str, robust: bool, dayfirst: bool = False) -> Result:
    cfg = TARGET_CFG[target]
    if not (1 <= horizon <= cfg["max_h"]):
        raise ValidationError(f"The forecast length must be between 1 and {cfg['max_h']} {cfg['unit']}s.")

    raw, parse_info = parse_input(df, ts_col, val_col, dayfirst)
    profile = profile_frequency(raw, target)
    agg_series, agg_meta = aggregate(raw, target, agg)

    n_periods = len(agg_series)
    if n_periods < cfg["min_obs"]:
        raise ValidationError(
            f"Only {n_periods} {cfg['unit']}s of history found; at least {cfg['min_obs']} are needed "
            f"for a {target.lower()} forecast."
        )
    n_observed = int(agg_series.notna().sum())
    if n_observed < 8:
        raise ValidationError(f"Only {n_observed} periods contain a value; at least 8 are needed.")

    n_missing = int(agg_series.isna().sum())
    filled = fill_missing(agg_series, missing)
    if robust:
        cleaned, n_capped = cap_outliers(filled, target)
    else:
        cleaned, n_capped = filled, 0

    classification = classify_demand(cleaned.to_numpy())
    fc = forecast_series(cleaned.to_numpy(), target, horizon, classification["category"])

    warns = list(fc.notes)
    if n_missing / n_periods > 0.2:
        warns.append(f"{100 * n_missing / n_periods:.0f}% of periods had no value and were filled; "
                     "forecast quality may suffer.")
    if missing == "zero" and n_missing / n_periods > 0.1:
        warns.append("Zero-filling treats every missing period as a real zero, which can make the series "
                     "look intermittent. Try forward fill or interpolation if the gaps are just missing data.")
    if horizon > n_periods:
        warns.append(f"You asked for {horizon} {cfg['unit']}s but the history has only {n_periods}; "
                     "long horizons are unreliable.")

    idx = future_index(cleaned.index[-1], target, horizon)
    forecast = pd.DataFrame({"forecast": fc.values}, index=idx)
    forecast.index.name = "period_start"
    return Result(
        target=target, horizon=horizon, profile=profile, parse_info=parse_info,
        n_missing_filled=n_missing, n_empty_periods=agg_meta["n_empty_periods"],
        n_outliers_capped=n_capped, aggregated_flag=agg_meta["aggregated"],
        cleaned=cleaned, classification=classification, model_name=fc.model_name,
        backtest=fc.backtest, val_h=fc.val_h, n_folds=fc.n_folds, forecast=forecast, warnings=warns,
    )
