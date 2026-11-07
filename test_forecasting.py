"""Tests for the core logic. Run with `python test_forecasting.py` (or pytest)."""
import time

import numpy as np
import pandas as pd

import forecasting as fc
from forecasting import ValidationError
from sample_data import SAMPLES, make_sample


def _df(index, values):
    return pd.DataFrame({"timestamp": index, "value": values})


# ---------------- frequency detection / validation ----------------
def test_detect_frequencies():
    d = lambda freq, n=60: pd.date_range("2025-01-01", periods=n, freq=freq)
    assert fc.detect_native_frequency(d("h"))["label"] == "Hourly"
    assert fc.detect_native_frequency(d("D"))["label"] == "Daily"
    assert fc.detect_native_frequency(d("W-FRI"))["label"] == "Weekly"
    assert fc.detect_native_frequency(d("MS"))["label"] == "Monthly"
    assert fc.detect_native_frequency(d("ME"))["label"] == "Monthly"
    assert fc.detect_native_frequency(d("15min"))["rank"] == 0
    assert fc.detect_native_frequency(pd.bdate_range("2025-01-01", periods=60))["label"] == "Daily"


def test_target_finer_than_data_is_rejected():
    idx = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    raw, _ = fc.parse_input(_df(idx.strftime("%Y-%m-%d"), np.arange(60.0)), "timestamp", "value")
    for target in ("Daily",):
        try:
            fc.profile_frequency(raw, target)
            raise AssertionError("weekly data should be rejected for a daily forecast")
        except ValidationError as e:
            assert "coarser" in str(e)
    assert fc.profile_frequency(raw, "Weekly")["needs_aggregation"] is False
    assert fc.profile_frequency(raw, "Monthly")["needs_aggregation"] is True


def test_hourly_needs_aggregation_and_sum_is_right():
    idx = pd.date_range("2025-03-01", periods=24 * 40, freq="h")
    df = _df(idx.strftime("%Y-%m-%d %H:%M"), np.ones(len(idx)))
    raw, _ = fc.parse_input(df, "timestamp", "value")
    prof = fc.profile_frequency(raw, "Daily")
    assert prof["needs_aggregation"] and prof["native_label"] == "Hourly"
    try:
        fc.aggregate(raw, "Daily", None)
        raise AssertionError("aggregation should be required")
    except ValidationError:
        pass
    s, meta = fc.aggregate(raw, "Daily", "sum")
    assert (s == 24).all() and len(s) == 40 and meta["n_empty_periods"] == 0
    s, _ = fc.aggregate(raw, "Daily", "mean")
    assert (s == 1).all()


def test_aggregation_functions():
    idx = pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00", "2025-01-02 01:00"])
    raw = pd.Series([1.0, 3.0, 3.0, 9.0], index=idx)
    out = {a: fc.aggregate(raw, "Daily", a)[0].iloc[0] for a in
           ("sum", "mean", "median", "mode", "min", "max", "first", "last")}
    assert out == dict(sum=7.0, mean=7 / 3, median=3.0, mode=3.0, min=1.0, max=3.0, first=1.0, last=3.0), out


def test_weekly_and_monthly_bins():
    # Wed 2024-01-03 -> week starting Mon 2024-01-01; month -> 2024-01-01
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-07", "2024-01-08", "2024-02-15"])
    assert list(fc._to_bins(idx, "Weekly")) == list(pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-02-12"]))
    assert list(fc._to_bins(idx, "Monthly")) == list(pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"]))


def test_gaps_become_nan_on_full_calendar():
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-05", "2025-01-06"])
    s, meta = fc.aggregate(pd.Series([1.0, 2, 5, 6], index=idx), "Daily", None)
    assert len(s) == 6 and s.isna().sum() == 2 and meta["n_empty_periods"] == 2


# ---------------- missing-value handling ----------------
def test_fill_methods():
    s = pd.Series([np.nan, 2.0, np.nan, np.nan, 8.0, np.nan], index=pd.date_range("2025-01-01", periods=6))
    assert fc.fill_missing(s, "zero").tolist() == [0, 2, 0, 0, 8, 0]
    assert fc.fill_missing(s, "ffill").tolist() == [2, 2, 2, 2, 8, 8]
    assert fc.fill_missing(s, "bfill").tolist() == [2, 2, 8, 8, 8, 8]
    assert fc.fill_missing(s, "mean").tolist() == [5, 2, 5, 5, 8, 5]
    assert fc.fill_missing(s, "median").tolist() == [5, 2, 5, 5, 8, 5]
    assert fc.fill_missing(s, "linear").tolist() == [2, 2, 4, 6, 8, 8]


# ---------------- outliers ----------------
def test_robust_caps_spike_but_not_normal_data():
    rng = np.random.default_rng(0)
    y = 100 + rng.normal(0, 2, 120)
    y[60] = 500
    s = pd.Series(y, index=pd.date_range("2025-01-01", periods=120))
    treated, n = fc.cap_outliers(s, "Daily")
    assert n >= 1 and treated.iloc[60] < 130 and treated.max() < 130
    clean = pd.Series(100 + rng.normal(0, 2, 120), index=s.index)
    _, n2 = fc.cap_outliers(clean, "Daily")
    assert n2 <= 3   # a clean series should be left essentially alone


def test_robust_does_not_flatten_intermittent():
    rng = np.random.default_rng(1)
    y = np.where(rng.random(200) < 0.2, 10 + rng.normal(0, 1, 200), 0.0)
    y[50] = 400   # genuine outlier amid the demand
    s = pd.Series(y, index=pd.date_range("2025-01-01", periods=200))
    treated, n = fc.cap_outliers(s, "Daily")
    assert n >= 1 and treated.iloc[50] < 100
    assert (treated[(y != 0) & (np.arange(200) != 50)] > 5).all()   # ordinary demand untouched


# ---------------- classification ----------------
def test_classification_quadrants():
    rng = np.random.default_rng(3)
    smooth = 100 + rng.normal(0, 5, 200)
    erratic = rng.lognormal(3, 1.0, 200)
    inter = np.where(rng.random(400) < 0.2, 10 + rng.normal(0, 1, 400), 0)
    lumpy = np.where(rng.random(400) < 0.2, rng.lognormal(3, 1.0, 400), 0)
    assert fc.classify_demand(smooth)["category"] == "Smooth"
    assert fc.classify_demand(erratic)["category"] == "Erratic"
    assert fc.classify_demand(inter)["category"] == "Intermittent"
    assert fc.classify_demand(lumpy)["category"] == "Lumpy"
    assert fc.classify_demand(np.zeros(30))["category"] == "No demand"
    c = fc.classify_demand(np.array([0, 0, 5, 0, 5, 0, 0, 5, 0, 0.0]))
    assert abs(c["adi"] - 10 / 3) < 1e-9 and c["cv2"] == 0


# ---------------- models ----------------
def test_models_sanity():
    y = np.full(40, 7.0)
    for name, fn, seasonal in fc.CONTINUOUS_MODELS + fc.INTERMITTENT_MODELS:
        if seasonal:
            continue
        out = fn(y, 5, 7)
        # Croston-SBA deliberately deflates its forecast by (1 - alpha/2), so allow up to 5% for it
        tol = 0.35 if name == "Croston-SBA" else 1e-6
        assert out.shape == (5,) and np.allclose(out, 7.0, atol=tol), (name, out)
    trend = np.arange(1.0, 61.0)
    drift = fc._drift(trend, 3, 7)
    assert np.allclose(drift, [61, 62, 63])
    assert fc._theta(trend, 3, 7)[0] > 60   # trending data -> forecast continues upward
    seas = np.tile([10.0, 20, 30, 20, 10, 5, 5], 10)
    assert np.allclose(fc._seasonal(fc._ses)(seas, 7, 7), seas[:7], atol=1e-6)
    assert np.allclose(fc._snaive(seas, 7, 7), seas[:7])
    sparse = np.array([0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4.0] * 5)
    assert abs(fc._croston(sparse, 1, 1)[0] - 4 / 3) < 0.2   # demand of 4 every 3 periods


# ---------------- end to end ----------------
def test_end_to_end_all_samples():
    expected = {
        "Smooth": {"Smooth"},
        "Erratic": {"Erratic"},
        "Intermittent": {"Intermittent"},
        "Lumpy": {"Lumpy", "Intermittent"},
        "Stock": {"Smooth"},
        "Monthly": {"Smooth"},
    }
    for name in SAMPLES:
        df = make_sample(name)
        target = "Monthly" if name.startswith("Monthly") else "Daily"
        agg = "mean" if name.startswith("Hourly") else None
        missing = "ffill" if name.startswith(("Stock", "Hourly")) else "zero"
        horizon = 6 if target == "Monthly" else 30
        t0 = time.time()
        res = fc.run_forecast(df, "timestamp", "value", target, horizon, agg, missing, robust=True)
        dt = time.time() - t0
        assert len(res.forecast) == horizon and np.isfinite(res.forecast["forecast"]).all()
        assert (res.forecast["forecast"] >= 0).all()
        assert res.forecast.index[0] > res.cleaned.index[-1]
        step = pd.Series(res.forecast.index).diff().dropna()
        assert step.nunique() == 1 if target == "Daily" else step.min() >= pd.Timedelta(days=28)
        key = next((k for k in expected if name.startswith(k)), None)   # hourly sample: any category is fine
        if key is not None:
            assert res.classification["category"] in expected[key], (name, res.classification)
        print(f"{name[:44]:44s} -> {res.classification['category']:12s} ADI={res.classification['adi']:.2f} "
              f"CV2={res.classification['cv2']:.2f} | {res.model_name} | {dt:.1f}s")


def test_forecast_follows_level_and_season():
    df = make_sample(next(k for k in SAMPLES if k.startswith("Smooth")))
    res = fc.run_forecast(df, "timestamp", "value", "Daily", 28, None, "zero", False)
    recent = res.cleaned.iloc[-28:].mean()
    assert abs(res.forecast["forecast"].mean() - recent) / recent < 0.1
    # weekly pattern: the model should reproduce that weekends (Sat) are the peak
    by_dow = res.forecast["forecast"].groupby(res.forecast.index.dayofweek).mean()
    assert by_dow.idxmax() == 5, by_dow


def test_intermittent_uses_intermittent_model():
    df = make_sample(next(k for k in SAMPLES if k.startswith("Intermittent")))
    res = fc.run_forecast(df, "timestamp", "value", "Daily", 30, None, "zero", False)
    assert res.model_name.split(" (")[0] in {m[0] for m in fc.INTERMITTENT_MODELS}
    assert res.forecast["forecast"].nunique() == 1   # flat demand-rate forecast


def test_weekly_from_daily_and_validation_errors():
    df = make_sample(next(k for k in SAMPLES if k.startswith("Smooth")))
    res = fc.run_forecast(df, "timestamp", "value", "Weekly", 8, "sum", "zero", False)
    assert len(res.forecast) == 8 and res.forecast.index[0].dayofweek == 0
    try:
        fc.run_forecast(df.head(10), "timestamp", "value", "Daily", 5, None, "zero", False)
        raise AssertionError("short history should fail")
    except ValidationError as e:
        assert "at least" in str(e)
    try:
        fc.run_forecast(df, "timestamp", "value", "Daily", 9999, None, "zero", False)
        raise AssertionError("huge horizon should fail")
    except ValidationError:
        pass


def test_bad_inputs():
    try:
        fc.parse_input(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), "a", "b")
        raise AssertionError("numeric timestamp column should be rejected")
    except ValidationError:
        pass
    df = pd.DataFrame({"d": ["2025-01-01", "oops", "2025-01-03"], "v": ["1,000", "x", "3"]})
    raw, info = fc.parse_input(df, "d", "v")
    assert info["n_bad_timestamps"] == 1 and raw.iloc[0] == 1000.0
    # day-first parsing
    raw, _ = fc.parse_input(pd.DataFrame({"d": ["05/11/2026"], "v": [1.0]}), "d", "v", dayfirst=True)
    assert raw.index[0] == pd.Timestamp("2026-11-05")
    assert fc.guess_columns(pd.DataFrame({"Date": ["2025-01-01"], "Sales": [1.0], "x": ["a"]})) == ("Date", "Sales")


def test_robust_mode_keeps_skewed_demand_category():
    # regression: a MAD rule on the raw scale flagged ~10% of a heavy-tailed series and turned Erratic into Smooth
    df = make_sample("Erratic daily demand")
    plain = fc.run_forecast(df, "timestamp", "value", "Daily", 30, None, "zero", False)
    robust = fc.run_forecast(df, "timestamp", "value", "Daily", 30, None, "zero", True)
    assert plain.classification["category"] == robust.classification["category"] == "Erratic"
    assert robust.n_outliers_capped <= 0.03 * len(robust.cleaned)
    spiked = df.copy()
    spiked.loc[200, "value"] = 50000
    r = fc.run_forecast(spiked, "timestamp", "value", "Daily", 30, None, "zero", True)
    assert r.cleaned.max() < 1000   # a genuine spike is still capped


def test_timezone_offsets_do_not_shift_dates():
    # regression: converting +05:30 to UTC moved local-midnight rows onto the previous day
    one = pd.DataFrame({"d": ["2025-01-01T00:00:00+05:30", "2025-01-02T00:00:00+05:30"], "v": [1.0, 2.0]})
    raw, _ = fc.parse_input(one, "d", "v")
    assert raw.index[0] == pd.Timestamp("2025-01-01")
    mixed = pd.DataFrame({"d": ["2025-01-01T00:00:00+05:30", "2025-01-02T00:00:00+00:00"], "v": [1.0, 2.0]})
    raw, info = fc.parse_input(mixed, "d", "v")
    assert len(raw) == 2 and info["n_bad_timestamps"] == 0


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", n, "->", repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
