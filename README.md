# Time-series forecaster (Streamlit)

Upload a table with a **timestamp** column and a **value** column, choose how to prepare it, and get:

1. a **demand category** (Smooth / Erratic / Intermittent / Lumpy), and
2. the next **n** forecast values, with a chart and a CSV download.

## What the app does

| Step | Behaviour |
|---|---|
| Frequency | Detects how your data is spaced. You pick a forecast frequency: **Daily, Weekly or Monthly**. The data must be at that level or finer; coarser data is rejected with an explanation. |
| Aggregation | If several rows fall into one day/week/month (e.g. hourly data), you choose how to combine them: sum, average, median, mode, min, max, first, last. |
| Missing values | Applied after aggregation, to missing periods *and* blank values: zero, forward fill, backward fill, mean, median, or linear interpolation. |
| Outliers | **Simple** uses the data as is. **Robust** caps extreme points (median + MAD rule, judged on a log scale for non-negative data; for mostly-zero series only unusually large non-zero values are capped). |
| Classification | Syntetos–Boylan: ADI cut-off 1.32, CV² cut-off 0.49, computed on the cleaned series. All-zero series are labelled "No demand". |
| Model | Chosen automatically by a rolling-origin backtest on the end of your history. Smooth/erratic: naive, drift, exponential smoothing, damped trend, Theta, plus seasonal versions when there are two full cycles. Intermittent/lumpy: mean, exponential smoothing, Croston, Croston-SBA, TSB. |

Seasonal periods used: 7 (daily), 52 (weekly), 12 (monthly). Only the most recent 1,000 periods are used to fit models.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish on Streamlit Community Cloud (free)

1. Create a GitHub repository and upload all files in this folder (`app.py`, `forecasting.py`, `sample_data.py`, `requirements.txt`, `.streamlit/`, `samples/`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, pick the repository, branch `main`, and main file `app.py`.
3. Click **Deploy**. You get a public `*.streamlit.app` link.

The free tier has limited CPU and memory (and apps sleep after a period without visitors), which is fine for this app. Open the link shortly before a demo to wake it.

## Tests

```bash
python test_forecasting.py
```

Covers frequency detection, validation, aggregation, missing-value fills, outlier handling, classification, model sanity and end-to-end runs on the sample datasets.

## Files

- `app.py`: Streamlit interface
- `forecasting.py`: all cleaning, classification and forecasting logic (no Streamlit dependency)
- `sample_data.py`: synthetic sample datasets (also written to `samples/` as CSVs)
- `test_forecasting.py`: tests

Forecasts are statistical estimates for demonstration, not financial or business advice.
