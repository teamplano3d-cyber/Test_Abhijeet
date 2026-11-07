"""Streamlit front-end for the time-series forecaster.   Run with:  streamlit run app.py"""
import io
import math

import pandas as pd
import streamlit as st

import forecasting as fc
from forecasting import ValidationError
from sample_data import SAMPLES, make_sample

MAX_UPLOAD_MB = 20
MAX_ROWS = 500_000

st.set_page_config(page_title="Time-series forecaster", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        if df.shape[1] == 1:   # maybe a ; or tab separated file
            try:
                alt = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
                if alt.shape[1] > 1:
                    df = alt
            except Exception:
                pass
        return df
    return pd.read_excel(io.BytesIO(data))


def fmt(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{digits}f}"


def show_results(res: fc.Result) -> None:
    unit = fc.TARGET_CFG[res.target]["unit"]
    cl = res.classification

    st.header("3. Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Demand category", cl["category"])
    m2.metric("ADI", fmt(cl["adi"]), help="Average demand interval: periods per non-zero value. "
              f"Values at or above {fc.ADI_CUT} count as 'sparse'.")
    m3.metric("CV²", fmt(cl["cv2"]), help="Squared coefficient of variation of the non-zero values. "
              f"Values at or above {fc.CV2_CUT} count as 'highly variable'.")
    st.write(cl["description"])
    st.caption(f"Syntetos–Boylan classification on the cleaned {res.target.lower()} series. It looks at the level of the "
               "values (not returns), so a trending price series is judged on how widely its level varies.")
    st.markdown(f"**Selected model:** {res.model_name}")
    for w in res.warnings:
        st.warning(w)

    hist = res.cleaned
    keep = min(len(hist), max(5 * res.horizon, 60))
    chart_df = pd.concat([hist.iloc[-keep:].rename("History (cleaned)"),
                          res.forecast["forecast"].rename("Forecast")], axis=1)
    st.line_chart(chart_df)

    st.subheader(f"Next {res.horizon} {unit}{'s' if res.horizon != 1 else ''}")
    table = res.forecast.copy()
    table.index = pd.Index(res.forecast.index.strftime("%Y-%m-%d"), name="period_start")
    st.dataframe(table.round(4))
    st.download_button("Download forecast (CSV)", table.reset_index().to_csv(index=False),
                       file_name="forecast.csv", mime="text/csv")

    with st.expander("Data preparation summary"):
        pi = res.parse_info
        st.markdown(
            f"- Rows read: **{pi['n_rows']:,}** (unreadable dates ignored: {pi['n_bad_timestamps']:,})\n"
            f"- Detected data frequency: **{res.profile['native_label']}**\n"
            f"- Several rows per {unit} combined: **{'yes' if res.aggregated_flag else 'not needed'}**\n"
            f"- {res.target} periods in history: **{len(res.cleaned):,}**\n"
            f"- Missing periods filled: **{res.n_missing_filled:,}** "
            f"({res.n_empty_periods:,} had no rows at all, the rest had blank or non-numeric values)\n"
            f"- Outliers capped: **{res.n_outliers_capped:,}**"
        )

    with st.expander("How the model was chosen"):
        if res.backtest.empty:
            st.write("No model comparison was possible for this series.")
        else:
            metric = "RMSE" if cl["category"] in ("Intermittent", "Lumpy") else "MAE"
            st.write(f"Each candidate model was tested on {res.n_folds} rolling hold-out window(s) of {res.val_h} "
                     f"{unit}(s) taken from the end of your history. The lowest **{metric}** wins. "
                     "Seasonal models are only tried when there are at least two full seasonal cycles of history.")
            st.dataframe(res.backtest)

    st.caption("Statistical forecasts for demonstration purposes only. They are not financial or business advice.")


# ------------------------------------------------------------------ page
st.title("📈 Time-series forecaster")
st.write("Upload a timestamp + value table, choose how to prepare it, and get a demand category plus an n-period forecast.")

# ---- 1. data
st.header("1. Data")
source = st.radio("Data source", ["Upload a file", "Use a sample dataset"], horizontal=True)
if source == "Upload a file":
    up = st.file_uploader("CSV or Excel (.xlsx) file with a timestamp column and a value column", type=["csv", "xlsx"])
    if up is None:
        st.info("Upload a file, or switch to a sample dataset to try the app.")
        st.stop()
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"That file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller file.")
        st.stop()
    try:
        df = read_table(up.name, up.getvalue())
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    data_id = (up.name, up.size)
else:
    sample = st.selectbox("Sample dataset", list(SAMPLES))
    st.caption(SAMPLES[sample])
    df = make_sample(sample)
    data_id = ("sample", sample)

if df.shape[1] < 2:
    st.error("The file needs at least two columns: a timestamp and a value.")
    st.stop()
if len(df) > MAX_ROWS:
    st.error(f"The file has {len(df):,} rows; the limit for this demo is {MAX_ROWS:,}.")
    st.stop()

st.dataframe(df.head(8))
st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

cols = list(df.columns)
guess_ts, guess_val = fc.guess_columns(df)
tag = f"{data_id[0]}-{data_id[1]}"
c1, c2, c3 = st.columns(3)
ts_col = c1.selectbox("Timestamp column", cols, index=cols.index(guess_ts), key=f"ts-{tag}")
val_options = [c for c in cols if c != ts_col]
val_col = c2.selectbox("Value column", val_options,
                       index=val_options.index(guess_val) if guess_val in val_options else 0, key=f"val-{tag}")
dayfirst = c3.checkbox("Dates are day-first (05/11/2026 = 5 Nov)", value=False, key=f"df-{tag}")

# ---- 2. settings
st.header("2. Settings")
s1, s2 = st.columns(2)
target = s1.selectbox("Forecast frequency", fc.TARGETS)
cfg = fc.TARGET_CFG[target]
horizon = int(s2.number_input(f"How many {cfg['unit']}s to forecast (n)", min_value=1, max_value=cfg["max_h"],
                              value=cfg["default_h"], step=1))

try:
    raw, parse_info = fc.parse_input(df, ts_col, val_col, dayfirst)
    profile = fc.profile_frequency(raw, target)
except ValidationError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(f"**Detected data frequency:** {profile['native_label']} · {profile['n_rows']:,} usable rows · "
            f"{profile['start']:%Y-%m-%d} → {profile['end']:%Y-%m-%d}")
if parse_info["n_bad_timestamps"]:
    st.warning(f"{parse_info['n_bad_timestamps']:,} rows have dates that could not be read and will be ignored.")

agg = None
if profile["needs_aggregation"]:
    st.warning(f"Your data has up to {profile['max_rows_per_period']:,} rows per {cfg['unit']}, so rows must be "
               f"combined into one value per {cfg['unit']} for a {target.lower()} forecast.")
    agg_label = st.selectbox(f"How should rows within a {cfg['unit']} be combined?", list(fc.AGG_OPTIONS))
    agg = fc.AGG_OPTIONS[agg_label]
else:
    st.success(f"The data is already at {target.lower()} level, so no aggregation is needed.")

o1, o2 = st.columns(2)
miss_label = o1.selectbox(
    "How should missing values be handled?", list(fc.MISSING_OPTIONS), index=1,
    help="A value is missing when a calendar period has no row, or the row is blank/non-numeric. "
         "This is applied after rows are combined. Zero-fill suits 'no record = no sales' demand data; "
         "forward fill or interpolation suits prices and readings.")
missing = fc.MISSING_OPTIONS[miss_label]
outlier_label = o2.radio("Outlier handling", fc.OUTLIER_OPTIONS,
                         help="Robust mode caps extreme points (to a limit based on the median and MAD) before "
                              "classifying and forecasting.")
robust = outlier_label == fc.OUTLIER_OPTIONS[1]

start = st.button("Start forecast", type="primary")
signature = (data_id, ts_col, val_col, dayfirst, target, horizon, agg, missing, robust)

if start:
    with st.spinner("Preparing data, classifying and fitting models..."):
        try:
            st.session_state["result"] = fc.run_forecast(df, ts_col, val_col, target, horizon, agg,
                                                         missing, robust, dayfirst)
            st.session_state["signature"] = signature
        except ValidationError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # unexpected failure: show it instead of a stack trace
            st.session_state.pop("result", None)
            st.error(f"Something went wrong while forecasting: {exc}")
            st.stop()

result = st.session_state.get("result")
if result is not None:
    if st.session_state.get("signature") != signature:
        st.info("Your settings changed since this forecast was made. Press **Start forecast** to refresh it.")
    show_results(result)
