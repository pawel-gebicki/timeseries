"""Feature engineering helpers — Phase 3.

Converts the cleaned time-series into a rich tabular feature matrix
suitable for gradient-boosted tree models (XGBoost, LightGBM, etc.).

Public API
----------
build_feature_matrix(df, target_col='unit_sales', drop_na=True)
    Master builder called by every ML notebook.  Returns a copy of *df*
    augmented with lag and rolling features; optionally drops the leading
    NaN rows introduced by the longest lag.

add_lag_features(df, target_col, lags)
    Appends columns  ``lag_<n>``  for each n in *lags*.

add_rolling_features(df, target_col, windows)
    Appends  ``roll_mean_<w>``  and  ``roll_std_<w>``  for each window.
    A mandatory  ``shift(1)``  is applied first so no future information
    leaks into the feature set.

All functions operate on a copy; the original DataFrame is never mutated.
The DataFrame is expected to be sorted by date ascending before calling
any of these helpers.
"""

import numpy as np
import pandas as pd


# ── Lag features ─────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "unit_sales",
    lags: list = None,
) -> pd.DataFrame:
    """Return *df* (copy) with one lag column per entry in *lags*.

    Column names: ``lag_1``, ``lag_7``, ``lag_14``, ``lag_30``.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date ascending.
    target_col : str
        Column to lag (default ``'unit_sales'``).
    lags : list[int]
        Periods to shift.  Defaults to [1, 7, 14, 30].

    Returns
    -------
    pd.DataFrame
        A copy of *df* with additional lag columns.
    """
    if lags is None:
        lags = [1, 7, 14, 30]

    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


# ── Rolling features ──────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "unit_sales",
    windows: list = None,
) -> pd.DataFrame:
    """Return *df* (copy) with rolling mean and std for each window.

    Column names: ``roll_mean_<w>``, ``roll_std_<w>``.

    A ``shift(1)`` is applied to the series before the rolling window so
    the feature for day *t* is computed from days *[t-w, t-1]*, never
    including the current observation.  This prevents look-ahead leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date ascending.
    target_col : str
        Column to roll over (default ``'unit_sales'``).
    windows : list[int]
        Window sizes in days.  Defaults to [7, 14, 28].

    Returns
    -------
    pd.DataFrame
        A copy of *df* with additional rolling columns.
    """
    if windows is None:
        windows = [7, 14, 28]

    out = df.copy()
    shifted = out[target_col].shift(1)           # t-1 to avoid leakage
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        out[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=1).std()
    return out


# ── Oil-price lag ─────────────────────────────────────────────────────────────

def add_oil_lag(
    df: pd.DataFrame,
    oil_col: str = "dcoilwtico",
    lags: list = None,
) -> pd.DataFrame:
    """Return *df* (copy) with lagged oil-price columns.

    Adds ``oil_lag_<n>`` for each n in *lags* (default [1, 7]).
    Useful for capturing delayed macro-economic effects on grocery sales.
    """
    if lags is None:
        lags = [1, 7]

    out = df.copy()
    if oil_col not in out.columns:
        return out
    for lag in lags:
        out[f"oil_lag_{lag}"] = out[oil_col].shift(lag)
    return out


# ── Master builder ────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "unit_sales",
    lag_list: list = None,
    rolling_windows: list = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build the complete tabular feature matrix for ML models.

    Steps applied (in order):
    1. Drop string helper columns (``dow_name``) not useful to tree models.
    2. Add lag features via :func:`add_lag_features`.
    3. Add rolling features via :func:`add_rolling_features`.
    4. Add oil-price lags via :func:`add_oil_lag`.
    5. Optionally drop the leading NaN rows caused by the longest lag.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, date-sorted DataFrame produced by the EDA notebook.
        Must contain ``unit_sales`` and the calendar / exogenous columns
        from ``timeseries_features.csv``.
    target_col : str
        Target column (default ``'unit_sales'``).
    lag_list : list[int]
        Lags to include (default [1, 7, 14, 30]).
    rolling_windows : list[int]
        Rolling windows to include (default [7, 14, 28]).
    drop_na : bool
        If *True* (default), drop rows where any feature column is NaN.
        This removes the first ``max(lag_list)`` rows from the matrix.

    Returns
    -------
    pd.DataFrame
        Feature matrix.  The ``date`` and ``unit_sales`` columns are
        preserved alongside the engineered features so downstream
        code can split on date and evaluate on actual sales values.
    """
    if lag_list is None:
        lag_list = [1, 7, 14, 30]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]

    out = df.copy()

    # 1. Drop string columns not useful for tree models
    out = out.drop(columns=[c for c in ["dow_name"] if c in out.columns])

    # 2. Lag features
    out = add_lag_features(out, target_col=target_col, lags=lag_list)

    # 3. Rolling features
    out = add_rolling_features(out, target_col=target_col, windows=rolling_windows)

    # 4. Oil-price lags
    out = add_oil_lag(out)

    # 5. Drop NaN rows (leading rows affected by the longest lag)
    if drop_na:
        feature_cols = [c for c in out.columns if c not in ["date", target_col]]
        before = len(out)
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            print(f"[feature_engineering] Dropped {dropped} leading NaN row(s) "
                  f"(longest lag = {max(lag_list)}).")

    return out
