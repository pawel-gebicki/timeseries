"""Shared model evaluation utilities.

A single evaluate_model() function is reused by every notebook so that all
models in the project are scored on identical metrics and the comparison is
defensible.  All phases — statistical baselines, ML models, and HyperOpt-tuned
champions — call this function, so the final comparison table is honest.
"""

import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    y_true,
    y_pred,
    model_name: str,
    fit_time: float,
) -> dict:
    """Score one model on the project's standard metric suite.

    Parameters
    ----------
    y_true : array-like
        Actual observed values (the test window, including store-closed days
        where unit_sales = 0).
    y_pred : array-like
        Model predictions for the same window.
    model_name : str
        Human-readable label shown in comparison tables.
    fit_time : float
        Wall-clock seconds taken to *fit* the model (not to predict).

    Returns
    -------
    dict
        Keys: Model, MAE, RMSE, MAPE (%), Bias, R², Fit time (s).

    Notes
    -----
    MAPE is computed only over rows where ``y_true > 0``.  Store-closed days
    (unit_sales = 0) are excluded from the percentage calculation because
    dividing by zero is undefined and would inflate the metric unfairly.
    Those rows are still included in MAE, RMSE, Bias, and R².
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # --- standard metrics (all rows) -----------------------------------
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    r2   = float(r2_score(y_true, y_pred))

    # --- MAPE: skip zero-sales rows ------------------------------------
    mask = y_true > 0
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )

    return {
        "Model":         model_name,
        "MAE":           round(mae,  1),
        "RMSE":          round(rmse, 1),
        "MAPE (%)":      round(mape, 1),
        "Bias":          round(bias, 1),
        "R²":            round(r2,   3),
        "Fit time (s)":  round(fit_time, 2),
    }

