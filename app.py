"""
Phase 5 — Streamlit Forecasting Dashboard
==========================================
MSIT Time Series Forecasting Project
Corporación Favorita Grocery Sales · Jan 2013 – Mar 2014

Pages
-----
  📈  Forecast        — Prophet champion model with CI & residuals
  📊  Model Comparison — All 5 models from MLflow, ranked by R²
  🔍  EDA             — Full EDA story from Phase 1
  ⚙️  HyperOpt        — Baseline vs tuned, delta table, best params
"""

import os
import sys
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import mlflow

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluation import evaluate_model  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_END  = "2013-12-31"
TEST_START = "2014-01-01"
TEST_END   = "2014-03-31"

PLOTLY_TMPL = "plotly_dark"
ACCENT      = "#7C3AED"
CYAN        = "#06B6D4"
GREEN       = "#10B981"
AMBER       = "#F59E0B"
RED         = "#EF4444"

MODEL_COLORS = {
    "prophet_baseline":     ACCENT,
    "prophet_tuned":        "#A78BFA",
    "sarima_baseline":      CYAN,
    "sarima_tuned":         "#67E8F9",
    "holtwinters_baseline": GREEN,
    "holtwinters_tuned":    "#6EE7B7",
    "xgboost_baseline":     AMBER,
    "xgboost_tuned":        "#FCD34D",
}

CHART_MARGIN = dict(l=44, r=24, t=56, b=44)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSIT · Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Card / badge helpers */
    .badge-champion {
        display: inline-block;
        background: linear-gradient(135deg,#7c3aed,#4f46e5);
        color: #fff; font-weight: 700; font-size: 0.95rem;
        padding: 6px 18px; border-radius: 8px; margin-bottom: 4px;
    }
    .divider { border-top: 1px solid #2d3148; margin: 22px 0; }
    /* Tighten metric numbers */
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; }
    /* Sidebar footer */
    .sidebar-footer { font-size: 0.78rem; color: #6B7280; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (all cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_timeseries() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "timeseries_features.csv")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["is_closed"] = (df["store_open"] == 0).astype(int)
    # Short day-of-week label for charts
    dow_short = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["dow_label"] = df["date"].dt.dayofweek.map(dow_short)
    month_short = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df["month_label"] = df["date"].dt.month.map(month_short)
    return df


@st.cache_resource(show_spinner=False)
def load_prophet_model():
    path = os.path.join(ROOT, "models", "best_model.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def load_mlflow_runs() -> pd.DataFrame:
    tracking_uri = os.path.join(ROOT, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiments = mlflow.search_experiments()
        if not experiments:
            return pd.DataFrame()
        exp_ids = [e.experiment_id for e in experiments]
        runs = mlflow.search_runs(experiment_ids=exp_ids)
        return runs
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_best_params() -> dict:
    path = os.path.join(ROOT, "models", "phase4_best_params.json")
    with open(path) as fh:
        return json.load(fh)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def prophet_forecast(model, df: pd.DataFrame, horizon: int = 0) -> pd.DataFrame:
    """Return full Prophet forecast covering train + test window + *horizon* extra days."""
    # Number of periods beyond training data = test rows + extra
    test_rows = int((df["date"] > TRAIN_END).sum())
    n_periods  = test_rows + horizon

    future = model.make_future_dataframe(periods=n_periods)

    # Conditionally inject store_open regressor
    if hasattr(model, "extra_regressors") and "store_open" in model.extra_regressors:
        store_map = df.set_index("date")["store_open"].to_dict()
        future["store_open"] = future["ds"].map(store_map).fillna(1).astype(int)

    fc = model.predict(future)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        fc[col] = fc[col].clip(lower=0)
    return fc


def section_divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def champion_badge(label: str):
    st.markdown(f'<div class="badge-champion">🏆 {label}</div>', unsafe_allow_html=True)
    st.markdown("")


def parse_mlflow_runs(runs: pd.DataFrame):
    """
    Normalise column names that differ between MLflow versions
    (dots vs underscores) and return a clean display DataFrame.
    """
    # Try both naming conventions
    metric_map = {}
    for key, label in [
        ("metrics.MAE",       "MAE"),
        ("metrics.RMSE",      "RMSE"),
        ("metrics.MAPE_pct",  "MAPE (%)"),
        ("metrics.Bias",      "Bias"),
        ("metrics.R2",        "R²"),
        ("metrics.fit_time_s","Fit time (s)"),
    ]:
        if key in runs.columns:
            metric_map[key] = label

    name_col = next(
        (c for c in ["tags.mlflow.runName", "run_name"] if c in runs.columns),
        None,
    )

    cols = list(metric_map.keys())
    df = runs[cols].copy().rename(columns=metric_map)

    if name_col:
        df.insert(0, "Run", runs[name_col].values)

    df = df.dropna(subset=["MAE"]).sort_values("R²", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

def page_forecast():
    st.title("📈 Sales Forecast — Champion Model")
    st.caption(
        "**Prophet (baseline)** · 90-day held-out test window (Jan – Mar 2014) · "
        "Corporación Favorita, Ecuador"
    )

    # ── Sidebar controls ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Forecast settings")
    horizon    = st.sidebar.slider("Extra days to forecast beyond Mar 31 2014", 0, 30, 14)
    show_ci    = st.sidebar.toggle("Show 80 % confidence interval", value=True)
    show_train = st.sidebar.toggle("Show training data", value=True)
    show_closed= st.sidebar.toggle("Highlight store-closed days", value=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    with st.spinner("Loading model & data…"):
        df    = load_timeseries()
        model = load_prophet_model()

    train = df[df["date"] <= TRAIN_END]
    test  = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]

    # ── Predict ────────────────────────────────────────────────────────────────
    fc = prophet_forecast(model, df, horizon=horizon)

    # Align test predictions with actuals
    fc_idx   = fc.set_index("ds")
    test_pred = fc_idx.loc[fc_idx.index.isin(test["date"]), "yhat"]
    test_act  = test.set_index("date")["unit_sales"]
    common    = test_pred.index.intersection(test_act.index)

    metrics = evaluate_model(
        test_act.loc[common].values,
        test_pred.loc[common].values,
        "Prophet (baseline)",
        fit_time=0.06,
    )

    # ── Metric cards ───────────────────────────────────────────────────────────
    st.markdown("### 🏆 Champion Model — Test Window Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE",       f"{metrics['MAE']:.1f}")
    c2.metric("RMSE",      f"{metrics['RMSE']:.1f}")
    c3.metric("MAPE",      f"{metrics['MAPE (%)']:.1f} %")
    c4.metric("Bias",      f"{metrics['Bias']:.1f}")
    c5.metric("R²",        f"{metrics['R²']:.3f}")

    section_divider()

    # ── Main forecast chart ────────────────────────────────────────────────────
    fig = go.Figure()

    if show_train:
        fig.add_trace(go.Scatter(
            x=train["date"], y=train["unit_sales"],
            name="Training actuals",
            mode="lines",
            line=dict(color="#4B5563", width=1.2),
            opacity=0.65,
        ))

    fig.add_trace(go.Scatter(
        x=test["date"], y=test["unit_sales"],
        name="Actuals (test)",
        mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(size=4, color=GREEN),
    ))

    if show_ci:
        upper = fc_idx["yhat_upper"]
        lower = fc_idx["yhat_lower"]
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(124,58,237,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80 % CI",
        ))

    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"],
        name="Prophet forecast",
        mode="lines",
        line=dict(color=ACCENT, width=2.2, dash="dot"),
    ))

    # Vertical split line — use add_shape to avoid pandas 2.x Timestamp arithmetic bug
    fig.add_shape(
        type="line", x0=TRAIN_END, x1=TRAIN_END, y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="#6B7280", width=1.5),
    )
    fig.add_annotation(
        x=TRAIN_END, y=0.97, yref="paper",
        text="Train / Test split", showarrow=False,
        xanchor="left", xshift=6,
        font=dict(color="#9CA3AF", size=11),
    )

    # Future shading
    if horizon > 0:
        fut_start = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)
        fig.add_vrect(
            x0=fut_start, x1=fc["ds"].max(),
            fillcolor="rgba(6,182,212,0.07)", line_width=0,
        )
        fig.add_annotation(
            x=fut_start, y=0.97, yref="paper",
            text=f"+{horizon} d forecast", showarrow=False,
            xanchor="left", xshift=6,
            font=dict(color=CYAN, size=11),
        )

    # Store-closed markers
    if show_closed:
        closed_days = df[df["store_open"] == 0]["date"].tolist()
        for d in closed_days:
            fig.add_vrect(
                x0=d, x1=d + pd.Timedelta(days=1),
                fillcolor="rgba(239,68,68,0.25)", line_width=0,
            )
        if closed_days:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", color="rgba(239,68,68,0.5)", size=10),
                name="Store closed",
            ))

    fig.update_layout(
        template=PLOTLY_TMPL,
        title="Prophet Forecast with 80 % Confidence Interval",
        xaxis_title="Date", yaxis_title="Unit Sales",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=CHART_MARGIN,
    )
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Residual charts ────────────────────────────────────────────────────────
    st.markdown("### 📉 Residual Analysis — Test Window")
    residuals = test_act.loc[common] - test_pred.loc[common]

    col_a, col_b = st.columns(2)

    with col_a:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=common, y=residuals,
            mode="lines+markers",
            line=dict(color=ACCENT, width=1.8),
            marker=dict(size=4),
            name="Residual",
        ))
        fig_r.add_hline(y=0, line_dash="dash", line_color="#6B7280")
        fig_r.update_layout(
            template=PLOTLY_TMPL,
            title="Residuals over time",
            xaxis_title="Date", yaxis_title="Actual − Predicted",
            height=320, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=residuals, nbinsx=20,
            marker_color=ACCENT, opacity=0.8, name="Residuals",
        ))
        fig_h.update_layout(
            template=PLOTLY_TMPL,
            title="Residual distribution",
            xaxis_title="Residual value", yaxis_title="Count",
            height=320, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.caption(
        "Negative bias (mean residual ≈ −10.6) confirms systematic under-prediction — "
        "a pattern shared by all models in this project. Root cause: only ~335 training rows "
        "after lag trimming. The residual distribution is roughly symmetric, confirming no "
        "directional drift in forecast errors."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_comparison():
    st.title("📊 Model Comparison")
    st.caption(
        "All models evaluated on the **same** 90-day held-out test window · "
        "metrics stored in MLflow · Phase 4"
    )

    runs = load_mlflow_runs()

    if runs.empty:
        st.error("No MLflow runs found. Please run notebook 04 first to populate the experiment.")
        return

    df = parse_mlflow_runs(runs)

    if df.empty:
        st.warning("Runs found but no metric data. Check that notebook 04 has been executed.")
        return

    # ── Champion banner ────────────────────────────────────────────────────────
    best = df.iloc[0]
    run_label = best.get("Run", "—")
    champion_badge(
        f"Champion: {run_label}  ·  R² = {best['R²']:.3f}  ·  MAE = {best['MAE']:.1f}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best MAE",  f"{df['MAE'].min():.1f}",
              delta=f"vs worst: −{df['MAE'].max() - df['MAE'].min():.1f}",
              delta_color="inverse")
    c2.metric("Best RMSE", f"{df['RMSE'].min():.1f}")
    c3.metric("Best R²",   f"{df['R²'].max():.3f}")
    c4.metric("Runs logged", len(df))

    section_divider()

    # ── Sortable table ─────────────────────────────────────────────────────────
    st.markdown("### All Runs — sorted by R²")

    fmt = {}
    for col in df.select_dtypes("float").columns:
        fmt[col] = "{:.3f}" if col == "R²" else "{:.1f}"

    def _color_r2(val):
        try:
            v = float(val)
            return f"color: {'#10B981' if v > 0 else '#EF4444'}"
        except Exception:
            return ""

    styled = (
        df.style
        .map(_color_r2, subset=["R²"])
        .format(fmt)
        .background_gradient(cmap="RdYlGn_r", subset=["MAE", "RMSE"],  axis=0)
        .background_gradient(cmap="RdYlGn",   subset=["R²"],           axis=0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    section_divider()

    # ── Bar charts ─────────────────────────────────────────────────────────────
    if "Run" in df.columns:
        bar_colors = [MODEL_COLORS.get(r, ACCENT) for r in df["Run"]]
    else:
        bar_colors = [ACCENT] * len(df)

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure(go.Bar(
            x=df.get("Run", df.index), y=df["MAE"],
            marker_color=bar_colors, name="MAE",
            text=df["MAE"].round(1), textposition="auto",
        ))
        fig.update_layout(
            template=PLOTLY_TMPL, title="MAE by Model (lower = better)",
            xaxis_title="", yaxis_title="MAE",
            height=390, margin=CHART_MARGIN,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure(go.Bar(
            x=df.get("Run", df.index), y=df["R²"],
            marker_color=bar_colors, name="R²",
            text=df["R²"].round(3), textposition="auto",
        ))
        fig.update_layout(
            template=PLOTLY_TMPL, title="R² by Model (higher = better)",
            xaxis_title="", yaxis_title="R²",
            height=390, margin=CHART_MARGIN,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── MAE vs MAPE scatter ────────────────────────────────────────────────────
    if "MAPE (%)" in df.columns and "Run" in df.columns:
        section_divider()
        st.markdown("### Error Profile — MAE vs MAPE")
        fig = px.scatter(
            df, x="MAE", y="MAPE (%)",
            color="Run",
            color_discrete_map=MODEL_COLORS,
            text="Run",
            template=PLOTLY_TMPL,
            title="Lower-left corner = best model",
        )
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_layout(
            height=400, showlegend=False, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "All models exhibit negative bias (systematic under-prediction). "
        "XGBoost underperformed statistical models due to data scarcity — "
        "only ~335 training rows after lag trimming is too few for tree ensembles. "
        "Prophet's simplicity gave it an edge on this small dataset."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ═══════════════════════════════════════════════════════════════════════════════

def page_eda():
    st.title("🔍 Exploratory Data Analysis")
    st.caption(
        "Phase 1 findings · Corporación Favorita grocery data · "
        "Jan 2013 – Mar 2014 · 454 observations"
    )

    df = load_timeseries()

    # ── Summary cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rows",        len(df))
    c2.metric("Training rows",     int((df["date"] <= TRAIN_END).sum()))
    c3.metric("Test rows",         int(((df["date"] >= TEST_START) & (df["date"] <= TEST_END)).sum()))
    c4.metric("Store-closed days", int(df["is_closed"].sum()))
    c5.metric("Features (Phase 3)", 20)

    section_divider()

    # ── Full time series ───────────────────────────────────────────────────────
    st.markdown("### Full Time Series + Oil Price")

    train_df = df[df["date"] <= TRAIN_END]
    test_df  = df[df["date"] >  TRAIN_END]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_df["date"], y=train_df["unit_sales"],
        name="Training", mode="lines",
        line=dict(color=ACCENT, width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=test_df["unit_sales"],
        name="Test window", mode="lines",
        line=dict(color=GREEN, width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["dcoilwtico"],
        name="WTI Oil price",
        mode="lines",
        line=dict(color=AMBER, width=1, dash="dot"),
        yaxis="y2",
        opacity=0.7,
    ))

    closed_days = df[df["store_open"] == 0]["date"]
    for d in closed_days:
        fig.add_shape(
            type="line", x0=d, x1=d, y0=0, y1=1, yref="paper",
            line=dict(color=RED, width=1),
            opacity=0.5,
        )

    fig.add_shape(
        type="line", x0=TRAIN_END, x1=TRAIN_END, y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="#6B7280", width=1.5),
    )
    fig.add_annotation(
        x=TRAIN_END, y=0.97, yref="paper",
        text="Train / Test", showarrow=False,
        xanchor="left", xshift=6,
        font=dict(color="#9CA3AF", size=11),
    )

    fig.update_layout(
        template=PLOTLY_TMPL,
        title="Unit Sales (left axis) & WTI Oil Price (right axis)",
        xaxis_title="Date",
        yaxis=dict(title="Unit Sales"),
        yaxis2=dict(title="WTI (USD)", overlaying="y", side="right", showgrid=False),
        height=420,
        legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
        margin=dict(l=44, r=70, t=70, b=44),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red vertical lines = store-closed days (Dec 25 2013 & Jan 1 2014). Dashed = train/test split.")

    section_divider()

    # ── Seasonality ────────────────────────────────────────────────────────────
    st.markdown("### Seasonality Patterns")
    col_a, col_b = st.columns(2)

    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    with col_a:
        open_df   = df[df["store_open"] == 1]
        dow_stats = (
            open_df.groupby("dow_label")["unit_sales"]
            .agg(["mean", "std"])
            .reindex(DOW_ORDER)
        )
        bar_colors_dow = [ACCENT if d in ("Sat", "Sun") else "#3F4866" for d in DOW_ORDER]

        fig_dow = go.Figure(go.Bar(
            x=dow_stats.index,
            y=dow_stats["mean"],
            error_y=dict(type="data", array=dow_stats["std"].values, visible=True),
            marker_color=bar_colors_dow,
            name="Mean sales",
        ))
        fig_dow.update_layout(
            template=PLOTLY_TMPL,
            title="Avg Sales by Day of Week",
            yaxis_title="Mean Unit Sales",
            height=360, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        st.caption("Weekend uplift: **Sat +33 %**, **Sun +45 %** vs weekday average — the dominant seasonal signal (s = 7).")

    MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    with col_b:
        month_stats = (
            open_df.groupby("month_label")["unit_sales"]
            .mean()
            .reindex(MONTH_ORDER)
            .dropna()
        )
        fig_mon = go.Figure(go.Bar(
            x=month_stats.index,
            y=month_stats.values,
            marker_color=CYAN,
            name="Mean sales",
        ))
        fig_mon.update_layout(
            template=PLOTLY_TMPL,
            title="Avg Sales by Month",
            yaxis_title="Mean Unit Sales",
            height=360, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig_mon, use_container_width=True)
        st.caption("Data spans 15 months. Feb–Mar 2014 appear only in the test window.")

    section_divider()

    # ── Distributions ──────────────────────────────────────────────────────────
    st.markdown("### Sales Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        open_sales = df[df["store_open"] == 1]["unit_sales"]
        fig = go.Figure(go.Histogram(
            x=open_sales, nbinsx=30,
            marker_color=ACCENT, opacity=0.85, name="Open days",
        ))
        fig.update_layout(
            template=PLOTLY_TMPL,
            title="Distribution of Daily Unit Sales (store-open days)",
            xaxis_title="Unit Sales", yaxis_title="Count",
            height=330, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure(go.Box(
            x=df["dow_label"], y=df["unit_sales"],
            marker_color=ACCENT,
            boxmean="sd",
            name="Unit Sales",
        ))
        fig.update_layout(
            template=PLOTLY_TMPL,
            title="Sales Box Plot by Day of Week",
            xaxis=dict(categoryorder="array", categoryarray=DOW_ORDER),
            yaxis_title="Unit Sales",
            height=330, margin=CHART_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── ADF stationarity result ────────────────────────────────────────────────
    st.markdown("### 📐 Key Statistical Findings — Phase 1")
    col1, col2, col3 = st.columns(3)
    col1.success("✅ **Stationary series**  \nADF test p < 0.05 → no differencing needed")
    col2.info("📅 **Weekly seasonality**  \ns = 7 — weekend uplift confirmed")
    col3.warning("⚠️ **Train/test split**  \nTrain ≤ 2013-12-31 | Test 2014 Q1")

    section_divider()

    # ── Raw data explorer ──────────────────────────────────────────────────────
    with st.expander("🔎 Raw Data Explorer"):
        drop_cols = ["dow_label", "month_label", "is_closed"]
        st.dataframe(
            df.drop(columns=[c for c in drop_cols if c in df.columns]),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HYPEROPT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_hyperopt():
    st.title("⚙️ HyperOpt Tuning Results")
    st.caption(
        "Phase 4 · 50 Hyperopt trials per model · 4 models · "
        "MLflow experiment tracking · Champion selected by lowest MAE"
    )

    runs        = load_mlflow_runs()
    best_params = load_best_params()
    champion    = best_params.get("champion", "Prophet (baseline)")

    if runs.empty:
        st.error("No MLflow runs found. Please run notebook 04 first.")
        return

    df = parse_mlflow_runs(runs)
    if df.empty:
        st.warning("Runs found but metrics are missing. Run notebook 04 to log metrics.")
        return

    champion_badge(
        f"Champion: {champion}  ·  "
        f"Chosen by MAE on 90-day held-out test set"
    )

    section_divider()

    # ── Baseline vs Tuned delta table ──────────────────────────────────────────
    if "Run" in df.columns:
        df["_type"]   = df["Run"].apply(lambda x: "Baseline" if "baseline" in str(x).lower() else "Tuned")
        df["_family"] = df["Run"].apply(
            lambda x: str(x).replace("_baseline", "").replace("_tuned", "").replace("_", " ").title()
        )

        baseline_df = df[df["_type"] == "Baseline"].set_index("_family")
        tuned_df    = df[df["_type"] == "Tuned"].set_index("_family")
        common_fams = baseline_df.index.intersection(tuned_df.index)

        if len(common_fams) > 0:
            st.markdown("### 📊 Baseline vs Tuned — Performance Delta")
            rows = []
            for fam in common_fams:
                b = baseline_df.loc[fam]
                t = tuned_df.loc[fam]
                rows.append({
                    "Model":          fam,
                    "MAE (baseline)": b["MAE"],
                    "MAE (tuned)":    t["MAE"],
                    "Δ MAE":          round(b["MAE"] - t["MAE"], 1),
                    "R² (baseline)":  b["R²"],
                    "R² (tuned)":     t["R²"],
                    "Δ R²":           round(t["R²"] - b["R²"], 3),
                })
            delta_df = pd.DataFrame(rows)

            def _color_delta(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color: #10B981; font-weight: 600"
                    elif v < 0:
                        return "color: #EF4444; font-weight: 600"
                except Exception:
                    pass
                return ""

            styled_delta = (
                delta_df.style
                .map(_color_delta, subset=["Δ MAE", "Δ R²"])
                .format({
                    "MAE (baseline)": "{:.1f}", "MAE (tuned)": "{:.1f}", "Δ MAE": "{:+.1f}",
                    "R² (baseline)":  "{:.3f}", "R² (tuned)":  "{:.3f}", "Δ R²": "{:+.3f}",
                })
            )
            st.dataframe(styled_delta, use_container_width=True, hide_index=True)
            st.caption("Δ MAE > 0 = tuned MAE is lower (improvement).  Δ R² > 0 = higher R² (improvement).")

            # ── Grouped bar ───────────────────────────────────────────────────
            section_divider()
            st.markdown("### MAE: Baseline vs Tuned by Model Family")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Baseline",
                x=delta_df["Model"], y=delta_df["MAE (baseline)"],
                marker_color="#3F4866",
                text=delta_df["MAE (baseline)"].round(1), textposition="auto",
            ))
            fig.add_trace(go.Bar(
                name="Tuned",
                x=delta_df["Model"], y=delta_df["MAE (tuned)"],
                marker_color=ACCENT,
                text=delta_df["MAE (tuned)"].round(1), textposition="auto",
            ))
            fig.update_layout(
                template=PLOTLY_TMPL,
                barmode="group",
                title="HyperOpt Impact on MAE — Baseline vs Tuned",
                yaxis_title="MAE (lower = better)",
                height=390, margin=CHART_MARGIN,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── R² grouped bar ────────────────────────────────────────────────
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Baseline",
                x=delta_df["Model"], y=delta_df["R² (baseline)"],
                marker_color="#3F4866",
                text=delta_df["R² (baseline)"].round(3), textposition="auto",
            ))
            fig2.add_trace(go.Bar(
                name="Tuned",
                x=delta_df["Model"], y=delta_df["R² (tuned)"],
                marker_color=CYAN,
                text=delta_df["R² (tuned)"].round(3), textposition="auto",
            ))
            fig2.update_layout(
                template=PLOTLY_TMPL,
                barmode="group",
                title="R²: Baseline vs Tuned",
                yaxis_title="R² (higher = better)",
                height=390, margin=CHART_MARGIN,
            )
            st.plotly_chart(fig2, use_container_width=True)

    section_divider()

    # ── All runs table ─────────────────────────────────────────────────────────
    with st.expander("📋 All MLflow Runs"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    section_divider()

    # ── Best hyperparameters ───────────────────────────────────────────────────
    st.markdown("### 🎛️ Best Hyperparameters — Phase 4 (50 trials each)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔮 Prophet (Champion)")
        prophet_p = best_params.get("prophet", {})
        for k, v in prophet_p.items():
            disp = f"{v:.6f}" if isinstance(v, float) else str(v)
            st.markdown(f"- **{k}**: `{disp}`")

        st.markdown("#### 🌊 Holt-Winters")
        hw_p = best_params.get("holtwinters", {})
        for k, v in hw_p.items():
            st.markdown(f"- **{k}**: `{v}`")

    with col2:
        st.markdown("#### 📈 SARIMA")
        sarima_p = best_params.get("sarima", {})
        # Reconstruct ARIMA orders for readability
        p,d,q = sarima_p.get("p","-"),sarima_p.get("d","-"),sarima_p.get("q","-")
        P,D,Q = sarima_p.get("P","-"),sarima_p.get("D","-"),sarima_p.get("Q","-")
        st.markdown(f"- **order**: `({p},{d},{q})`")
        st.markdown(f"- **seasonal_order**: `({P},{D},{Q},7)`")

        st.markdown("#### 🌳 XGBoost")
        xgb_p = best_params.get("xgboost", {})
        for k, v in xgb_p.items():
            disp = f"{v:.5f}" if isinstance(v, float) else str(v)
            st.markdown(f"- **{k}**: `{disp}`")

    section_divider()

    # ── Key insight box ────────────────────────────────────────────────────────
    st.info(
        "**Key finding:** Prophet's baseline configuration outperformed its HyperOpt-tuned variant. "
        "With only ~335 training rows, heavy regularisation (low changepoint_prior_scale = 0.001) "
        "may have over-constrained the tuned model. XGBoost consistently underperformed all "
        "statistical baselines — a textbook data-starvation scenario for tree ensembles. "
        "The project's core lesson: **model simplicity wins when data is scarce**."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("📈 MSIT Forecasting")
    st.sidebar.caption("Time Series Project · Phase 5 Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate to",
        [
            "📈 Forecast",
            "📊 Model Comparison",
            "🔍 EDA / Data Explorer",
            "⚙️ HyperOpt Results",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
        <b>Data</b>: Corporación Favorita<br>
        <b>Period</b>: Jan 2013 – Mar 2014<br>
        <b>Models</b>: Prophet · SARIMA<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Holt-Winters · XGBoost<br>
        <b>Champion</b>: Prophet (baseline)<br>
        <b>Optimiser</b>: Hyperopt (50 trials)<br>
        <b>Tracking</b>: MLflow
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_fn = {
        "📈 Forecast":            page_forecast,
        "📊 Model Comparison":    page_model_comparison,
        "🔍 EDA / Data Explorer": page_eda,
        "⚙️ HyperOpt Results":   page_hyperopt,
    }
    page_fn[page]()


if __name__ == "__main__":
    main()
