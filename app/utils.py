"""
Shared data-loading and caching utilities for the Supply Chain Demand Forecasting dashboard.

Every page imports from this module rather than reading CSVs directly -- this keeps the
caching strategy, path configuration, and column contracts in one place (Roadmap Item #26).
No model is loaded or trained here: every function reads a precomputed CSV produced by the
project's notebooks (Items #6, #10, #12-15, #20), consistent with this project's
notebook-to-notebook independence pattern established from Notebook 04 onward.
"""

import os
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATHS -- same convention used in every notebook
# ============================================================
PROJECT_DIR = "/content/drive/MyDrive/Supply_Chain_Demand_Forecasting"
RAW_DIR = f"{PROJECT_DIR}/data/raw"
PROCESSED_DIR = f"{PROJECT_DIR}/data/processed"
MODELS_DIR = f"{PROJECT_DIR}/models"

# When deployed outside of Colab (e.g. Streamlit Community Cloud, or run locally), the app
# falls back to a relative "data/processed" folder next to app.py -- this makes the same
# codebase runnable both from Google Drive (Colab) and from a cloned GitHub repo.
if not os.path.isdir(PROCESSED_DIR):
    PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")


def _read_csv(filename, **kwargs):
    """Small wrapper so every loader shares one consistent path-join and error message."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Missing data file: {filename}. Expected at: {path}")
        st.stop()
    return pd.read_csv(path, **kwargs)


# ============================================================
# OVERVIEW PAGE DATA
# ============================================================
@st.cache_data
def load_overview_stats():
    """Single-row summary stats (Roadmap Item #22 support) -- tiny file, loads instantly."""
    return _read_csv("dashboard_overview_stats.csv").iloc[0].to_dict()


@st.cache_data
def load_daily_trend():
    """Full-history aggregated daily demand (~1,941 rows) -- cheap to load and plot."""
    df = _read_csv("daily_demand_trend.csv", parse_dates=["date"])
    return df


# ============================================================
# DEMAND FORECAST PAGE DATA
# ============================================================
@st.cache_data
def load_forecast_vs_actual():
    """Daily actual vs. forecast for the full catalog, final fold only (~170K rows).
    Precomputed once by the build notebook using the saved Random Forest model --
    the app never runs inference live."""
    df = _read_csv(
        "forecast_vs_actual_final_fold.csv",
        parse_dates=["date"],
        dtype={"store_id": "category", "item_id": "category", "cat_id": "category", "dept_id": "category"},
    )
    return df


# ============================================================
# INVENTORY OPTIMIZATION PAGE DATA
# ============================================================
@st.cache_data
def load_inventory_recommendations():
    """Per-store-item safety stock, reorder point, EOQ, and risk tier (Roadmap Item #20)."""
    df = _read_csv(
        "inventory_policy_recommendations.csv",
        dtype={"store_id": "category", "item_id": "category", "cat_id": "category", "dept_id": "category",
               "risk_tier": "category"},
    )
    return df


# ============================================================
# MODEL INSIGHTS PAGE DATA
# ============================================================
@st.cache_data
def load_comparison_a():
    """Full-catalog, 5-fold Baseline vs. ML comparison (Roadmap Item #15, Comparison A)."""
    return _read_csv("model_comparison_full_catalog.csv")


@st.cache_data
def load_comparison_b():
    """Fair three-way comparison: Baseline vs. ML vs. Classical, same 180 series, final fold
    (Roadmap Item #15, Comparison B)."""
    return _read_csv("model_comparison_fair_sample.csv")


@st.cache_data
def load_ml_fold_metrics():
    """Fold-by-fold ML metrics, used for the stability chart (Roadmap Item #13)."""
    return _read_csv("ml_model_metrics.csv")


@st.cache_data
def load_error_analysis():
    """Error breakdown by category/store/month/event (Roadmap Item #15, Enhancement #3)."""
    return _read_csv("error_analysis_summary.csv")


# ============================================================
# SHARED FILTER HELPERS
# ============================================================
def get_category_options(df, col="cat_id"):
    return sorted(df[col].dropna().unique().tolist())


def get_store_options(df, col="store_id"):
    return sorted(df[col].dropna().unique().tolist())


def to_csv_download(df):
    """Convert a dataframe to UTF-8 CSV bytes for st.download_button, used consistently
    across every page's downloadable-output requirement."""
    return df.to_csv(index=False).encode("utf-8")
