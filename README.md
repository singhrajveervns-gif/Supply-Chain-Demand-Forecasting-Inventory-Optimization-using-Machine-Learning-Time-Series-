<div align="center">

# 📦 Supply Chain Demand Forecasting & Inventory Optimization

### An End-to-End Machine Learning Pipeline for Retail Demand Forecasting and Data-Driven Inventory Policy

*Built on real Walmart sales data (M5 Forecasting Competition) — from raw data to a deployed interactive dashboard*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006400?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9ACD32?logo=lightgbm&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMA-8A2BE2)
![Prophet](https://img.shields.io/badge/Prophet-Meta-0668E1)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF4B4B)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Viz-3F4F75?logo=plotly&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[**Live Dashboard**](#) &nbsp;•&nbsp; [**LinkedIn**](#) &nbsp;•&nbsp; [**Resume**](#) &nbsp;•&nbsp; [**Portfolio**](#)

</div>

---

<p align="center">
  <img src="assets/dashboard_screenshot_overview.png" alt="Dashboard Overview Screenshot" width="850">
  <br>
  <em>[Placeholder — Project Overview page screenshot]</em>
</p>

<p align="center">
  <img src="assets/demo.gif" alt="Dashboard Demo GIF" width="850">
  <br>
  <em>[Placeholder — animated walkthrough of the dashboard]</em>
</p>

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Business Problem](#business-problem)
- [Dataset Overview](#dataset-overview)
- [Project Objectives](#project-objectives)
- [End-to-End Workflow](#end-to-end-workflow)
- [Folder Structure](#folder-structure)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Exploratory Data Analysis Highlights](#exploratory-data-analysis-highlights)
- [Feature Engineering Summary](#feature-engineering-summary)
- [Baseline Models](#baseline-models)
- [Machine Learning Models](#machine-learning-models)
- [Classical Time Series Models](#classical-time-series-models)
- [Model Comparison](#model-comparison)
- [Inventory Optimization](#inventory-optimization)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Final Business Recommendations](#final-business-recommendations)
- [Key Project Results](#key-project-results)
- [Future Improvements](#future-improvements)
- [Installation Guide](#installation-guide)
- [How to Run Locally](#how-to-run-locally)
- [How to Launch the Streamlit Dashboard](#how-to-launch-the-streamlit-dashboard)
- [Project Roadmap](#project-roadmap)
- [References](#references)
- [License](#license)

---

## Executive Summary

Retailers lose money two ways: stocking out (lost sales, lost trust) and overstocking (tied-up capital, waste). This project builds a complete, production-style pipeline that forecasts daily product-level demand and converts that forecast directly into an actionable inventory policy — safety stock, reorder points, and order quantities — for a real subset of Walmart's own M5 forecasting competition data.

Five machine learning models, two statistical baselines, and two classical time series models (SARIMA, Prophet) were trained and evaluated using proper walk-forward cross-validation, not a single lucky train/test split. The winning model — **Random Forest** — improved forecast accuracy by **17.65%** over the best naive baseline at full-catalog scale, and its output was used to derive inventory recommendations for every product-store combination, all surfaced in an interactive Streamlit dashboard.

This is not a Kaggle notebook. Every modeling decision — from the choice of walk-forward validation over K-Fold, to why WMAPE replaces plain MAPE on intermittent demand data, to why classical models were benchmarked on a stratified sample rather than the full catalog — is deliberate and documented, because the goal was to produce work that holds up under real technical interview questioning.

## Business Problem

**Problem statement:** a retailer cannot manually forecast demand for thousands of product-store combinations with any consistency, yet inventory decisions (how much safety stock to hold, when to reorder, how much to order) depend entirely on getting that forecast right. Poor forecasts compound into either stockouts (lost revenue, damaged customer trust) or excess inventory (holding costs, capital inefficiency, waste).

**Business goals:**
- Forecast daily demand accurately enough to support automated inventory decisions
- Quantify forecast uncertainty, not just a point estimate — uncertainty is what safety stock is built from
- Translate forecasts into a concrete, defensible inventory policy
- Make the results usable by a non-technical stakeholder through an interactive dashboard

**Why this matters to companies:** this is the exact problem behind the Sales & Operations Planning (S&OP) function at every retailer, FMCG company, and distributor — and it's why consulting firms like Deloitte, EY, and Accenture run dedicated supply chain analytics practices.

## Dataset Overview

**Source:** [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart), a real, industry-benchmark dataset used across the forecasting research and applied ML community.

**Scope of this project:** subsetted to **2 stores** (`CA_1`, `TX_1`) across **all 3 product categories** (Foods, Household, Hobbies) to keep the project tractable on Google Colab's free tier while preserving the full hierarchical structure (state → store → category → department → item).

| Property | Value |
|---|---|
| Raw files used | `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` |
| Final engineered dataset | **9,415,478 rows**, 51 columns |
| Date range | Full M5 history — approximately 5.4 years (~1,941 days) |
| Stores | CA_1, TX_1 |
| Categories | Foods, Household, Hobbies |
| Granularity | Daily, per store-item combination |
| Target variable | `units_sold` (daily demand) |

The dataset intentionally has **no actual inventory / stock-on-hand column** — this is a deliberate, stated design choice (see [Inventory Optimization](#inventory-optimization)) rather than a limitation glossed over.

## Project Objectives

1. Build a real, defensible demand forecasting pipeline — not a toy Kaggle notebook
2. Compare baseline, machine learning, and classical statistical forecasting approaches on equal footing
3. Apply proper time series validation methodology throughout (no data leakage, no random K-Fold)
4. Derive an actionable inventory policy analytically from forecast uncertainty
5. Deploy the results as an interactive, filterable dashboard
6. Produce work suitable for technical discussion in Data Analyst / Data Scientist / ML Engineer interviews

## End-to-End Workflow

```
Raw M5 Data (Kaggle)
        │
        ▼
[1] Data Understanding, Cleaning, EDA  ──────────────►  cleaned_sales_data.csv
        │
        ▼
[2] Time Series Theory (ADF, ACF/PACF, walk-forward CV)
        │
        ▼
[3] Feature Engineering (lags, rolling stats, EMA, calendar, SNAP, price)  ──►  feature_engineered_data.csv
        │
        ▼
[4] Baseline Models (Naive, Seasonal Naive)  ─────────►  baseline_metrics.csv
        │
        ▼
[5] Machine Learning Models (5 models, walk-forward CV)  ─────►  ml_model_metrics.csv + saved models
        │
        ▼
[6] Classical Time Series Models (AR→SARIMAX, Prophet)  ─────►  classical_model_metrics.csv
        │
        ▼
[7] Model Evaluation — 3-way comparison, SHAP, error analysis  ─────►  model_comparison_*.csv
        │
        ▼
[8] Inventory Optimization (safety stock, reorder point, EOQ)  ─────►  inventory_policy_recommendations.csv
        │
        ▼
[9] Streamlit Dashboard (interactive, filterable, deployable)
```

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Pipeline Architecture Diagram" width="750">
  <br>
  <em>[Placeholder — full pipeline architecture diagram]</em>
</p>

## Folder Structure

```
Supply_Chain_Demand_Forecasting/
│
├── data/
│   ├── raw/                              # Original M5 files (calendar, sales, prices)
│   └── processed/                        # Every cleaned/engineered/output CSV produced by the pipeline
│
├── notebooks/
│   ├── 01_data_understanding_cleaning_eda.ipynb
│   ├── 02_time_series_theory.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_naive_model.ipynb
│   ├── 05_ml_models.ipynb
│   ├── 06_classical_time_series_models.ipynb
│   ├── 07_model_evaluation_comparison.ipynb
│   ├── 08_inventory_optimization.ipynb
│   └── 09_streamlit_dashboard.ipynb
│
├── models/                               # Saved final-fold model artifacts (.pkl)
│
├── checkpoints/                          # Fault-tolerant training checkpoints (per notebook)
│
├── app/
│   ├── app.py                            # Dashboard entry point (Project Overview)
│   ├── utils.py                          # Shared caching / data-loading functions
│   ├── requirements.txt
│   └── pages/
│       ├── 1_Demand_Forecast.py
│       ├── 2_Inventory_Dashboard.py
│       └── 3_Model_Insights.py
│
├── assets/                               # Screenshots, diagrams, demo GIF
├── docs/                                 # Resume bullets, LinkedIn post, interview prep notes
├── README.md
├── LICENSE
└── .gitignore
```

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost, LightGBM |
| Classical time series | Statsmodels (AR, MA, ARIMA, SARIMA, SARIMAX), Prophet |
| Explainability | SHAP |
| Visualization (analysis) | Matplotlib, Seaborn |
| Visualization (dashboard) | Plotly |
| Dashboard framework | Streamlit |
| Environment | Google Colab (development), Google Drive (storage) |
| Version control | Git, GitHub |

## Methodology

- **Walk-forward validation throughout.** Every model — baseline, ML, and classical — was evaluated using `TimeSeriesSplit` (5 folds, 28-day horizon), never a random train/test split or K-Fold, which would leak future information into training. This was demonstrated concretely (not just asserted) in Notebook 02 and again in Notebook 05.
- **Leak-safe feature engineering.** Every lag, rolling, and EMA feature is computed using `shift(1)` before any window operation, guaranteeing no feature for day *t* ever uses information from day *t* itself.
- **WMAPE over plain MAPE.** This dataset has meaningful intermittent/zero-demand periods, where plain MAPE is undefined. WMAPE (aggregate absolute error ÷ aggregate actual demand) is used throughout instead.
- **Pooled metrics for panel-scale evaluation.** When averaging performance across many series, a single zero-demand series can make an individual WMAPE ratio undefined. Pooled WMAPE/RMSSE (sums computed first, then divided once) avoids this — used consistently from Notebook 06 onward.
- **Fair comparison discipline.** Classical models were only tractable on a stratified 180-series sample (final fold). Rather than comparing that directly against full-catalog, 5-fold ML metrics, this project runs **two explicitly separate comparisons** — one full-catalog/5-fold (baseline vs. ML), one same-sample/same-fold (baseline vs. ML vs. classical) — so no conclusion mixes incompatible populations.

## Exploratory Data Analysis Highlights

- **Strong weekly seasonality** — weekend demand is consistently higher than mid-week, confirmed both visually and via ACF spikes at lag 7 and its multiples.
- **FOODS is the dominant category by volume**, consistent with M5's known characteristics.
- **SNAP days show a measurable positive demand lift**, confirmed via direct EDA and later via a SARIMAX exogenous regressor with the expected coefficient sign.
- **Named calendar events** (Christmas, Thanksgiving, etc.) produce some of the largest single-day demand swings in the dataset.
- **Meaningful zero-sales rate** at the individual store-item-day level — this intermittent-demand characteristic shaped multiple downstream decisions (WMAPE over MAPE, pooled metrics, zero-sales-streak flagging rather than deletion).

## Feature Engineering Summary

| Feature Family | Details |
|---|---|
| Lag features | 1, 7, 14, 28 days |
| Rolling statistics | Mean, std, median over 7/14/28-day windows |
| Exponential moving average | Spans of 7 and 14 days |
| Calendar features | Day of week, month, week of year, weekend flag, cyclic sin/cos encoding |
| Promotion / event features | SNAP flag, days-until-next-SNAP, binary event flag |
| Price features | Rolling 28-day average price, relative price, price-change flag |

All engineering was built and validated at panel scale (thousands of independent store-item series simultaneously) using grouped, leak-safe transforms — not just a single toy series.

## Baseline Models

| Model | Definition | Avg. WMAPE (Full Catalog, 5-Fold) |
|---|---|---|
| Naive | Tomorrow = today (`lag_1`) | **87.12%** |
| Seasonal Naive | This weekday = same weekday last week (`lag_7`) | 88.75% |

**Notable finding:** Seasonal Naive underperformed plain Naive at the individual store-item level — the opposite of what aggregate weekly seasonality would suggest. At fine granularity, demand is intermittent enough that a single week-old observation is a noisier point estimate than yesterday's value, despite the real aggregate weekly cycle. This distinction (cyclical patterns in *total* demand vs. point-forecast reliability per series) directly informed how classical seasonal models were interpreted later.

## Machine Learning Models

Five models trained with identical, leak-safe features and identical walk-forward folds: **Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM.**

| Model | Avg. WMAPE (Full Catalog, 5-Fold) | Improvement vs. Best Baseline |
|---|---|---|
| **Random Forest** | **71.75%** | **+17.65%** |
| LightGBM | 72.00% | +17.36% |
| XGBoost | 72.00% | +17.36% |
| Decision Tree | 72.19% | +17.14% |
| Linear Regression | 74.37% | +14.64% |

All five models beat both baselines. Random Forest was selected as the primary production candidate — it achieved the lowest average error **and** the most consistent performance across all 5 folds (lowest fold-to-fold variance), which matters more than a marginally lower average for a model that will be redeployed on a recurring schedule.

## Classical Time Series Models

Implemented and taught from first principles: **Moving Average, AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, Prophet**, with every parameter choice (differencing order, seasonal period) justified by ADF/ACF/PACF diagnostics rather than guessed.

**On the aggregated (total-demand) series:**

| Model | WMAPE |
|---|---|
| **Prophet** | **6.97%** |
| SARIMA | 8.80% |
| SARIMAX | 10.46% |
| ARIMA | 12.75% |
| AR / ARMA | ~13.5% |
| MA / Smoothing | ~15.0–15.3% |

**At panel scale** (180-series stratified sample, final fold, pooled WMAPE):

| Model | Pooled WMAPE |
|---|---|
| **SARIMA** | **68.30%** |
| Prophet | 70.81% |

The ranking reversed between aggregate and panel scale — Prophet's smooth curve-fitting excelled on the clean aggregated series, while SARIMA's explicit autoregressive error correction generalized better to noisy, individual-series-level demand. This aggregation-level dependency is a deliberate, documented finding rather than a contradiction.

## Model Comparison

Two explicitly separate, fair comparisons (see [Methodology](#methodology)):

**Comparison A — Full catalog, 5-fold (Baseline vs. ML):** Random Forest wins at 71.75% WMAPE, a 17.65% relative improvement over the best baseline.

**Comparison B — Fair three-way, same 180 series, same final fold (pooled WMAPE):**

| Family | WMAPE Range |
|---|---|
| Machine Learning (5 models) | ~61.3% – 63.7% |
| Classical (SARIMA, Prophet) | 68.3% – 70.8% |

On identical ground, the ML models outperformed both classical models — while classical models remain valuable for their interpretability and lower deployment complexity on a smaller catalog.

**Explainability (SHAP):** global feature importance, a local single-prediction explanation, and a dependence plot were generated for the winning model, confirming that recent-demand and weekly-cycle features (the lag and rolling-window family) are the dominant drivers of every prediction — consistent with the weekly seasonality established during EDA.

**Detailed error analysis** broke down forecast error by product category, store, month, and event day, surfacing which conditions the model handles worst and translating each into a concrete recommendation (e.g., event-specific lift features as a v2 improvement, rather than a single generic event flag).

## Inventory Optimization

Since M5 provides no actual stock-on-hand data, **safety stock, reorder point, and Economic Order Quantity (EOQ) are derived analytically** from the winning model's forecast and its forecast-error distribution — exactly how a demand planning team operates without a live ERP feed, rather than reading values off a column that doesn't exist in the real data.

| Metric | Formula |
|---|---|
| Safety Stock | `Z × σ(forecast error) × √(Lead Time)` |
| Reorder Point | `(Avg. Daily Demand × Lead Time) + Safety Stock` |
| EOQ | `√((2 × Annual Demand × Order Cost) / Holding Cost per Unit)` |

**Business assumptions** (explicit and editable, not hidden in the formulas): 7-day lead time, 95% service level (Z = 1.645), and standard retail benchmark ordering/holding cost rates.

Products are additionally classified into **risk tiers** (high stockout risk, moderate, overstock risk) based on each product's demand coefficient of variation — a defensible, data-grounded framing given that no real inventory feed exists to compare against.

## Streamlit Dashboard

An interactive, multipage dashboard (built with Streamlit + Plotly) that consumes only the pipeline's saved outputs — **no model is loaded or retrained inside the deployed app**, keeping it fast and lightweight even on free-tier hosting.

| Page | Contents |
|---|---|
| **Overview** | Project KPIs, full-history demand trend |
| **Demand Forecast** | Category/store filters, product search, actual-vs-forecast chart, downloadable CSV |
| **Inventory Dashboard** | Safety stock by category, reorder alert list, risk-tier breakdown, downloadable CSV |
| **Model Insights** | Full model comparison (baseline/ML/classical), fold stability, error analysis breakdown |

<p align="center">
  <img src="assets/dashboard_forecast_page.png" alt="Demand Forecast Page Screenshot" width="800">
  <br>
  <em>[Placeholder — Demand Forecast page screenshot]</em>
</p>

<p align="center">
  <img src="assets/dashboard_inventory_page.png" alt="Inventory Dashboard Page Screenshot" width="800">
  <br>
  <em>[Placeholder — Inventory Dashboard page screenshot]</em>
</p>

**🔗 Live demo:** [Streamlit deployment link — placeholder](#)

## Final Business Recommendations

1. **Deploy Random Forest** as the production forecasting model — it delivers the strongest full-catalog accuracy and the most consistent fold-to-fold performance of any model tested.
2. **Prioritize forecast-quality investment on the highest-volume, highest-error category** identified in the error analysis, since that is where absolute forecast error translates to the largest dollar impact.
3. **Build event-specific demand-lift features** (rather than a single generic event flag) as the highest-leverage next feature engineering improvement, since event days remain disproportionately hard to forecast even after including SNAP and calendar features.
4. **Treat high-stockout-risk products flagged by the inventory dashboard as requiring human review**, not full automation — these are precisely the products where forecast uncertainty is largest relative to their own demand level.
5. **Reassess classical models (SARIMA) as a lighter-weight alternative** for smaller catalogs or lower-resource deployments, given their competitive — if not fully ML-matching — accuracy and lower operational complexity.

## Key Project Results

- **9,415,478 rows** processed across a real, multi-year Walmart dataset
- **5 ML models, 2 baselines, and 8 classical model types** implemented and rigorously compared
- **17.65%** relative WMAPE improvement (Random Forest vs. best baseline, full catalog)
- **Zero data leakage** — validated through explicit K-Fold vs. walk-forward demonstrations at two separate points in the project
- **Full inventory policy** (safety stock, reorder point, EOQ, risk tier) generated for every product-store combination
- **Deployed, interactive dashboard** consuming only precomputed outputs — no live model inference required

## Future Improvements

- Extend the classical-model panel evaluation from a 180-series sample to the full catalog, and from the final fold to all 5 folds, given sufficient compute budget
- Engineer event-specific demand-lift multipliers in place of the current generic binary event flag
- Extend the detailed error analysis across all 5 folds to build a genuine full-year seasonal error profile, rather than the current 28-day snapshot
- Add store-specific model variants if regional demand patterns are found to diverge meaningfully
- Integrate real inventory / stock-on-hand data, if available, to validate the analytically-derived safety stock policy against actual stockout/overstock outcomes
- Add automated hyperparameter tuning (e.g. Optuna) for the ML model family

## Installation Guide

```bash
git clone https://github.com/<your-username>/supply-chain-demand-forecasting.git
cd supply-chain-demand-forecasting
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r app/requirements.txt
```

## How to Run Locally

1. Download the M5 dataset files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`.
2. Run the notebooks in `notebooks/` sequentially (01 → 09) in Google Colab or Jupyter — each notebook loads only the outputs of prior notebooks, so they can also be re-run independently once their inputs exist.
3. All processed outputs are saved to `data/processed/`; trained models are saved to `models/`.

## How to Launch the Streamlit Dashboard

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads only the small, precomputed CSVs generated by Notebook 09 — it does not require the raw M5 files or a GPU/heavy compute environment to run.

## Project Roadmap

This project followed a fixed, 33-artifact roadmap contract, treated as immutable once agreed — spanning data understanding, feature engineering, baseline/ML/classical modeling, model evaluation with SHAP explainability and detailed error analysis, inventory optimization, and dashboard deployment. See `docs/` for the full itemized roadmap and interview preparation notes mapped to each phase.

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. — *The M5 Competition: Background, Organization, and Implementation*
- [M5 Forecasting — Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- Hyndman, R.J., & Athanasopoulos, G. — *Forecasting: Principles and Practice*
- Lundberg, S.M., & Lee, S.I. — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Taylor, S.J., & Letham, B. — *Forecasting at Scale* (Prophet, Meta)

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Rajveer Singh** — M.Sc. Statistics, IIT (BHU)

[LinkedIn](#) &nbsp;•&nbsp; [Resume](#) &nbsp;•&nbsp; [Portfolio](#) &nbsp;•&nbsp; [GitHub](#)

</div>
