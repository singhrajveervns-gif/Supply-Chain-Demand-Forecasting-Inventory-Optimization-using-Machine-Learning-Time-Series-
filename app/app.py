"""
Supply Chain Demand Forecasting & Inventory Optimization -- Dashboard Entry Point (Roadmap Item #22)

Landing page: project overview, headline KPI cards, and the full-history demand trend.
Detailed forecasting, inventory, and model-comparison views live in the pages/ folder
(Roadmap Items #23-25), following Streamlit's standard multipage app convention.
"""

import streamlit as st
import plotly.express as px

from utils import load_overview_stats, load_daily_trend

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Supply Chain Demand Forecasting & Inventory Optimization")
st.caption("M5 (Walmart) dataset, subsetted to CA_1 and TX_1 -- end-to-end forecasting and inventory policy")

st.markdown("""
This dashboard presents an end-to-end demand forecasting and inventory optimization pipeline,
built on real Walmart sales data (the M5 forecasting competition dataset). Every number shown
here comes from a saved, reproducible notebook pipeline -- no data is generated live by this app.

**Use the sidebar to navigate:**
- **Demand Forecast** -- actual vs. forecast by product, with category/store filters and search
- **Inventory Dashboard** -- safety stock, reorder points, and stockout/overstock risk
- **Model Insights** -- baseline vs. machine learning vs. classical model comparison
""")

stats = load_overview_stats()

st.markdown("### Project Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records Analyzed", f"{int(stats['total_rows']):,}")
col2.metric("Date Range (Days)", f"{int(stats['n_days']):,}")
col3.metric("Stores Covered", f"{int(stats['n_stores'])}")
col4.metric("Product Categories", f"{int(stats['n_categories'])}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Unique Products", f"{int(stats['n_items']):,}")
col6.metric("Best Model (Full Catalog)", stats['best_model'])
col7.metric("Improvement Over Baseline", f"{stats['best_model_improvement_pct']:.1f}%")
col8.metric("High-Risk Products (Stockout)", f"{int(stats['n_high_stockout_risk']):,}")

st.markdown("### Total Daily Demand — Full History")
daily_trend = load_daily_trend()

fig = px.line(
    daily_trend, x="date", y="units_sold",
    labels={"date": "Date", "units_sold": "Total Units Sold"},
    title="Total Daily Units Sold — CA_1 + TX_1 Combined (Full History)",
)
fig.update_layout(height=420, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
**Methodology summary:** demand was forecast using a Random Forest model, selected after
comparing 5 machine learning models, 2 baseline methods, and 2 classical time series models
(SARIMA, Prophet) across walk-forward cross-validation. Safety stock and reorder points were
then derived analytically from the winning model's forecast error distribution. See the
**Model Insights** page for the full comparison, and the **Inventory Dashboard** page for the
resulting reorder recommendations.
""")
