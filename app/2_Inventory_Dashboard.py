"""
Inventory Dashboard page (Roadmap Item #24) -- safety stock, reorder points, and risk tiers.

Loads inventory_policy_recommendations.csv (Roadmap Item #20), computed analytically in
Notebook 08 from the Random Forest forecast's error distribution -- this app does not
recompute any inventory formulas, it only visualizes the saved recommendations.
"""

import streamlit as st
import plotly.express as px

from utils import load_inventory_recommendations, get_category_options, get_store_options, to_csv_download

st.set_page_config(page_title="Inventory Dashboard", page_icon="📦", layout="wide")
st.title("📦 Inventory Optimization Dashboard")
st.caption("Safety stock, reorder points, and EOQ -- derived analytically from forecast uncertainty (Item #19)")

st.info(
    "This dataset does not include actual stock-on-hand levels. Safety stock, reorder point, "
    "and risk tier below are derived analytically from the forecast and its error distribution, "
    "using explicit business assumptions (7-day lead time, 95% service level) -- not from a live "
    "inventory feed. See Notebook 08 for the full methodology.",
    icon="ℹ️",
)

df = load_inventory_recommendations()

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")
categories = get_category_options(df, "cat_id")
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

stores = get_store_options(df, "store_id")
selected_stores = st.sidebar.multiselect("Store", stores, default=stores)

risk_tiers = sorted(df["risk_tier"].dropna().unique().tolist())
selected_risk_tiers = st.sidebar.multiselect("Risk Tier", risk_tiers, default=risk_tiers)

filtered_df = df[
    df["cat_id"].isin(selected_categories)
    & df["store_id"].isin(selected_stores)
    & df["risk_tier"].isin(selected_risk_tiers)
]

if len(filtered_df) == 0:
    st.warning("No products match the current filters. Adjust the selection in the sidebar.")
    st.stop()

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
st.markdown("### Inventory KPIs (Filtered Selection)")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Products in Selection", f"{len(filtered_df):,}")
kpi2.metric("Total Safety Stock (units)", f"{filtered_df['safety_stock'].sum():,.0f}")
kpi3.metric("Avg. Reorder Point", f"{filtered_df['reorder_point'].mean():,.1f}")
kpi4.metric("High Stockout-Risk Products", f"{(filtered_df['risk_tier'] == 'high_stockout_risk').sum():,}")

# ------------------------------------------------------------
# Safety stock by category
# ------------------------------------------------------------
st.markdown("### Total Safety Stock by Category")
safety_by_cat = filtered_df.groupby("cat_id", observed=True)["safety_stock"].sum().reset_index()
fig1 = px.bar(
    safety_by_cat.sort_values("safety_stock", ascending=True), x="safety_stock", y="cat_id",
    orientation="h", labels={"safety_stock": "Total Safety Stock (units)", "cat_id": "Category"},
    title="Total Safety Stock Required by Category",
)
fig1.update_layout(height=380)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------
# Risk tier distribution
# ------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Risk Tier Distribution")
    risk_counts = filtered_df["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["risk_tier", "count"]
    color_map = {
        "high_stockout_risk": "#C44E52", "moderate_risk": "#DD8452",
        "overstock_risk": "#4C72B0", "no_demand": "#8C8C8C",
    }
    fig2 = px.pie(
        risk_counts, names="risk_tier", values="count", color="risk_tier",
        color_discrete_map=color_map, title="Product-Store Count by Risk Tier",
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.markdown("### Reorder Point Distribution")
    fig3 = px.histogram(
        filtered_df, x="reorder_point", nbins=40,
        labels={"reorder_point": "Reorder Point (units)"},
        title="Distribution of Reorder Points",
    )
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# Reorder alert table
# ------------------------------------------------------------
st.markdown("### Reorder Alert List — Highest-Priority Products")
top_n = st.slider("Number of products to show", min_value=5, max_value=50, value=20, step=5)
alerts = filtered_df.sort_values("reorder_point", ascending=False).head(top_n)[
    ["store_id", "item_id", "cat_id", "avg_daily_demand_forecast", "safety_stock", "reorder_point",
     "eoq", "risk_tier"]
]
st.dataframe(alerts, use_container_width=True)

# ------------------------------------------------------------
# Downloadable output
# ------------------------------------------------------------
st.markdown("### Download Filtered Inventory Recommendations")
st.download_button(
    label="Download Filtered Inventory Data (CSV)",
    data=to_csv_download(filtered_df),
    file_name="filtered_inventory_recommendations.csv",
    mime="text/csv",
)
